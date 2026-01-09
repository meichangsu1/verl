# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import (
    get_device_id,
    get_device_name,
)
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()
        # build device mesh for Ulysses Sequence Parallel

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

            module = auto_class.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            # Copy adapter to local if needed
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]
            # Ensure task_type is TaskType enum, not string
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        optimizer = build_optimizer(module.parameters(), self.optimizer_config)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only:
                    loss.backward()

            output_lst.append(meta_info)

        # postprocess and return
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate.
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        Note that this function executes irrespective of offload config. It serves as manual control
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            # force cpu_offload
            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
            )
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )
        return per_tensor_param, peft_config


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            # support per sample temperature
            # temperature (bsz,)
            # input_ids (bsz, j1)
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:
                    # vlm model's inputs will be sliced after embedding
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                        skip_position_ids_rmpad=True if self.__class__.__name__ == "VeOmniEngineWithLMHead" else False,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    temperature_rmpad, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size, pad_value=1
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                # we store the per sample temperature
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)  # (4, batch_size, max_seq_len)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                # temperature is singleton
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

            # gather log_prob if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits  # (bsz, response_length, vocab_size)
                temperature = output_args["temperature"]  # (bsz,)
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = output[2].squeeze(0).unsqueeze(-1)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz, 1)
                # critic model arch is like Qwen3ForTokenClassfication and num_labels=1
                # so we squeeze the last dimension here to get the value for each token
                values_rmpad = values_rmpad.squeeze(-1)

            # gather output if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values = output[2]
            else:
                values = output.logits

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}


@EngineRegistry.register(model_type="language_model_with_speculator", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHeadAndSpeculator(FSDPEngineWithLMHead):
    """
    Language model engine with a speculator for speculative decoding training.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self.speculator = None
        self.speculator_config = getattr(model_config, "speculator", None)

    def _build_module(self):
        import types
        import torch
        from typing import Optional
        
        module = super()._build_module()
        
        # Create speculator if config is provided
        if self.speculator_config is not None:
            # Get model dimensions
            hidden_size = module.config.hidden_size
            vocab_size = module.config.vocab_size
            
            # Create speculator config
            speculator_config_dict = {
                'n_predict': self.speculator_config.get('n_predict', 5),
                'input_hidden_dim': hidden_size,
                'inner_dim': str(self.speculator_config.get('inner_dim', hidden_size)),
                'emb_dim': str(self.speculator_config.get('emb_dim', hidden_size)),
                'proj_dim': str(self.speculator_config.get('proj_dim', hidden_size)),
                'vocab_size': vocab_size,
                'scale_input': self.speculator_config.get('scale_input', False),
                'tie_weights': self.speculator_config.get('tie_weights', False),
                'tie_lstm_embs': self.speculator_config.get('tie_lstm_embs', False),
                'method': self.speculator_config.get('method', 'sum_rnn'),
            }
            
            from verl.models.transformers.speculator import create_speculator_from_config
            self.speculator = create_speculator_from_config(speculator_config_dict)
            
            # Attach speculator as a submodule
            module.speculator = self.speculator
            
            # Store the original forward method
            module.old_forward = module.forward
            
            # Create a new forward method that supports speculator_return parameter
            def forward_with_speculator(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values=None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                speculator_return: bool = False,
            ):
                """Forward pass of the SpeculatorModel.
                Returns:
                    torch.Tensor: A tensor containing predictions from all Medusa heads.
                    (Optional) Original predictions from the base model's LM head.
                """

                if not speculator_return:
                    return self.old_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                # Pass input through the base model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                return outputs
            
            # Bind the new forward method to the module
            module.forward = types.MethodType(forward_with_speculator, module)
            
            # Freeze base model parameters when speculator is present
            # This ensures only speculator is trained during speculative decoding training
            # Freeze all parameters of the base model
            for param in module.parameters():
                param.requires_grad = False
                
            # Unfreeze speculator parameters
            for param in module.speculator.parameters():
                param.requires_grad = True
                
            # Move speculator to appropriate device and dtype
            module.speculator.to(module.dtype).to(module.device)
            
            # Reset speculator parameters
            module.speculator.reset_parameters()
            
        return module

    def prepare_model_inputs(self, micro_batch: TensorDict):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        # Ensure we get hidden states for speculator
        model_inputs["output_hidden_states"] = True
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        """
        Prepare model outputs for speculator training.
        
        For speculator training, we need both base model outputs (for base loss) 
        and speculator outputs (for speculator loss).
        """
        # Get base model outputs (log_probs, entropy, etc.)
        model_output = super().prepare_model_outputs(output, output_args, micro_batch)
        
        # If speculator is present, we'll compute speculator outputs separately in forward_step
        # This method is kept for compatibility but speculator outputs are computed in forward_step
        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            # If speculator is present, we need to compute speculator loss
            if self.speculator is not None and not forward_only:
                # First, get hidden states from the base model with no_grad
                with torch.no_grad():
                    raw_output = self.module(
                        **model_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    # Get the last hidden state
                    hidden_states = raw_output.hidden_states[-1]
                
                # Prepare inputs for speculator
                use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
                input_ids = micro_batch["input_ids"]
                n_predict = self.speculator.n_predict
                
                if use_remove_padding:
                    # hidden_states shape: (1, total_nnz, hidden_dim)
                    hidden_states = hidden_states.squeeze(0)  # (total_nnz, hidden_dim)
                    input_ids_flat = input_ids.values()  # (total_nnz,)
                    
                    # Slice hidden_states to remove last n_predict+1 tokens
                    # Since we have flattened sequence, we need to handle per-sample lengths
                    # For simplicity, we'll use the full hidden_states and adjust loss later
                    hidden_states = hidden_states.unsqueeze(0)  # (1, total_nnz, hidden_dim)
                    
                    # Create inds: input_ids shifted right by 1
                    pad = torch.zeros(n_predict, dtype=input_ids_flat.dtype, device=input_ids_flat.device)
                    inds = torch.cat([input_ids_flat, pad]).unsqueeze(0)  # (1, total_nnz + n_predict)
                else:
                    # Not using remove_padding
                    # hidden_states shape: (batch, seq_len, hidden_dim)
                    # Slice hidden_states to remove last n_predict+1 tokens
                    hidden_states = hidden_states[:, : -n_predict - 1, :]
                    # inds is input_ids shifted right by 1
                    inds = input_ids[:, 1:]  # (batch, seq_len-1)
                    # Pad with zeros on the right for extra n_predict tokens
                    batch_size, seq_len = inds.shape
                    pad = torch.zeros(batch_size, n_predict, dtype=inds.dtype, device=inds.device)
                    inds = torch.cat([inds, pad], dim=1)  # (batch, seq_len-1 + n_predict)
                
                # Call speculator forward
                spec_logits = self.speculator(hidden_states, inds)  # (n_predict, b, n, vocab_size)
                
                # Compute speculator loss
                pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
                loss_mask = micro_batch["loss_mask"]
                input_ids = micro_batch["input_ids"]
                
                spec_loss_total = 0.0
                for i in range(n_predict):
                    if use_remove_padding:
                        # spec_logits[i] shape: (1, total_nnz, vocab_size)
                        logits_i = spec_logits[i].squeeze(0)  # (total_nnz, vocab_size)
                        input_ids_flat = input_ids.values()
                        loss_mask_flat = loss_mask.values()
                        
                        # Targets are shifted by i+2
                        targets = torch.roll(input_ids_flat, shifts=-(i+2), dims=0)
                        # Shift loss mask by i+2
                        loss_mask_shifted = torch.roll(loss_mask_flat, shifts=-(i+2), dims=0)
                        
                        # Compute cross-entropy loss
                        ce_loss = torch.nn.functional.cross_entropy(
                            logits_i, 
                            targets, 
                            reduction='none'
                        )
                        # Mask out invalid positions
                        ce_loss_masked = ce_loss * loss_mask_shifted
                        # Sum over all tokens
                        head_loss = ce_loss_masked.sum() / loss_mask_shifted.sum().clamp(min=1)
                    else:
                        # Not using remove_padding
                        logits_i = spec_logits[i]  # (batch, n, vocab_size)
                        batch_size, n = logits_i.shape[:2]
                        
                        # Targets are labels[:, i+2 : n + i+2]
                        # We'll use input_ids as labels
                        start = i + 2
                        end = start + n
                        targets = input_ids[:, start:end]  # (batch, n)
                        loss_mask_i = loss_mask[:, start:end]
                        
                        # Compute cross-entropy loss
                        ce_loss = torch.nn.functional.cross_entropy(
                            logits_i.reshape(-1, logits_i.shape[-1]),
                            targets.reshape(-1),
                            reduction='none'
                        ).view(batch_size, -1)
                        ce_loss_masked = ce_loss * loss_mask_i
                        head_loss = ce_loss_masked.sum() / loss_mask_i.sum().clamp(min=1)
                    
                    spec_loss_total += head_loss / n_predict
                
                # Scale speculator loss by a coefficient (could be configurable)
                spec_coeff = getattr(self.model_config, "speculator_loss_coeff", 1.0)
                spec_loss = spec_coeff * spec_loss_total
                
                # Also compute base model loss (optional, could be zero if we only train speculator)
                # For now, we'll compute base model loss as well
                base_model_output = self.prepare_model_outputs(
                    output=raw_output, output_args=output_args, micro_batch=micro_batch
                )
                
                if loss_function is not None:
                    base_loss, metrics = loss_function(
                        model_output=base_model_output, 
                        data=micro_batch, 
                        dp_group=self.get_data_parallel_group()
                    )
                    # Combine base loss and speculator loss
                    total_loss = base_loss + spec_loss
                else:
                    total_loss = spec_loss
                    metrics = {}
                
                # Add speculator loss to metrics
                metrics["speculator_loss"] = spec_loss.detach().item()
                
                output = {
                    "model_output": {**base_model_output, "spec_logits": spec_logits},
                    "loss": total_loss.detach().item(),
                    "metrics": metrics,
                }
                
                return total_loss, output
            else:
                # No speculator or forward_only mode, use parent implementation
                return super().forward_step(micro_batch, loss_function, forward_only)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save speculator checkpoint only (similar to LoRA), not the base model.
        
        Args:
            local_path: Directory to save the speculator checkpoint
            hdfs_path: Optional HDFS path for distributed storage
            global_step: Current training step
            max_ckpt_to_keep: Maximum number of checkpoints to keep
        """
        import os
        import json
        import torch
        
        if self.speculator is None:
            logger.warning("No speculator found, skipping speculator checkpoint save.")
            return
        
        # Ensure save directory exists
        os.makedirs(local_path, exist_ok=True)
        
        # Get the speculator state dict
        # If speculator is wrapped in FSDP, we need to get the full state dict
        speculator_module = self.speculator
        if hasattr(self.module, 'speculator'):
            # The speculator is attached to the module
            speculator_module = self.module.speculator
        
        # Save the speculator state dict
        state_dict_path = os.path.join(local_path, "pytorch_model.bin")
        
        # Handle FSDP wrapping if applicable
        if hasattr(speculator_module, '_fsdp_wrapped_module'):
            # Speculator is wrapped in FSDP, need to get full state dict
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
            
            # Set state dict type to FULL_STATE_DICT
            FSDP.set_state_dict_type(
                speculator_module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
            
            # Get full state dict
            state_dict = speculator_module.state_dict()
            
            # Save only on rank 0
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, state_dict_path)
                logger.info(f"Saved speculator state dict to {state_dict_path}")
            
            # Reset state dict type
            FSDP.set_state_dict_type(
                speculator_module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        else:
            # Regular module, just save the state dict
            state_dict = speculator_module.state_dict()
            torch.save(state_dict, state_dict_path)
            logger.info(f"Saved speculator state dict to {state_dict_path}")
        
        # Save the speculator config
        config_path = os.path.join(local_path, "config.json")
        
        # Get config from speculator
        if hasattr(speculator_module, 'config'):
            config_dict = speculator_module.config.__dict__
        else:
            # Create config from speculator attributes
            config_dict = {
                'n_predict': speculator_module.n_predict,
                'input_hidden_dim': speculator_module.input_hidden_dim,
                'inner_dim': speculator_module.inner_dim,
                'emb_dim': speculator_module.emb_dim,
                'proj_dim': speculator_module.proj_dim,
                'vocab_size': speculator_module.vocab_size,
                'scale_input': speculator_module.scale_input,
                'tie_weights': speculator_module.tie_weights,
                'tie_lstm_embs': speculator_module.tie_lstm_embs,
                'method': speculator_module.method,
            }
        
        # Save config only on rank 0
        if torch.distributed.get_rank() == 0:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved speculator config to {config_path}")
        
        # Sync all processes
        torch.distributed.barrier()
        
        # If HDFS path is provided, upload the checkpoint
        if hdfs_path is not None:
            from verl.utils.fs import upload_to_hdfs
            upload_to_hdfs(local_path, hdfs_path)
        
        # Handle checkpoint rotation if max_ckpt_to_keep is specified
        if max_ckpt_to_keep is not None and torch.distributed.get_rank() == 0:
            from verl.utils.checkpoint.checkpoint_handler import rotate_checkpoints
            rotate_checkpoints(local_path, max_ckpt_to_keep)
        
        return state_dict_path, config_path
