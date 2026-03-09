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
import logging
import os
import time
from functools import partial
from typing import Any, Callable, ContextManager, Iterator, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_forward_fused_no_padding_fn, get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.trainer.speculators.engine_helpers import build_engine_spec_decode_bundle, resolve_spec_decode_config
from verl.trainer.speculators.strategy_interface import LossOutput, TargetRuntimeView
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    get_transformer_layer_offset,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        if mpu.is_initialized():
            return

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        if self.enable_routing_replay:
            override_transformer_config["enable_routing_replay"] = True

        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # In case of invalid overrides, we need to make sure some critical params are set correctly
            provider.params_dtype = self.param_dtype

            # Ensure dtype settings propagate to Megatron-Bridge/TE
            provider.fp16 = self.param_dtype == torch.float16
            provider.bf16 = self.param_dtype == torch.bfloat16

            # Pass distributed info
            provider.tensor_model_parallel_size = self.engine_config.tensor_model_parallel_size
            provider.pipeline_model_parallel_size = self.engine_config.pipeline_model_parallel_size
            provider.expert_model_parallel_size = self.engine_config.expert_model_parallel_size
            provider.expert_tensor_parallel_size = self.engine_config.expert_tensor_parallel_size
            provider.virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider.context_parallel_size = self.engine_config.context_parallel_size
            provider.sequence_parallel = self.engine_config.sequence_parallel

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            provider.attention_backend = AttnBackend.flash
            provider.variable_seq_lengths = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_router_load_balancing_type = "none"

            # Apply transformer config overrides
            for key, value in override_transformer_config.items():
                setattr(provider, key, value)

            provider.finalize()
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        # TODO: add more cases
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )

        self.is_value_model = is_value_model

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=is_value_model,  # actor is not value model
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
        )
        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=self.engine_config.override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(module, self.engine_config.dist_checkpointing_path, is_value_model=is_value_model)
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels are not supported for value models or when MTP is enabled in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        self._maybe_enable_fused_kernels()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=True,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(self.forward_step, postprocess_micro_batch_func=postprocess_micro_batch_func)

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )

        if enable_routing_replay:
            if self.engine_config.router_replay.mode in ["R3"]:
                RouterReplay.clear_global_indices()
                RouterReplay.clear_global_router_replay_action()

        if self.model_config.mtp.enable and self.is_mp_src_rank_with_outputs():
            # add mtp_losses
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if "metrics" not in losses_reduced[0]:
                losses_reduced[0]["metrics"] = {}
            losses_reduced[0]["metrics"].update(metrics)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            return output
        else:
            return {}

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        # when lora adapter only, we only load adapter weights when base sync is done, otherwise load all weights
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            # Only export adapter weights
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
        else:
            per_tensor_param = self.bridge.export_hf_weights(self.module)
            if non_merge_lora_sync:
                per_tensor_param = add_base_layer_suffix(
                    per_tensor_param, model_type=self.model_config.hf_config.model_type
                )
        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", [])

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        calculate_entropy = tu.get_non_tensor_data(data, key="calculate_entropy", default=False)

        log_prob = output["log_probs"]
        model_output = {"log_probs": log_prob}
        if calculate_entropy:
            entropy = output["entropy"]
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, batch_iter: Iterator[TensorDict], model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        loss_mask = model_inputs["loss_mask"]

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            set_router_replay_data(layers_topk_idx, None, self.tf_config, vp_rank)

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        from verl.models.mcore import get_mcore_forward_no_padding_fn

        if use_fused_kernels:
            if not self.engine_config.use_remove_padding:
                logger.warning_once(
                    "Fused kernels require `use_remove_padding=True` for Megatron engine. Falling back to non-fused."
                )
                use_fused_kernels = False
            elif isinstance(temperature, torch.Tensor):
                if temperature.numel() != 1:
                    logger.warning_once(
                        "Fused kernels do not support per-sample temperature. Falling back to non-fused."
                    )
                    use_fused_kernels = False
                else:
                    temperature_value = float(temperature.item())
            else:
                temperature_value = float(temperature)

        if use_fused_kernels:
            fused_forward_fn = get_mcore_forward_fused_no_padding_fn(self.model_config.hf_config)
            output = fused_forward_fn(
                model=model,
                input_ids=input_ids,
                labels=label,
                multi_modal_inputs=multi_modal_inputs,
                temperature=temperature_value,
                calculate_entropy=calculate_entropy,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
            )
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  # (bsz, j1)

            forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

            def logits_processor(logits, label, temperature):
                assert logits.shape[:2] == label.shape[:2]
                # avoid non-positive temperature such as padding
                temperature[temperature <= 0] = 1e-8
                assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
                logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
                ret = {}
                if calculate_entropy:
                    logits_bak = logits.clone()
                    # # disable the hint until the fused_kernel is optimized for triton>=3.3
                    # if torch.distributed.get_rank() == 0:
                    #     logger.warning_once(
                    #         "For memory-efficient computation, enable fused kernels via "
                    #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
                    #         "The current `clone()` operation ensures correctness but increases memory usage."
                    #     )
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits

                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "temperature": temperature, "loss_mask": loss_mask}

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format="thd" if self.engine_config.use_remove_padding else "bshd",
                enable_mtp=self.model_config.mtp.enable_train,
            )

        # Router replay: switch to backward replay mode for next backward pass
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch)

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="language_model_with_speculator", backend="megatron")
class MegatronEngineWithLMHeadAndSpeculator(MegatronEngineWithLMHead):
    """Megatron engine for spec decode training with unified strategy interface."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.spec_decode_cfg = resolve_spec_decode_config(self.model_config, require=False)
        self.spec_decode_strategy = None
        self.spec_decode_runtime_ctx = None
        self.draft_chunks: list[torch.nn.Module] = []
        self._draft_active_on_rank: bool = False
        self.draft_optimizer = None
        self.draft_lr_scheduler = None
        self._draft_ckpt_subdir = "speculator"
        self._last_draft_grad_norm: Optional[float] = None

    def initialize(self):
        self._bootstrap_spec_decode()
        super().initialize()
        if self.spec_decode_strategy is None:
            raise RuntimeError("Spec decode strategy bootstrap failed for Megatron engine.")
        if self.spec_decode_runtime_ctx is None:
            raise RuntimeError("Spec decode runtime context bootstrap failed for Megatron engine.")

        self._draft_active_on_rank = self._is_draft_active_on_rank()
        self._validate_pp_spec_decode_support()
        if self.spec_decode_runtime_ctx is not None:
            self.spec_decode_runtime_ctx.enable_draft_module = self._draft_active_on_rank

        target_model = self.module[-1] if isinstance(self.module, list) and len(self.module) > 0 else self.module
        self.spec_decode_strategy.initialize(
            target_model=target_model,
            spec_decode_cfg=self.spec_decode_cfg,
            runtime_ctx=self.spec_decode_runtime_ctx,
        )
        if self._draft_active_on_rank:
            self._wrap_draft_with_megatron_ddp()
        if not self.engine_config.forward_only and self._draft_active_on_rank:
            self._build_draft_optimizer()
        if self._draft_active_on_rank and (self._is_offload_param or self._is_offload_optimizer):
            self.to(
                device="cpu",
                model=self._is_offload_param,
                optimizer=self._is_offload_optimizer,
                grad=self._is_offload_param,
            )

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        return super().forward_backward_batch(data=data, loss_function=loss_function, forward_only=forward_only)

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        output = super().train_batch(data=data, loss_function=loss_function)
        if output is not None and self.is_mp_src_rank_with_outputs() and self._last_draft_grad_norm is not None:
            metrics = output.get("metrics", None) if isinstance(output, dict) else tu.get(output, "metrics", None)
            if isinstance(metrics, dict):
                metrics["draft_grad_norm"] = self._last_draft_grad_norm
        return output
    def _wrap_draft_with_megatron_ddp(self) -> None:
        from megatron.core import tensor_parallel
        from megatron.core.distributed import DistributedDataParallel as MCoreDDP
        from megatron.core.distributed import DistributedDataParallelConfig

        if self.spec_decode_strategy is None:
            raise RuntimeError("Spec decode strategy is not initialized.")
        draft_module = self.spec_decode_strategy.get_draft_module()
        if draft_module is None:
            raise RuntimeError("Spec decode strategy did not build draft module on the active pipeline rank.")

        for param in draft_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        if self.param_dtype in (torch.float16, torch.bfloat16):
            draft_module = draft_module.to(dtype=self.param_dtype)
        draft_module = draft_module.to(get_device_id())

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
        )
        draft_ddp = MCoreDDP(
            config=self.tf_config,
            module=draft_module,
            disable_bucketing=False,
            ddp_config=ddp_config,
        )
        draft_ddp.broadcast_params()
        self.draft_chunks = [draft_ddp]
        self.spec_decode_strategy.bind_draft_module(draft_ddp)

    def _build_draft_optimizer(self) -> None:
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        if self.spec_decode_strategy is None:
            raise RuntimeError("Spec decode strategy is not initialized.")
        if len(self.draft_chunks) == 0:
            raise RuntimeError("Draft module must be wrapped before building draft optimizer.")
        params = list(self.spec_decode_strategy.get_draft_trainable_params() or [])
        if len(params) == 0:
            raise RuntimeError(
                "Spec decode strategy returned no trainable params for draft model. "
                "Please ensure draft parameters require_grad=True."
            )
        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
        )
        self.draft_optimizer = get_megatron_optimizer(model=self.draft_chunks, config=optim_config_megatron)
        register_megatron_training_hooks(self.draft_chunks, self.draft_optimizer)
        self.draft_lr_scheduler = None

    def forward_step(self, batch_iter: Iterator[TensorDict], model, postprocess_micro_batch_func):
        if self._is_forward_only_postprocess(postprocess_micro_batch_func):
            return super().forward_step(
                batch_iter=batch_iter,
                model=model,
                postprocess_micro_batch_func=postprocess_micro_batch_func,
            )

        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for Megatron spec decode.")

        is_ppo_batch = self._is_ppo_training_batch(batch)
        calculate_entropy = bool(tu.get_non_tensor_data(batch, key="calculate_entropy", default=False))
        temperature = batch.get("temperature", None)

        hidden_layers, include_input_embeddings = self._resolve_target_signal_request()
        data_format = "thd" if self.engine_config.use_remove_padding else "bshd"
        output = self._forward_target_with_hidden(
            model,
            input_ids,
            multi_modal_inputs,
            hidden_layers=hidden_layers,
            include_input_embeddings=include_input_embeddings,
            data_format=data_format,
            compute_log_probs=is_ppo_batch,
            calculate_entropy=calculate_entropy,
            temperature=temperature,
            loss_mask=model_inputs.get("loss_mask", None),
        )

        is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
        if is_last_stage:
            output_tensor = output
            loss_func = partial(postprocess_micro_batch_func, data=batch)
        else:
            output_tensor = output["model_output"] if isinstance(output, dict) else output
            loss_func = None
        return output_tensor, loss_func
    def _resolve_target_signal_request(self) -> tuple[list[int], bool]:
        if self.spec_decode_strategy is None:
            raise RuntimeError("Spec decode strategy is not initialized.")

        signal_req = self.spec_decode_strategy.get_target_signal_request()
        hidden_layers = list(getattr(signal_req, "hidden_layers", [-1]) or [-1])
        include_input_embeddings = bool(getattr(signal_req, "include_input_embeddings", False))

        target_signals_cfg = {}
        if hasattr(self.spec_decode_cfg, "get"):
            target_signals_cfg = self.spec_decode_cfg.get("target_signals", {}) or {}
        if hasattr(target_signals_cfg, "get"):
            cfg_hidden_layers = target_signals_cfg.get("hidden_layers", None)
            if cfg_hidden_layers is not None:
                hidden_layers = [int(x) for x in cfg_hidden_layers]
            include_input_embeddings = bool(
                target_signals_cfg.get("include_input_embeddings", include_input_embeddings)
            )

        if not hidden_layers:
            hidden_layers = [-1]
        return hidden_layers, include_input_embeddings
    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        if forward_only:
            return super().postprocess_micro_batch_func(
                output=output,
                data=data,
                forward_only=forward_only,
                loss_function=loss_function,
            )

        device = data["input_ids"].device
        is_ppo_batch = self._is_ppo_training_batch(data)

        ppo_loss = torch.zeros((), device=device)
        ppo_metrics: dict[str, Any] = {}
        if is_ppo_batch:
            if loss_function is None:
                raise RuntimeError("Megatron PPO training batch requires a non-None loss_function.")
            raw_model_output = output.get("model_output", output) if isinstance(output, dict) else output
            ppo_model_output = self.prepare_model_outputs(raw_model_output, data)
            ppo_loss, ppo_metrics = loss_function(
                model_output=ppo_model_output,
                data=data,
                dp_group=self.get_data_parallel_group(),
            )
            ppo_metrics = self._prefix_metrics(dict(ppo_metrics), "ppo")

        spec_loss = torch.zeros((), device=device)
        spec_metrics: dict[str, Any] = {"spec/speculator_loss": 0.0}
        if self._draft_active_on_rank:
            target_view = self._build_target_runtime_view(micro_batch=data, raw_output=output)
            loss_output = self._compute_spec_decode_loss(target_view=target_view)
            spec_loss = loss_output.total_loss
            if not torch.is_tensor(spec_loss):
                spec_loss = torch.tensor(float(spec_loss), device=device)
            spec_metrics = self._prefix_metrics(dict(loss_output.metrics), "spec")

        loss = ppo_loss + self._extract_spec_loss_coef() * spec_loss
        if loss.ndim > 0:
            loss = loss.mean()

        # Keep a zero-valued autograd edge to target pipeline output so PP backward
        # communication remains well-formed even when spec loss is independent from
        # target model output activations.
        pipeline_output_tensor = None
        if isinstance(output, dict):
            candidate = output.get("model_output", None)
            if torch.is_tensor(candidate):
                pipeline_output_tensor = candidate
        elif torch.is_tensor(output):
            pipeline_output_tensor = output
        if torch.is_tensor(pipeline_output_tensor):
            loss = loss + pipeline_output_tensor.sum() * 0.0

        scaled_loss = loss * data["num_micro_batch"]

        metrics: dict[str, Any] = {}
        metrics.update(ppo_metrics)
        metrics.update(spec_metrics)
        model_output: dict[str, Any] = {}
        output = {
            "model_output": model_output,
            "loss": float(loss.detach().item()),
            "metrics": metrics,
        }
        return scaled_loss, output


    def _forward_target_with_hidden(
        self,
        model,
        input_ids,
        multi_modal_inputs: dict[str, Any],
        *,
        hidden_layers: list[int],
        include_input_embeddings: bool,
        data_format: str,
        compute_log_probs: bool = False,
        calculate_entropy: bool = False,
        temperature: Optional[Any] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del include_input_embeddings
        from verl.models.mcore import get_mcore_forward_no_padding_fn
        from verl.models.mcore.util import (
            postprocess_bshd_no_padding,
            postprocess_thd_no_padding,
            preprocess_bshd_no_padding,
            preprocess_thd_no_padding,
        )

        unwrapped_model = unwrap_model(model)
        decoder_layers = self._get_decoder_layers(unwrapped_model)
        layer_global_ids = self._build_layer_global_ids(unwrapped_model, decoder_layers)
        total_layers = self._infer_total_hidden_layers(unwrapped_model)
        requested_pairs = self._normalize_requested_hidden_layer_pairs(hidden_layers, total_layers=total_layers)
        requested_abs_sorted = sorted({abs_layer_id for _, abs_layer_id in requested_pairs})

        if not decoder_layers:
            raise RuntimeError(
                "Megatron spec decode cannot capture hidden states because target decoder layers are not discoverable."
            )

        forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)
        logger.warning_once("Megatron target hidden extraction is forced to layer forward hooks in this build.")
        captured_local_hidden: dict[int, torch.Tensor] = {}
        hook_handles = []
        for layer, global_layer_id in zip(decoder_layers, layer_global_ids, strict=True):
            if global_layer_id not in requested_abs_sorted:
                continue

            def _hook(_module, _inputs, hook_output, *, _global_layer_id=global_layer_id):
                hidden = self._extract_hidden_from_hook_output(hook_output)
                if hidden is not None:
                    captured_local_hidden[_global_layer_id] = hidden

            hook_handles.append(layer.register_forward_hook(_hook))

        try:
            if compute_log_probs:
                if temperature is None:
                    raise RuntimeError("PPO training batch requires `temperature` in Megatron spec engine.")

                if not isinstance(temperature, torch.Tensor):
                    temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

                temperature = temperature.to(torch.float32)
                assert temperature.shape[0] == input_ids.shape[0], (
                    f"temperature batch size mismatch: {temperature.shape[0]} vs input {input_ids.shape[0]}"
                )
                temperature = verl_F.expand_as_nested(temperature, input_ids)

                label = input_ids.clone()

                def logits_processor(logits, label, temperature):
                    assert logits.shape[:2] == label.shape[:2]
                    temperature = temperature.clone()
                    temperature[temperature <= 0] = 1e-8
                    logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        ret["entropy"] = vocab_parallel_entropy(logits)
                    else:
                        logits_bak = logits
                    ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    return ret

                logits_processor_args = {"label": label, "temperature": temperature}
                if loss_mask is not None:
                    logits_processor_args["loss_mask"] = loss_mask

                model_output = forward_fn(
                    model,
                    input_ids,
                    multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                    data_format=data_format,
                    enable_mtp=self.model_config.mtp.enable_train,
                )
            else:
                model_output = forward_fn(
                    model,
                    input_ids,
                    multi_modal_inputs,
                    vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                    data_format=data_format,
                    enable_mtp=self.model_config.mtp.enable_train,
                )
        finally:
            for handle in hook_handles:
                handle.remove()

        captured_hidden_by_abs: dict[int, torch.Tensor] = {
            int(abs_layer_id): hidden
            for abs_layer_id, hidden in captured_local_hidden.items()
            if torch.is_tensor(hidden)
        }

        # Keep pipeline payload as tensor on non-last stages.
        pipeline_model_output = model_output
        if isinstance(model_output, (tuple, list)) and model_output:
            first = model_output[0]
            if torch.is_tensor(first):
                pipeline_model_output = first

        pre_process = bool(getattr(unwrapped_model, "pre_process", True))
        if data_format == "thd":
            _, packed_seq_params = preprocess_thd_no_padding(input_ids, pre_process=pre_process)
            postprocess_hidden = lambda hidden: postprocess_thd_no_padding(
                hidden,
                packed_seq_params,
                input_ids,
                batch_size=input_ids.shape[0],
                post_process=True,
            )
        else:
            _, attention_mask_bshd, _ = preprocess_bshd_no_padding(input_ids, pre_process=pre_process)
            postprocess_hidden = lambda hidden: postprocess_bshd_no_padding(
                hidden,
                attention_mask_bshd,
                post_process=True,
            )

        processed_hidden_by_abs: dict[int, torch.Tensor] = {}
        for global_layer_id, raw_hidden in captured_hidden_by_abs.items():
            normalized_hidden = self._normalize_hidden_for_postprocess(
                raw_hidden,
                data_format=data_format,
                batch_size=input_ids.shape[0],
            )
            if normalized_hidden is None:
                continue
            processed_hidden = postprocess_hidden(normalized_hidden)
            # Teacher hidden states are always consumed as detached draft inputs.
            processed_hidden_by_abs[global_layer_id] = self._as_dense_hidden_tensor(processed_hidden).detach()

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group() if tp_world_size > 1 else None
        expected_seq_len = self._infer_expected_seq_len_from_input_ids(input_ids)

        owner_hidden_by_abs: dict[int, torch.Tensor] = {}
        for abs_layer_id in requested_abs_sorted:
            local_hidden = processed_hidden_by_abs.get(abs_layer_id, None)
            local_has_hidden = int(torch.is_tensor(local_hidden))
            presence_device = local_hidden.device if torch.is_tensor(local_hidden) else input_ids.device

            if tp_world_size > 1:
                has_tensor = torch.tensor([local_has_hidden], dtype=torch.int32, device=presence_device)
                has_min = has_tensor.clone()
                has_max = has_tensor.clone()
                torch.distributed.all_reduce(has_min, op=torch.distributed.ReduceOp.MIN, group=tp_group)
                torch.distributed.all_reduce(has_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
                if int(has_min.item()) != int(has_max.item()):
                    raise RuntimeError(
                        "Megatron spec decode found inconsistent hidden capture across TP ranks. "
                        f"abs_layer_id={abs_layer_id}, local_has_hidden={bool(local_has_hidden)}."
                    )
                if int(has_max.item()) == 0:
                    raise RuntimeError(
                        "Megatron spec decode failed to capture requested target hidden state on any TP rank. "
                        f"abs_layer_id={abs_layer_id}."
                    )
            elif not local_has_hidden:
                raise RuntimeError(
                    "Megatron spec decode failed to capture requested target hidden state. "
                    f"abs_layer_id={abs_layer_id}."
                )

            if not torch.is_tensor(local_hidden):
                continue
            owner_hidden = self._gather_hidden_to_draft_owner_tp(local_hidden, expected_seq_len=expected_seq_len)
            if torch.is_tensor(owner_hidden):
                # Teacher hidden states are always consumed as detached draft inputs.
                owner_hidden_by_abs[abs_layer_id] = owner_hidden.detach()

        hidden_states_map: dict[int, torch.Tensor] = {}
        for layer_id, abs_layer_id in requested_pairs:
            hidden = owner_hidden_by_abs.get(abs_layer_id, None)
            if torch.is_tensor(hidden):
                hidden_states_map[layer_id] = hidden

        return {
            "model_output": pipeline_model_output,
            "hidden_states_map": hidden_states_map,
        }

    def _build_target_runtime_view(self, micro_batch: TensorDict, raw_output) -> TargetRuntimeView:
        def _to_padded(tensor, pad_value):
            if tensor is None or not torch.is_tensor(tensor):
                return tensor
            if tensor.is_nested:
                return torch.nested.to_padded_tensor(tensor, padding=pad_value)
            return tensor

        def _get_hidden_states(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get("hidden_states", None)
            return getattr(obj, "hidden_states", None)

        hidden_layers, include_input_embeddings = self._resolve_target_signal_request()
        raw_model_output = raw_output.get("model_output", None) if isinstance(raw_output, dict) else None

        pad_token_id = self.model_config.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = _to_padded(micro_batch.get("input_ids", None), pad_token_id)
        labels = _to_padded(micro_batch.get("labels", None), -100)
        if labels is None:
            labels = input_ids
        loss_mask = _to_padded(micro_batch.get("loss_mask", None), 0)
        position_ids = _to_padded(micro_batch.get("position_ids", None), 0)
        attention_mask = _to_padded(micro_batch.get("attention_mask", None), 0)
        if attention_mask is None and torch.is_tensor(loss_mask):
            attention_mask = loss_mask

        raw_hidden_states_map = None
        if isinstance(raw_output, dict):
            raw_hidden_states_map = raw_output.get("hidden_states_map", None)

        raw_hidden_states = _get_hidden_states(raw_output)
        if raw_hidden_states is None:
            raw_hidden_states = _get_hidden_states(raw_model_output)

        hidden_by_layer: dict[int, Any] = {}
        if isinstance(raw_hidden_states_map, dict):
            for layer_id_raw, hidden_tensor in raw_hidden_states_map.items():
                if hidden_tensor is None:
                    continue
                if not torch.is_tensor(hidden_tensor):
                    continue
                layer_id = int(layer_id_raw)
                hidden_by_layer[layer_id] = _to_padded(hidden_tensor, 0)

        if isinstance(raw_hidden_states, (tuple, list)):
            n_hidden = len(raw_hidden_states)
            for layer_id in hidden_layers:
                idx = layer_id if layer_id >= 0 else n_hidden + layer_id
                if idx < 0 or idx >= n_hidden:
                    continue
                hidden_tensor = raw_hidden_states[idx]
                if torch.is_tensor(hidden_tensor):
                    hidden_by_layer[layer_id] = _to_padded(hidden_tensor, 0)
        elif torch.is_tensor(raw_hidden_states):
            hidden_by_layer[-1] = _to_padded(raw_hidden_states, 0)

        missing_hidden_layers = [layer_id for layer_id in hidden_layers if layer_id not in hidden_by_layer]
        if missing_hidden_layers:
            available = sorted(hidden_by_layer.keys())
            raise RuntimeError(
                "Megatron spec decode target hidden states are incomplete. "
                f"requested={hidden_layers}, missing={missing_hidden_layers}, available={available}. "
                "Please verify target_signals.hidden_layers. "
                "Megatron spec decode training currently supports pipeline parallel size 1 only."
            )

        input_embeddings = None
        if include_input_embeddings and torch.is_tensor(input_ids):
            raw_input_embeddings = None
            if isinstance(raw_output, dict):
                raw_input_embeddings = raw_output.get("input_embeddings", raw_output.get("inputs_embeds", None))
            else:
                raw_input_embeddings = getattr(raw_output, "input_embeddings", None)
                if raw_input_embeddings is None:
                    raw_input_embeddings = getattr(raw_output, "inputs_embeds", None)
            if raw_input_embeddings is None:
                if isinstance(raw_model_output, dict):
                    raw_input_embeddings = raw_model_output.get(
                        "input_embeddings", raw_model_output.get("inputs_embeds", None)
                    )
                else:
                    raw_input_embeddings = getattr(raw_model_output, "input_embeddings", None)
                    if raw_input_embeddings is None:
                        raw_input_embeddings = getattr(raw_model_output, "inputs_embeds", None)

            if torch.is_tensor(raw_input_embeddings):
                input_embeddings = raw_input_embeddings
            else:
                try:
                    target_embed_module = None
                    target_model = (
                        self.spec_decode_strategy.target_model
                        if hasattr(self.spec_decode_strategy, "target_model")
                        else self.module[-1]
                    )
                    unwrapped_target = unwrap_model(target_model)
                    if hasattr(unwrapped_target, "get_input_embeddings"):
                        target_embed_module = unwrapped_target.get_input_embeddings()
                    if target_embed_module is None:
                        embedding = getattr(unwrapped_target, "embedding", None)
                        target_embed_module = getattr(embedding, "word_embeddings", None)
                    if target_embed_module is not None:
                        input_embeddings = target_embed_module(input_ids)
                except Exception:
                    input_embeddings = None
            if input_embeddings is None:
                raise RuntimeError(
                    "Megatron spec decode requested input embeddings, but they are unavailable on this pipeline stage. "
                    "For pipeline parallel > 1, embedding reuse requires explicit cross-stage embedding forwarding."
                )

        hidden_by_layer = {layer_id: hidden.detach() for layer_id, hidden in hidden_by_layer.items()}
        if torch.is_tensor(input_embeddings):
            input_embeddings = input_embeddings.detach()

        phase = tu.get_non_tensor_data(micro_batch, key="phase", default=self.mode)
        backend_payload = {"phase": phase}
        return TargetRuntimeView(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            labels=labels,
            hidden_by_layer=hidden_by_layer,
            input_embeddings=input_embeddings,
            packed_seq_params=None,
            raw_output=raw_output,
            backend_payload=backend_payload,
        )
    def _compute_spec_decode_loss(self, target_view: TargetRuntimeView) -> LossOutput:
        if self.spec_decode_strategy is None:
            raise RuntimeError("Spec decode strategy is not initialized.")
        if not self._draft_active_on_rank:
            zero = torch.zeros((), device=target_view.input_ids.device)
            return LossOutput(total_loss=zero, metrics={"speculator_loss": 0.0})
        draft_module = self.spec_decode_strategy.get_draft_module()
        if draft_module is not None:
            try:
                draft_param = next(draft_module.parameters())
                expected_device = None
                if torch.is_tensor(target_view.input_ids):
                    expected_device = target_view.input_ids.device
                elif target_view.hidden_by_layer:
                    hidden = next(iter(target_view.hidden_by_layer.values()))
                    if torch.is_tensor(hidden):
                        expected_device = hidden.device
                if expected_device is not None and draft_param.device != expected_device:
                    draft_module.to(expected_device)
            except StopIteration:
                pass

        loss_output = self.spec_decode_strategy.compute_step_loss(target_view=target_view)
        if not isinstance(loss_output, LossOutput):
            raise TypeError(
                f"Spec decode strategy must return LossOutput, got {type(loss_output).__name__}."
            )
        return loss_output

    def _bootstrap_spec_decode(self) -> None:
        from verl.utils.torch_dtypes import PrecisionType

        runtime_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        bundle = build_engine_spec_decode_bundle(
            model_config=self.model_config,
            backend="megatron",
            torch_dtype=runtime_dtype,
            supports_packed_seq=False,
            require=True,
        )
        self.spec_decode_cfg = bundle.config
        self.spec_decode_strategy = bundle.strategy
        self.spec_decode_runtime_ctx = bundle.runtime_context

    def _is_draft_active_on_rank(self) -> bool:
        try:
            is_last_stage = bool(mpu.is_pipeline_last_stage(ignore_virtual=True))
        except TypeError:
            is_last_stage = bool(mpu.is_pipeline_last_stage())
        if not is_last_stage:
            return False
        return bool(mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0)

    def _is_draft_checkpoint_io_rank(self) -> bool:
        if not self._draft_active_on_rank:
            return False
        return mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0

    @staticmethod
    def _hidden_seq_dim(hidden: torch.Tensor) -> int:
        # Hidden states are typically [B, S, H], but some paths may emit [S, H].
        return 1 if hidden.ndim >= 3 else 0

    @staticmethod
    def _infer_expected_seq_len_from_input_ids(input_ids: Any) -> Optional[int]:
        if not torch.is_tensor(input_ids):
            return None

        if input_ids.ndim >= 2:
            try:
                return int(input_ids.shape[1])
            except Exception:
                pass

        offsets_fn = getattr(input_ids, "offsets", None)
        if callable(offsets_fn):
            try:
                offsets = offsets_fn()
                if torch.is_tensor(offsets) and offsets.numel() > 1:
                    return int(offsets.diff().max().item())
            except Exception:
                pass
        return None

    def _gather_hidden_to_draft_owner_tp(
        self,
        hidden: torch.Tensor,
        *,
        expected_seq_len: Optional[int],
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(hidden):
            return None
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size <= 1:
            return hidden

        is_draft_owner = bool(mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0)
        if not self.engine_config.sequence_parallel:
            return hidden if is_draft_owner else None

        seq_dim = self._hidden_seq_dim(hidden)
        gather_needed = True
        if expected_seq_len is not None and hidden.shape[seq_dim] >= expected_seq_len:
            # Some kernels already materialize full sequence activations even when SP is enabled.
            gather_needed = False

        tp_group = mpu.get_tensor_model_parallel_group()
        need_tensor = torch.tensor([int(gather_needed)], dtype=torch.int32, device=hidden.device)
        need_min = need_tensor.clone()
        need_max = need_tensor.clone()
        torch.distributed.all_reduce(need_min, op=torch.distributed.ReduceOp.MIN, group=tp_group)
        torch.distributed.all_reduce(need_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        if int(need_min.item()) != int(need_max.item()):
            raise RuntimeError(
                "Megatron spec decode found inconsistent TP hidden layout across ranks under sequence_parallel. "
                f"local_shape={tuple(hidden.shape)}, expected_seq_len={expected_seq_len}."
            )
        gather_needed = bool(int(need_max.item()))
        if not gather_needed:
            return hidden if is_draft_owner else None

        gathered = [torch.empty_like(hidden) for _ in range(tp_world_size)]
        torch.distributed.all_gather(gathered, hidden, group=tp_group)
        if not is_draft_owner:
            return None
        owner_hidden = torch.cat(gathered, dim=seq_dim).contiguous()
        if expected_seq_len is not None and owner_hidden.shape[seq_dim] > expected_seq_len:
            owner_hidden = owner_hidden.narrow(dim=seq_dim, start=0, length=expected_seq_len).contiguous()
        return owner_hidden

   
    def optimizer_zero_grad(self):
        super().optimizer_zero_grad()
        if self.draft_optimizer is not None:
            self.draft_optimizer.zero_grad()
            for chunk in self.draft_chunks:
                chunk.zero_grad_buffer()

    def optimizer_step(self):
        grad_norm = super().optimizer_step()
        self._last_draft_grad_norm = None
        if self.draft_optimizer is not None:
            update_successful, draft_grad_norm, _ = self.draft_optimizer.step()
            if not update_successful:
                raise RuntimeError("Draft Megatron optimizer step failed.")
            if torch.is_tensor(draft_grad_norm):
                self._last_draft_grad_norm = float(draft_grad_norm.detach().item())
            elif draft_grad_norm is not None:
                self._last_draft_grad_norm = float(draft_grad_norm)
        return grad_norm

    def lr_scheduler_step(self):
        current_lr = super().lr_scheduler_step()
        if self.draft_lr_scheduler is not None:
            self.draft_lr_scheduler.step()
        return current_lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if len(self.draft_chunks) == 0:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.draft_chunks, load_grad=grad)
            if optimizer and self.draft_optimizer is not None:
                load_megatron_optimizer(self.draft_optimizer)
        else:
            if model:
                offload_megatron_model_to_cpu(self.draft_chunks)
            if optimizer and self.draft_optimizer is not None:
                offload_megatron_optimizer(self.draft_optimizer)

    def _save_draft_checkpoint(self, local_path: str) -> None:
        if self.spec_decode_strategy is None or not self._is_draft_checkpoint_io_rank():
            return
        draft_module = self.spec_decode_strategy.get_draft_module()
        if draft_module is None:
            return
        draft_module = unwrap_model(draft_module)
        speculator_dir = os.path.join(local_path, self._draft_ckpt_subdir)
        os.makedirs(speculator_dir, exist_ok=True)
        torch.save(draft_module.state_dict(), os.path.join(speculator_dir, "pytorch_model.bin"))
        draft_config = self.spec_decode_strategy.get_draft_config_obj()
        if draft_config is not None and hasattr(draft_config, "to_json_file"):
            draft_config.to_json_file(os.path.join(speculator_dir, "config.json"))
        optimizer_payload: dict[str, Any] = {}
        if self.draft_optimizer is not None:
            optimizer_payload["optimizer"] = self.draft_optimizer.state_dict()
        if self.draft_lr_scheduler is not None:
            optimizer_payload["lr_scheduler"] = self.draft_lr_scheduler.state_dict()
        if optimizer_payload:
            torch.save(optimizer_payload, os.path.join(speculator_dir, "optimizer.pt"))

    def _load_draft_checkpoint(self, local_path: str) -> None:
        if self.spec_decode_strategy is None or not self._draft_active_on_rank:
            return
        draft_module = self.spec_decode_strategy.get_draft_module()
        if draft_module is None:
            return
        draft_module = unwrap_model(draft_module)
        speculator_path = os.path.join(local_path, self._draft_ckpt_subdir, "pytorch_model.bin")
        if not os.path.exists(speculator_path):
            return
        state_dict = torch.load(speculator_path, map_location="cpu", weights_only=False)
        draft_module.load_state_dict(state_dict, strict=False)
        optimizer_path = os.path.join(local_path, self._draft_ckpt_subdir, "optimizer.pt")
        if not os.path.exists(optimizer_path):
            return
        optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        if not isinstance(optimizer_payload, dict):
            logger.warning("Skip loading draft optimizer checkpoint because payload is not a mapping.")
            return
        optimizer_state = optimizer_payload.get("optimizer", None)
        if self.draft_optimizer is not None and isinstance(optimizer_state, dict):
            self.draft_optimizer.load_state_dict(optimizer_state)
        lr_scheduler_state = optimizer_payload.get("lr_scheduler", None)
        if self.draft_lr_scheduler is not None and isinstance(lr_scheduler_state, dict):
            self.draft_lr_scheduler.load_state_dict(lr_scheduler_state)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._save_draft_checkpoint(local_path=local_path)
        super().save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            **kwargs,
        )
        torch.distributed.barrier()

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: bool = True,
        **kwargs,
    ) -> None:
        super().load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
            **kwargs,
        )
        self._load_draft_checkpoint(local_path=local_path)
        torch.distributed.barrier()



    @staticmethod
    def _extract_hidden_from_hook_output(hook_output: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(hook_output):
            return hook_output
        if isinstance(hook_output, (tuple, list)):
            for item in hook_output:
                if torch.is_tensor(item):
                    return item
            return None
        if isinstance(hook_output, dict):
            candidate = hook_output.get("hidden_states", hook_output.get("last_hidden_state", None))
            if torch.is_tensor(candidate):
                return candidate
            if isinstance(candidate, (tuple, list)):
                for item in candidate:
                    if torch.is_tensor(item):
                        return item
            return None
        candidate = getattr(hook_output, "hidden_states", None)
        if torch.is_tensor(candidate):
            return candidate
        if isinstance(candidate, (tuple, list)):
            for item in candidate:
                if torch.is_tensor(item):
                    return item
        candidate = getattr(hook_output, "last_hidden_state", None)
        if torch.is_tensor(candidate):
            return candidate
        return None

    @staticmethod
    def _resolve_abs_layer_id(layer_id: int, total_layers: Optional[int]) -> Optional[int]:
        if layer_id >= 0:
            return layer_id
        if total_layers is None:
            return None
        abs_id = total_layers + layer_id
        if abs_id < 0 or abs_id >= total_layers:
            return None
        return abs_id

    def _normalize_requested_hidden_layer_pairs(
        self,
        hidden_layers: list[int],
        *,
        total_layers: Optional[int],
    ) -> list[tuple[int, int]]:
        requested_pairs: list[tuple[int, int]] = []
        for layer_id_raw in hidden_layers:
            layer_id = int(layer_id_raw)
            abs_layer_id = self._resolve_abs_layer_id(layer_id, total_layers)
            if abs_layer_id is None:
                if layer_id < 0:
                    raise RuntimeError(
                        "Megatron spec decode cannot resolve negative hidden layer indices without the total "
                        f"number of transformer layers. requested layer={layer_id}, total_layers={total_layers}."
                    )
                raise RuntimeError(
                    "Megatron spec decode received an invalid hidden layer request. "
                    f"requested layer={layer_id}, total_layers={total_layers}."
                )
            requested_pairs.append((layer_id, abs_layer_id))
        return requested_pairs

    def _validate_pp_spec_decode_support(self) -> None:
        pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_world_size <= 1:
            return

        raise RuntimeError(
            "Megatron spec decode training does not support pipeline parallel > 1. "
            f"current_pp_world_size={pp_world_size}. Please set engine.pipeline_model_parallel_size=1."
        )

    @staticmethod
    def _normalize_hidden_for_postprocess(
        tensor: torch.Tensor,
        *,
        data_format: str,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(tensor):
            return None
        if data_format == "thd":
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    return tensor
                if tensor.shape[1] == 1:
                    return tensor.transpose(0, 1).contiguous()
                return tensor
            if tensor.ndim == 2:
                return tensor.unsqueeze(0)
            return None
        if data_format == "bshd":
            if tensor.ndim == 3:
                if tensor.shape[0] == batch_size:
                    return tensor
                if tensor.shape[1] == batch_size:
                    return tensor.transpose(0, 1).contiguous()
                return tensor
            if tensor.ndim == 2 and batch_size == 1:
                return tensor.unsqueeze(0)
            return None
        return None

    @staticmethod
    def _as_dense_hidden_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected tensor for hidden states, got {type(tensor).__name__}.")
        if tensor.is_nested:
            return torch.nested.to_padded_tensor(tensor, padding=0.0)
        return tensor.contiguous()

    @staticmethod
    def _get_decoder_layers(unwrapped_model) -> list[torch.nn.Module]:
        candidates = [unwrapped_model]
        language_model = getattr(unwrapped_model, "language_model", None)
        if language_model is not None:
            candidates.append(language_model)
        inner_model = getattr(unwrapped_model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)

        for candidate in candidates:
            decoder = getattr(candidate, "decoder", None)
            layers = getattr(decoder, "layers", None)
            if layers is not None:
                return list(layers)
        return []

    def _infer_total_hidden_layers(self, unwrapped_model) -> Optional[int]:
        config = getattr(unwrapped_model, "config", None)
        for attr in ("num_layers", "num_hidden_layers"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        hf_config = getattr(self.model_config, "hf_config", None)
        for attr in ("num_hidden_layers", "num_layers"):
            value = getattr(hf_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _build_layer_global_ids(self, unwrapped_model, decoder_layers: list[torch.nn.Module]) -> list[int]:
        offset = None
        try:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            vp_stage = getattr(unwrapped_model, "vp_stage", None)
            offset = get_transformer_layer_offset(pp_rank, vp_stage, self.tf_config)
        except Exception:
            offset = None

        global_ids: list[int] = []
        for local_idx, layer in enumerate(decoder_layers):
            layer_number = getattr(layer, "layer_number", None)
            if isinstance(layer_number, int) and layer_number > 0:
                global_ids.append(layer_number - 1)
                continue

            layer_idx = getattr(layer, "layer_idx", None)
            if isinstance(layer_idx, int) and layer_idx >= 0:
                if offset is not None and layer_idx < len(decoder_layers):
                    global_ids.append(offset + layer_idx)
                else:
                    global_ids.append(layer_idx)
                continue

            if offset is not None:
                global_ids.append(offset + local_idx)
            else:
                global_ids.append(local_idx)
        return global_ids


    @staticmethod
    def _is_forward_only_postprocess(postprocess_micro_batch_func) -> bool:
        if isinstance(postprocess_micro_batch_func, partial):
            return bool(postprocess_micro_batch_func.keywords.get("forward_only", False))
        return False

    @staticmethod
    def _is_ppo_training_batch(batch: TensorDict) -> bool:
        required_keys = {"old_log_probs", "advantages", "responses", "response_mask"}
        return all(key in batch.keys() for key in required_keys)

    def _extract_spec_loss_coef(self) -> float:
        default_coef = 1.0
        if self.spec_decode_cfg is None:
            return default_coef
        loss_cfg = self.spec_decode_cfg.get("loss", {})
        if not hasattr(loss_cfg, "get"):
            return default_coef
        try:
            return float(loss_cfg.get("coef", default_coef))
        except (TypeError, ValueError):
            return default_coef

    @staticmethod
    def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_forward_no_padding_fn

        forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            enable_mtp=self.model_config.mtp.enable_train,
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
