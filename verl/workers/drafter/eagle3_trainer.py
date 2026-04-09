import asyncio
import logging
import os
import time
from collections import deque
from concurrent.futures import Future
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.data_buffer import DataBuffer
from verl.utils.device import is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import (
    get_fsdp_full_state_dict,
    get_device_id,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.ulysses import ulysses_pad_and_slice_inputs

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _completed_future():
    future = Future()
    future.set_result(None)
    return future


class EAGLE3BackgroundTrainer:
    """FSDP2-capable background trainer for online Eagle3 drafter training."""

    def __init__(
        self,
        drafter_module_fsdp,
        drafter_optimizer,
        drafter_lr_scheduler,
        drafter_train_config,
        drafter_device_mesh,
        model_config=None,
        base_model_lm_head=None,
    ):
        self.model = drafter_module_fsdp
        self.optimizer = drafter_optimizer
        self.lr_scheduler = drafter_lr_scheduler
        self.config = drafter_train_config
        self.training_device_mesh = drafter_device_mesh
        self.model_config = model_config
        self.base_model_lm_head = base_model_lm_head

        self.is_offload_param = self.config.get("is_offload_param", False)
        self.is_offload_optimizer = self.config.get("is_offload_optimizer", False)
        self.checkpoint_dir = self.config.get("checkpoint_path")

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(
            f"[EAGLE3Trainer rank {self.rank}] initialized with "
            f"base_model_lm_head={'present' if self.base_model_lm_head is not None else 'missing'}, "
            f"checkpoint_path={self.checkpoint_dir}"
        )

        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0

        self.collected_data = deque(maxlen=int(self.config.get("buffer_max_samples", 2000)))
        self.shared_data_buffer = None
        self.batch_size = int(self.config.get("batch_size_per_gpu", 32))

        buffer_max_size = int(self.config.get("data_buffer_max_size", 10000))
        collect_hidden_states_from_sgl = bool(self.config.get("collect_hidden_states_from_sgl", False))
        self.data_buffer = DataBuffer(max_size=buffer_max_size, store_hidden_states=collect_hidden_states_from_sgl)

        self._last_ckpt_step = -1
        self.enable_mesh_barrier = bool(self.config.get("enable_step_barrier", False))
        self._pending_checkpoint_future = None

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        if torch.distributed.get_rank() == 0:
            print(
                f"EAGLE3BackgroundTrainer use_ulysses_sp={self.use_ulysses_sp} "
                f"(sp_size={self.ulysses_sequence_parallel_size})"
            )

    def _build_empty_batch(self) -> dict[str, torch.Tensor]:
        """Build a zero-loss placeholder batch so all global FSDP ranks can enter the same step."""
        dev = next(self.model.parameters()).device
        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        target_hidden_size = int(getattr(self.model.config, "target_hidden_size", self.model.config.hidden_size))
        base_hidden_size = int(getattr(getattr(self.model, "fc", None), "in_features", target_hidden_size))

        input_ids = torch.full((1, 1), pad_id, dtype=torch.long, device=dev)
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=dev)
        hidden_states = torch.zeros((1, 1, base_hidden_size), dtype=torch.bfloat16, device=dev)
        target_hidden_states = torch.zeros((1, 1, target_hidden_size), dtype=torch.bfloat16, device=dev)
        loss_mask = torch.zeros((1, 1), dtype=torch.float32, device=dev)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
            "target_hidden_states": target_hidden_states,
            "loss_mask": loss_mask,
            "is_placeholder": torch.tensor(True, device=dev),
        }

    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict excluding frozen parameters while keeping buffers like d2t/t2d."""
        full_state_dict = self.model.state_dict()
        trainable_state_dict = {}
        frozen_param_names = {name for name, param in self.model.named_parameters() if not param.requires_grad}

        for name, param in full_state_dict.items():
            if name in frozen_param_names:
                continue
            trainable_state_dict[name] = param

        return trainable_state_dict

    def _get_trainable_full_state_dict(self) -> dict[str, torch.Tensor]:
        """Get a rank-0 full trainable state dict that can be reloaded via the existing *.pt loader."""
        full_state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        trainable_state_dict = {}
        frozen_param_names = {name for name, param in self.model.named_parameters() if not param.requires_grad}
        for name, value in full_state_dict.items():
            if name in frozen_param_names:
                continue
            if isinstance(value, torch.Tensor):
                trainable_state_dict[name] = value.detach().cpu()
            else:
                trainable_state_dict[name] = value

        return trainable_state_dict

    def _save_checkpoint_async(self, step: int, is_final: bool = False):
        if not self.checkpoint_dir:
            logger.warning(
                f"[EAGLE3Trainer rank {self.rank}] Skip checkpoint save at step={step}: checkpoint_path is not set"
            )
            return None

        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"eagle3_step_{step}")
            os.makedirs(checkpoint_path, exist_ok=True)
            logger.warning(
                f"[EAGLE3Trainer rank {self.rank}] Saving {'final ' if is_final else ''}checkpoint to {checkpoint_path}"
            )

            # Materialize a full trainable state dict before launching DCP async_save.
            # get_fsdp_full_state_dict() participates in FSDP collectives, so it must not run
            # concurrently with dcp.async_save() on the same training mesh.
            full_model_state_dict = self._get_trainable_full_state_dict()
            if self.rank == 0 and len(full_model_state_dict) > 0:
                model_export_path = os.path.join(checkpoint_path, "model.pt")
                torch.save(full_model_state_dict, model_export_path)
                logger.warning(f"[EAGLE3Trainer rank 0] Exported reloadable drafter weights to {model_export_path}")
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
                torch.save({"step": step, "is_final": is_final}, trainer_state_path)

            if self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
                dist.barrier(self.training_device_mesh.get_group())

            if not bool(self.config.get("save_distributed_checkpoint", False)):
                logger.warning(
                    f"[EAGLE3Trainer rank {self.rank}] Skipping distributed DCP save at step={step}; "
                    "saved rank-0 reloadable checkpoint only."
                )
                return _completed_future()

            model_state_dict = self._get_trainable_state_dict()
            optimizer_state_dict = self.optimizer.state_dict() if self.optimizer else {}
            state_dict = {"model": model_state_dict, "optimizer": optimizer_state_dict, "step": step}

            future = dcp.async_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_path,
                process_group=self.training_device_mesh.get_group(),
            )

            return future
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Async checkpoint save failed on rank {self.rank}: {e}")
            return None

    async def activate_training_model(
        self, device_mesh: DeviceMesh, training_ranks: list[int], base_model=None
    ) -> bool:
        start_ts = time.time()
        try:
            logger.warning(
                f"[EAGLE3Trainer rank {getattr(self, 'rank', -1)}] activate_training_model enter "
                f"training_ranks={training_ranks}"
            )

            first_param = next(self.model.parameters(), None)
            param_device = first_param.device.type if first_param is not None else None

            if self.is_offload_param or param_device != "cuda":
                load_fsdp_model_to_gpu(self.model)

            if self.optimizer is not None:
                load_fsdp_optimizer(optimizer=self.optimizer, device_id=get_device_id())

            if self.base_model_lm_head is not None:
                target_device = torch.device("cuda", get_device_id())
                self.base_model_lm_head = self.base_model_lm_head.to(device=target_device, non_blocking=True)
                self.base_model_lm_head.eval()

            self.training_device_mesh = device_mesh
            self._training_initialized = True
            self._training_active = True

            logger.warning(
                f"[EAGLE3Trainer rank {getattr(self, 'rank', -1)}] activate_training_model success "
                f"elapsed={time.time() - start_ts:.2f}s "
                f"base_model_lm_head={'present' if self.base_model_lm_head is not None else 'missing'}"
            )
            return True
        except Exception as e:
            logger.error(f"[EAGLE3Trainer rank {getattr(self, 'rank', -1)}] activate_training_model failed: {e}")
            return False

    def collect_online_data(
        self,
        batch: dict[str, torch.Tensor],
        hidden_states: list[torch.Tensor],
        last_hidden_states: Optional[list[torch.Tensor]] = None,
    ):
        """Collect online hidden states from rollout for Eagle3 training."""
        input_ids = batch.get("input_ids")
        responses = batch.get("responses")
        prompts = batch.get("prompts")
        pad_token_id = getattr(self.model_config, "pad_token_id", 0) if self.model_config else 0

        if input_ids is None or input_ids.dim() != 2:
            logger.warning(
                f"[Rank {self.rank}] Non-batched data or wrong dimensions. input_ids dim: "
                f"{input_ids.dim() if input_ids is not None else None}"
            )
            return

        if not isinstance(last_hidden_states, list) or len(last_hidden_states) < input_ids.size(0):
            last_hidden_states = hidden_states

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            if isinstance(hidden_states, list) and len(hidden_states) > i:
                h_state = hidden_states[i]
                if h_state.dim() == 1:
                    h_state = h_state.unsqueeze(0)
                elif h_state.dim() > 2:
                    h_state = h_state.view(-1, h_state.size(-1))
            else:
                continue

            last_h_state = None
            if isinstance(last_hidden_states, list) and len(last_hidden_states) > i:
                last_h_state = last_hidden_states[i]
                if last_h_state.dim() == 1:
                    last_h_state = last_h_state.unsqueeze(0)
                elif last_h_state.dim() > 2:
                    last_h_state = last_h_state.view(-1, last_h_state.size(-1))

            seq = input_ids[i]

            loss_mask = torch.zeros_like(seq, dtype=torch.float32)
            if prompts is not None and responses is not None:
                prompt_len = prompts[i].size(0)
                response_len = responses[i].size(0)
                for j in range(response_len):
                    if responses[i][j] != pad_token_id:
                        loss_mask[prompt_len + j] = 1.0
            elif responses is not None:
                response_start = seq.size(0) - responses[i].size(0)
                response_mask = (responses[i] != pad_token_id).float()
                loss_mask[response_start:] = response_mask

            seq_len = seq.size(0)
            if h_state.size(0) < seq_len:
                pad_len = seq_len - h_state.size(0)
                padding = torch.zeros(pad_len, h_state.size(1), dtype=h_state.dtype)
                h_state = torch.cat([h_state, padding], dim=0)
            elif h_state.size(0) > seq_len:
                h_state = h_state[:seq_len]

            if last_h_state is not None:
                if last_h_state.size(0) < seq_len:
                    pad_len = seq_len - last_h_state.size(0)
                    padding = torch.zeros(pad_len, last_h_state.size(1), dtype=last_h_state.dtype)
                    last_h_state = torch.cat([last_h_state, padding], dim=0)
                elif last_h_state.size(0) > seq_len:
                    last_h_state = last_h_state[:seq_len]

            item = {
                "input_ids": seq.detach().cpu(),
                "loss_mask": loss_mask.detach().cpu(),
                "hidden_states": h_state.detach().cpu(),
            }
            if last_h_state is not None:
                item["last_hidden_states"] = last_h_state.detach().cpu()
            self.data_buffer.add_batch(item)
            self.collected_data.append(item)

    def _prepare_training_batch(self, use_buffer_data: bool = True, buffer_steps: int = 2) -> dict[str, torch.Tensor]:
        effective_batch_size = min(self.batch_size, 4)

        if use_buffer_data and len(self.data_buffer) > 0:
            available_data = self.data_buffer.get_data_from_last_n_steps(buffer_steps)
            if len(available_data) < effective_batch_size:
                if 0 < len(available_data) >= min(2, effective_batch_size // 2):
                    items = available_data
                else:
                    return self._build_empty_batch()
            else:
                import random

                items = random.sample(available_data, min(len(available_data), effective_batch_size))
        else:
            if len(self.collected_data) < effective_batch_size:
                if 0 < len(self.collected_data) >= min(2, effective_batch_size // 2):
                    items = list(self.collected_data)
                else:
                    return self._build_empty_batch()
            else:
                items = list(self.collected_data)[:effective_batch_size]

        items = [item for item in items if "hidden_states" in item]
        if len(items) == 0:
            logger.warning(f"[Rank {self.rank}] No items with hidden_states found, cannot prepare batch")
            return self._build_empty_batch()
        if len(items) < min(2, effective_batch_size // 2):
            logger.warning(
                f"[Rank {self.rank}] Only {len(items)} items with hidden_states found "
                f"(need at least {min(2, effective_batch_size // 2)}), cannot prepare batch"
            )
            return self._build_empty_batch()

        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        dev = next(self.model.parameters()).device

        input_ids_list = []
        loss_mask_list = []
        hidden_states_list = []
        last_hidden_states_list = []

        for item in items:
            full_len = item["input_ids"].numel()

            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(item["input_ids"], dtype=torch.float32)
                if "prompts" in item and "responses" in item:
                    prompt_len = item["prompts"].size(0)
                    response_len = item["responses"].size(0)
                    for j in range(response_len):
                        if item["responses"][j] != pad_id:
                            item_loss_mask[prompt_len + j] = 1.0
                elif "responses" in item:
                    response_start = full_len - item["responses"].size(0)
                    response_mask = (item["responses"] != pad_id).float()
                    item_loss_mask[response_start:] = response_mask
                else:
                    item_loss_mask[:] = 1.0
            else:
                item_loss_mask = item["loss_mask"]

            max_len = min(full_len, 512)
            nonzero = torch.nonzero(item_loss_mask).flatten().cpu()
            if nonzero.numel() > 0:
                resp_start_idx = int(nonzero[0].item())
                resp_end_idx = int(nonzero[-1].item()) + 1
                window_span = max_len
                start = max(0, min(resp_start_idx, full_len - window_span))
                if resp_end_idx - start > window_span:
                    start = resp_end_idx - window_span
                end = min(full_len, start + window_span)
            else:
                start = max(0, full_len - max_len)
                end = full_len

            seq_input_ids = item["input_ids"][start:end].to(dev, non_blocking=True)
            seq_loss_mask = item_loss_mask[start:end].to(dev, non_blocking=True)
            h_states = item["hidden_states"].to(dev, dtype=torch.bfloat16, non_blocking=True)
            if "last_hidden_states" not in item:
                raise ValueError("EAGLE3 online training batch item is missing last_hidden_states")
            last_h_states = item["last_hidden_states"].to(dev, dtype=torch.bfloat16, non_blocking=True)

            h_seq_len = h_states.size(0)
            window_len = end - start
            if h_seq_len < window_len:
                if h_seq_len > 0:
                    pad_len = window_len - h_seq_len
                    padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                    seq_hidden_states = torch.cat([h_states, padding], dim=0)
                else:
                    continue
            else:
                if start < h_seq_len:
                    actual_end = min(h_seq_len, start + window_len)
                    seq_hidden_states = h_states[start:actual_end]
                    if seq_hidden_states.size(0) < window_len:
                        pad_len = window_len - seq_hidden_states.size(0)
                        padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                        seq_hidden_states = torch.cat([seq_hidden_states, padding], dim=0)
                else:
                    seq_hidden_states = h_states[-window_len:]

            input_ids_list.append(seq_input_ids)
            loss_mask_list.append(seq_loss_mask)
            hidden_states_list.append(seq_hidden_states)

            last_h_seq_len = last_h_states.size(0)
            if last_h_seq_len < window_len:
                if last_h_seq_len > 0:
                    pad_len = window_len - last_h_seq_len
                    padding = torch.zeros(pad_len, last_h_states.size(-1), dtype=last_h_states.dtype, device=dev)
                    seq_last_hidden_states = torch.cat([last_h_states, padding], dim=0)
                else:
                    raise ValueError("EAGLE3 online training batch item has empty last_hidden_states")
            else:
                if start < last_h_seq_len:
                    actual_end = min(last_h_seq_len, start + window_len)
                    seq_last_hidden_states = last_h_states[start:actual_end]
                    if seq_last_hidden_states.size(0) < window_len:
                        pad_len = window_len - seq_last_hidden_states.size(0)
                        padding = torch.zeros(
                            pad_len, last_h_states.size(-1), dtype=last_h_states.dtype, device=dev
                        )
                        seq_last_hidden_states = torch.cat([seq_last_hidden_states, padding], dim=0)
                else:
                    seq_last_hidden_states = last_h_states[-window_len:]

            last_hidden_states_list.append(seq_last_hidden_states)

        if len(input_ids_list) == 0:
            return self._build_empty_batch()

        input_ids_concat = torch.cat(input_ids_list, dim=0).unsqueeze(0)
        loss_mask_concat = torch.cat(loss_mask_list, dim=0).unsqueeze(0)
        hidden_states_concat = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
        base_hidden_states_concat = hidden_states_concat
        if len(last_hidden_states_list) != len(input_ids_list):
            raise ValueError("EAGLE3 online training requires last_hidden_states for every batch item")
        last_hidden_states_concat = torch.cat(last_hidden_states_list, dim=0).unsqueeze(0)

        expected_hidden_dim = int(getattr(self.model.config, "target_hidden_size", self.model.config.hidden_size))
        expected_fc_in_features = getattr(getattr(self.model, "fc", None), "in_features", None)
        if expected_fc_in_features is not None and base_hidden_states_concat.size(-1) != expected_fc_in_features:
            if base_hidden_states_concat.size(-1) == expected_hidden_dim and expected_fc_in_features == expected_hidden_dim * 3:
                logger.warning(
                    f"[Rank {self.rank}] Collected single-layer hidden states with dim={base_hidden_states_concat.size(-1)} "
                    f"but drafter fc expects {expected_fc_in_features}. Repeating hidden states 3x as a temporary "
                    "online-training fallback."
                )
                base_hidden_states_concat = torch.cat(
                    [base_hidden_states_concat, base_hidden_states_concat, base_hidden_states_concat], dim=-1
                )
            else:
                logger.warning(
                    f"[Rank {self.rank}] Hidden-state dim mismatch for EAGLE3 batch prep: "
                    f"got={base_hidden_states_concat.size(-1)}, expected_fc_in_features={expected_fc_in_features}, "
                    f"target_hidden_size={expected_hidden_dim}"
                )

        target_hidden_states_concat = last_hidden_states_concat

        if target_hidden_states_concat.size(-1) != expected_hidden_dim:
            if target_hidden_states_concat.size(-1) > expected_hidden_dim:
                logger.warning(
                    f"[Rank {self.rank}] target_hidden_states dim={target_hidden_states_concat.size(-1)} "
                    f"does not match base_model_lm_head in_features={expected_hidden_dim}. "
                    "Using the last hidden-state chunk for teacher logits."
                )
                target_hidden_states_concat = target_hidden_states_concat[..., -expected_hidden_dim:]
            else:
                logger.warning(
                    f"[Rank {self.rank}] target_hidden_states dim={target_hidden_states_concat.size(-1)} "
                    f"is smaller than expected_hidden_dim={expected_hidden_dim}"
                )
        total_seq_len = input_ids_concat.size(1)
        attn_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=dev)

        if self.use_ulysses_sp:
            input_ids_concat, _, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_concat, None, sp_size=self.ulysses_sequence_parallel_size
            )
            if pad_size > 0:
                loss_mask_concat = torch.nn.functional.pad(loss_mask_concat, (0, pad_size), value=0.0)
                hidden_states_concat = torch.nn.functional.pad(hidden_states_concat, (0, 0, 0, pad_size), value=0.0)
                base_hidden_states_concat = torch.nn.functional.pad(
                    base_hidden_states_concat, (0, 0, 0, pad_size), value=0.0
                )
                if last_hidden_states_concat is not None:
                    last_hidden_states_concat = torch.nn.functional.pad(
                        last_hidden_states_concat, (0, 0, 0, pad_size), value=0.0
                    )
                target_hidden_states_concat = torch.nn.functional.pad(
                    target_hidden_states_concat, (0, 0, 0, pad_size), value=0.0
                )
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_size), value=0)

            from verl.utils.ulysses import slice_input_tensor

            loss_mask_concat = slice_input_tensor(loss_mask_concat, dim=1, padding=False)
            hidden_states_concat = slice_input_tensor(hidden_states_concat, dim=1, padding=False)
            base_hidden_states_concat = slice_input_tensor(base_hidden_states_concat, dim=1, padding=False)
            if last_hidden_states_concat is not None:
                last_hidden_states_concat = slice_input_tensor(last_hidden_states_concat, dim=1, padding=False)
            target_hidden_states_concat = slice_input_tensor(target_hidden_states_concat, dim=1, padding=False)
            attn_mask = slice_input_tensor(attn_mask, dim=1, padding=False)
            self._current_pad_size = pad_size
        else:
            self._current_pad_size = 0

        target_hidden_states = target_hidden_states_concat[:, 1:].contiguous()
        loss_mask = loss_mask_concat[:, 1:].contiguous()
        input_ids = input_ids_concat[:, :-1].contiguous()
        attention_mask = attn_mask[:, :-1].contiguous()
        base_hidden_states = base_hidden_states_concat[:, :-1].contiguous()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden_states": base_hidden_states,
            "target_hidden_states": target_hidden_states,
            "loss_mask": loss_mask,
            "is_placeholder": torch.tensor(False, device=dev),
        }

    async def training_step(self, step: int) -> bool:
        try:
            logger.warning(f"[EAGLE3Trainer rank {self.rank}] training_step enter step={step}")
            with torch.enable_grad():
                return await self._training_step_impl(step)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Training step {step} failed with error: {e}")
            return False

    async def _training_step_impl(self, step: int) -> bool:
        if not self.model:
            logger.warning("No model available for training")
            return False

        collect_hidden_states_from_sgl = bool(self.config.get("collect_hidden_states_from_sgl", False))
        if not collect_hidden_states_from_sgl:
            logger.debug(
                f"[EAGLE3Trainer rank {self.rank}] Skipping training step {step} "
                f"because collect_hidden_states_from_sgl=False"
            )
            return False

        batch = self._prepare_training_batch()
        is_placeholder = bool(batch["is_placeholder"].item()) if "is_placeholder" in batch else False

        if self.enable_mesh_barrier and self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            try:
                torch.distributed.barrier(self.training_device_mesh.get_group())
            except Exception as e:
                logger.warning(f"Training barrier failed at step {step}: {e}")

        self.model.train()
        self.optimizer.zero_grad()

        if self.base_model_lm_head is None:
            raise ValueError("EAGLE3 online training requires a frozen base_model_lm_head")

        prediction_length = int(self.config.get("prediction_length", 1))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                target_logits = self.base_model_lm_head(
                    batch["target_hidden_states"].to(dtype=self.model.config.torch_dtype)
                )

            loss_list, accuracy_list = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                base_model_hidden_states=batch["hidden_states"],
                output_hidden_states=False,
                use_cache=True,
                prediction_length=prediction_length,
                target=target_logits.to(dtype=self.model.config.torch_dtype),
                loss_mask=batch["loss_mask"][..., None],
            )

        loss_weights = [0.8**i for i in range(len(loss_list))]
        loss = sum(loss_weights[i] * loss_list[i] for i in range(len(loss_list)))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.training_steps += 1
        if is_placeholder:
            logger.debug(
                f"[EAGLE3Trainer rank {self.rank}] Step {step} executed with placeholder batch "
                f"(buffer={len(self.data_buffer)}, collected={len(self.collected_data)})"
            )
        if self.training_steps % 10 == 0:
            avg_acc = sum(float(acc) for acc in accuracy_list) / max(len(accuracy_list), 1)
            logger.info(
                f"Step {self.training_steps}: loss={float(loss.item()):.4f}, "
                f"ploss={float(loss_list[0].item()):.4f}, avg_acc={avg_acc:.4f}"
            )

        if self.checkpoint_dir and (step // 100) > self._last_ckpt_step:
            logger.warning(
                f"[EAGLE3Trainer rank {self.rank}] checkpoint gate reached at step={step}, "
                f"training_steps={self.training_steps}, last_ckpt_bucket={self._last_ckpt_step}"
            )
            if self._pending_checkpoint_future is not None:
                try:
                    self._pending_checkpoint_future.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Previous checkpoint save failed: {e}")

            self._pending_checkpoint_future = self._save_checkpoint_async(step, is_final=False)
            self._last_ckpt_step = step // 100

        return True

    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        if not self.model:
            return None
        trainable_state = self._get_trainable_state_dict()
        return {k: v.detach().cpu() for k, v in trainable_state.items()}

    def update_rl_step(self, step: int):
        self.data_buffer.update_rl_step(step)
        logger.debug(
            f"[Rank {self.rank}] DataBuffer RL step updated to {self.data_buffer.get_current_step()}, "
            f"total samples: {len(self.data_buffer)}"
        )

    def has_training_data(self) -> bool:
        local_has_data = len(self.data_buffer) > 0 or len(self.collected_data) > 0
        if not dist.is_initialized():
            return local_has_data

        flag = torch.tensor(int(local_has_data), device=next(self.model.parameters()).device, dtype=torch.int32)
        if self.training_device_mesh is not None:
            group = self.training_device_mesh.get_group()
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        else:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    async def cleanup_training(self):
        self._training_active = False

        if self._pending_checkpoint_future is not None:
            logger.debug(f"[Rank {self.rank}] Waiting for pending checkpoint save to complete...")
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._pending_checkpoint_future.result)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Pending checkpoint save failed: {e}")
            self._pending_checkpoint_future = None

        if self.checkpoint_dir and self.model is not None:
            logger.warning(
                f"[EAGLE3Trainer rank {self.rank}] Triggering final checkpoint save at step={self.training_steps}"
            )
            final_future = self._save_checkpoint_async(self.training_steps, is_final=True)
            if final_future is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, final_future.result)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Final checkpoint save failed: {e}")

        if self.training_device_mesh is not None:
            try:
                await asyncio.sleep(0.1)
                if self.training_device_mesh.size() > 1:
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, lambda: torch.distributed.barrier(self.training_device_mesh.get_group())
                            ),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Rank {self.rank} barrier timeout during cleanup, continuing anyway")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Process group cleanup error (expected): {e}")

        if self.model is not None:
            try:
                offload_fsdp_model_to_cpu(self.model)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter model during cleanup: {e}")

        if self.base_model_lm_head is not None:
            try:
                self.base_model_lm_head = self.base_model_lm_head.to("cpu")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to move base_model_lm_head to CPU during cleanup: {e}")

        if self.optimizer is not None:
            try:
                offload_fsdp_optimizer(self.optimizer)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter optimizer during cleanup: {e}")

        self.collected_data.clear()
        self.data_buffer.clear()
        self.training_device_mesh = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._training_initialized = False
        self.training_steps = 0

    @property
    def is_training_initialized(self) -> bool:
        return self._training_initialized

    @property
    def is_training_active(self) -> bool:
        return self._training_active

    def set_training_active(self, active: bool):
        self._training_active = active
