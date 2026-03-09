# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from verl.trainer.speculators.base_strategy import TemplateSpecDecodeStrategy
from verl.trainer.speculators.strategy_interface import (
    DraftForwardOutput,
    DraftForwardRequest,
    LossOutput,
    TargetSignalRequest,
    TargetRuntimeView,
)

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
):
    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        return None

    def dflash_mask_mod(batch_idx, _head_idx, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[batch_idx, q_block_id]

        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[batch_idx, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    batch_size, n_blocks = anchor_positions.shape
    q_len = n_blocks * block_size
    kv_len = seq_len + n_blocks * block_size
    return create_block_mask(dflash_mask_mod, B=batch_size, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)


class DFlashStrategy(TemplateSpecDecodeStrategy):
    """DFlash strategy with block-wise CE objective."""

    REQUIRED_ROLLOUT_ENGINE = "sglang"

    def __init__(self):
        super().__init__()
        self.block_size = 16
        self.num_anchors = 512
        self.mask_token_id = 0
        self.loss_decay_gamma: Optional[float] = None
        self.attention_backend = "flex_attention"
        self.target_layer_ids: list[int] = [-1]
        self.reuse_target_lm_head = True

    def initialize(self, target_model, spec_decode_cfg: dict[str, Any], runtime_ctx) -> None:
        super().initialize(target_model=target_model, spec_decode_cfg=spec_decode_cfg, runtime_ctx=runtime_ctx)
        cfg = self.strategy_cfg
        self.block_size = int(cfg.get("block_size", 16))
        self.num_anchors = int(cfg.get("num_anchors", 512))
        self.mask_token_id = int(cfg.get("mask_token_id", 0))
        self.loss_decay_gamma = cfg.get("loss_decay_gamma", None)
        if self.loss_decay_gamma is not None:
            self.loss_decay_gamma = float(self.loss_decay_gamma)
        self.attention_backend = str(cfg.get("attention_backend", "flex_attention"))

        self.target_layer_ids = [int(x) for x in cfg.get("target_layer_ids", [-1])]
        self.reuse_target_lm_head = bool(cfg.get("reuse_target_lm_head", True))

        if self.reuse_target_lm_head and self._target_lm_head is None:
            raise ValueError("DFlashStrategy configured with reuse_target_lm_head=True, but target model has no lm_head.")

    def build_draft_module(self, target_model, strategy_cfg: dict[str, Any]):
        return super().build_draft_module(target_model=target_model, strategy_cfg=strategy_cfg)

    def get_target_signal_request(self) -> TargetSignalRequest:
        return TargetSignalRequest(
            hidden_layers=list(self.target_layer_ids),
            include_input_embeddings=True,
            reuse_target_lm_head_module=self.reuse_target_lm_head,
        )

    def _resolve_hidden_tensor(self, target_view: TargetRuntimeView, layer_id: int):
        tensor = target_view.hidden_by_layer.get(layer_id, None)
        if tensor is not None:
            return tensor
        if layer_id == -1 and target_view.last_hidden is not None:
            return target_view.last_hidden
        if layer_id < 0 and target_view.hidden_by_layer:
            sorted_ids = sorted(target_view.hidden_by_layer.keys())
            idx = len(sorted_ids) + layer_id
            if 0 <= idx < len(sorted_ids):
                return target_view.hidden_by_layer[sorted_ids[idx]]
        return None

    @staticmethod
    def _extract_logits_tensor(raw_logits: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(raw_logits):
            return raw_logits
        if isinstance(raw_logits, (tuple, list)):
            for item in raw_logits:
                if torch.is_tensor(item):
                    return item
        return None

    @staticmethod
    def _normalize_logits_layout(logits: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
        if logits.ndim != 3:
            return logits
        if logits.shape[0] == batch_size and logits.shape[1] == seq_len:
            return logits
        if logits.shape[0] == seq_len and logits.shape[1] == batch_size:
            return logits.transpose(0, 1).contiguous()
        return logits

    def _get_tp_world_size(self) -> int:
        backend = str(getattr(self.runtime_ctx, "backend", "")).lower()
        if backend != "megatron":
            return 1
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 1
        try:
            from megatron.core import parallel_state as mpu

            return torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())
        except Exception:
            return 1

    def _is_megatron_vocab_parallel_logits(
        self,
        logits: torch.Tensor,
        *,
        expected_vocab_size: Optional[int],
    ) -> bool:
        if not torch.is_tensor(logits) or logits.ndim != 3:
            return False
        if self._get_tp_world_size() <= 1:
            return False
        if bool(getattr(logits, "tensor_model_parallel", False)):
            return True
        if expected_vocab_size is None:
            return False
        if logits.shape[-1] == expected_vocab_size:
            return False
        return logits.shape[-1] * self._get_tp_world_size() == expected_vocab_size

    def _vocab_parallel_argmax(self, flat_logits: torch.Tensor) -> torch.Tensor:
        if self._get_tp_world_size() <= 1:
            return flat_logits.argmax(dim=-1)

        from megatron.core import parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_world_size = torch.distributed.get_world_size(group=tp_group)
        tp_rank = torch.distributed.get_rank(group=tp_group)

        local_values, local_indices = flat_logits.max(dim=-1)
        local_indices = local_indices + tp_rank * flat_logits.shape[-1]

        gathered_values = [torch.empty_like(local_values) for _ in range(tp_world_size)]
        gathered_indices = [torch.empty_like(local_indices) for _ in range(tp_world_size)]
        torch.distributed.all_gather(gathered_values, local_values, group=tp_group)
        torch.distributed.all_gather(gathered_indices, local_indices, group=tp_group)

        stacked_values = torch.stack(gathered_values, dim=0)
        stacked_indices = torch.stack(gathered_indices, dim=0)
        winner = stacked_values.argmax(dim=0)
        return stacked_indices.gather(dim=0, index=winner.unsqueeze(0)).squeeze(0)

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - self.block_size, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("DFlashStrategy found no valid anchor positions; please verify loss_mask preprocessing.")

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.full_like(indices, seq_len + 1))
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        bsz, n_blocks = anchor_positions.shape
        offsets = torch.arange(self.block_size, device=anchor_positions.device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.reshape(bsz, n_blocks * self.block_size)

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        input_embeddings: Optional[torch.Tensor],
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device
        noise_ids = torch.full(
            (bsz, n_blocks * self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = torch.arange(n_blocks, device=device) * self.block_size
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)
        flat_batch = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n_blocks)
        noise_ids[flat_batch, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.full_like(anchor_tokens, self.mask_token_id),
        )

        if self._target_embed_tokens is not None:
            return self._target_embed_tokens(noise_ids)

        if input_embeddings is None:
            raise ValueError(
                "DFlashStrategy cannot build noise embeddings: neither reusable target embed_tokens "
                "nor target_view.input_embeddings is available."
            )

        hidden_size = input_embeddings.shape[-1]
        noise_embedding = torch.zeros((bsz, n_blocks * self.block_size, hidden_size), device=device, dtype=input_embeddings.dtype)
        gathered_anchor_emb = torch.gather(
            input_embeddings,
            1,
            valid_anchor_positions.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        noise_embedding[flat_batch, block_starts] = torch.where(
            block_keep_mask.unsqueeze(-1),
            gathered_anchor_emb,
            torch.zeros_like(gathered_anchor_emb),
        )
        return noise_embedding

    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        input_ids = target_view.input_ids
        if input_ids is None:
            raise ValueError("DFlashStrategy requires input_ids in target_view.")

        hidden_tensors = []
        for layer_id in self.target_layer_ids:
            hidden = self._resolve_hidden_tensor(target_view, layer_id)
            if hidden is not None:
                hidden_tensors.append(hidden)
        if not hidden_tensors:
            if target_view.last_hidden is None:
                raise ValueError("DFlashStrategy requires target hidden states.")
            hidden_tensors = [target_view.last_hidden]

        target_hidden = hidden_tensors[0] if len(hidden_tensors) == 1 else torch.cat(hidden_tensors, dim=-1)

        loss_mask = self._resolve_loss_mask(target_view=target_view)
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32, device=input_ids.device)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden": target_hidden,
            "input_embeddings": target_view.input_embeddings,
            "phase": target_view.backend_payload.get("phase", None),
        }

    def build_draft_forward_request(
        self,
        teacher_signals: dict[str, Any],
    ) -> DraftForwardRequest:
        input_ids: torch.Tensor = teacher_signals["input_ids"]
        loss_mask: torch.Tensor = teacher_signals["loss_mask"]
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(seq_len=seq_len, loss_mask=loss_mask, device=device)
        noise_embedding = self._create_noise_embed(
            input_ids=input_ids,
            input_embeddings=teacher_signals.get("input_embeddings", None),
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        attention_mask = None
        if self.attention_backend == "flex_attention":
            attention_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                block_size=self.block_size,
                device=device,
            )

        teacher_signals["anchor_positions"] = anchor_positions
        teacher_signals["block_keep_mask"] = block_keep_mask

        return DraftForwardRequest(
            kwargs={
                "position_ids": full_position_ids,
                "noise_embedding": noise_embedding,
                "target_hidden": teacher_signals["target_hidden"],
                "attention_mask": attention_mask,
            }
        )

    def forward_draft(self, draft_request: DraftForwardRequest) -> DraftForwardOutput:
        kwargs = dict(draft_request.kwargs)
        raw_output = self.draft_model(
            position_ids=kwargs["position_ids"],
            attention_mask=kwargs.get("attention_mask", None),
            noise_embedding=kwargs["noise_embedding"],
            target_hidden=kwargs["target_hidden"],
        )
        parsed = self._normalize_forward_output(raw_output)
        hidden = parsed.hidden_states
        if hidden is None and torch.is_tensor(raw_output):
            hidden = raw_output
        logits = self._extract_logits_tensor(parsed.logits)
        if logits is None:
            if hidden is None:
                raise ValueError("DFlashStrategy requires draft output hidden states or logits.")
            if self.reuse_target_lm_head and self._target_lm_head is not None:
                logits = self._extract_logits_tensor(self._target_lm_head(hidden))
            else:
                raise ValueError("DFlashStrategy requires draft model to return logits when reuse_target_lm_head=False.")
        if logits is None:
            raise ValueError("DFlashStrategy expected tensor logits from draft output or target lm_head fallback.")
        if torch.is_tensor(hidden) and hidden.ndim == 3:
            logits = self._normalize_logits_layout(logits, batch_size=hidden.shape[0], seq_len=hidden.shape[1])

        return DraftForwardOutput(logits=logits, hidden_states=hidden, raw_output=raw_output)

    def compute_draft_loss(
        self,
        draft_output: DraftForwardOutput,
        teacher_signals: dict[str, Any],
    ) -> LossOutput:
        logits = draft_output.logits
        if logits is None:
            raise ValueError("DFlashStrategy requires logits for loss computation.")
        if logits.ndim != 3:
            raise ValueError(f"DFlashStrategy expects logits shape [B, N*block_size, V], got {tuple(logits.shape)}")

        input_ids: torch.Tensor = teacher_signals["input_ids"]
        loss_mask: torch.Tensor = teacher_signals["loss_mask"]
        anchor_positions: torch.Tensor = teacher_signals["anchor_positions"]
        block_keep_mask: torch.Tensor = teacher_signals["block_keep_mask"]

        bsz, seq_len = input_ids.shape
        shard_vocab_size = logits.shape[-1]
        n_blocks = min(anchor_positions.shape[1], logits.shape[1] // self.block_size)
        if n_blocks <= 0:
            zero = logits.new_zeros(())
            metrics = {"speculator_loss": 0.0}
            phase = teacher_signals.get("phase", None)
            if phase:
                metrics[f"{phase}/speculator_loss"] = 0.0
            return LossOutput(total_loss=zero, metrics=metrics)

        logits = logits[:, : n_blocks * self.block_size, :].contiguous()
        logits = logits.view(bsz, n_blocks, self.block_size, shard_vocab_size)
        anchor_positions = anchor_positions[:, :n_blocks]
        block_keep_mask = block_keep_mask[:, :n_blocks]

        label_offsets = torch.arange(0, self.block_size, device=logits.device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_indices = label_indices.clamp(max=seq_len - 1)

        expanded_ids = input_ids.unsqueeze(1).expand(-1, n_blocks, -1)
        target_ids = torch.gather(expanded_ids, 2, safe_indices)

        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=logits.device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        expanded_loss_mask = loss_mask.unsqueeze(1).expand(-1, n_blocks, -1)
        original_loss_mask = torch.gather(expanded_loss_mask, 2, safe_indices)
        weight_mask = weight_mask * original_loss_mask.float()

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=logits.device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        flat_logits = logits.view(-1, shard_vocab_size)
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)
        expected_vocab_size = self._infer_vocab_size_from_model(self._unwrap_model(self.target_model))
        use_vocab_parallel_loss = self._is_megatron_vocab_parallel_logits(
            logits=logits,
            expected_vocab_size=expected_vocab_size,
        )
        if use_vocab_parallel_loss:
            from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

            loss_per_token = -vocab_parallel_log_probs_from_logits(flat_logits, flat_targets)
        else:
            loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_count = flat_weights.sum().clamp_min(1e-6)
        total_loss = (loss_per_token * flat_weights).sum() / valid_count

        with torch.no_grad():
            if use_vocab_parallel_loss:
                pred_ids = self._vocab_parallel_argmax(flat_logits)
            else:
                pred_ids = flat_logits.argmax(dim=-1)
            correct = ((pred_ids == flat_targets).float() * (flat_weights > 0.5).float()).sum()
            acc = correct / (flat_weights > 0.5).float().sum().clamp_min(1e-6)

        metrics = {
            "speculator_loss": float(total_loss.detach().float().item()),
            "speculator_acc": float(acc.detach().float().item()),
        }
        phase = teacher_signals.get("phase", None)
        if phase:
            metrics[f"{phase}/speculator_loss"] = metrics["speculator_loss"]
            metrics[f"{phase}/speculator_acc"] = metrics["speculator_acc"]
        return LossOutput(total_loss=total_loss, metrics=metrics)
