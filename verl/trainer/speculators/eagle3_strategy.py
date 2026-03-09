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

from typing import Any

import torch

from verl.trainer.speculators.base_strategy import TemplateSpecDecodeStrategy
from verl.trainer.speculators.strategy_interface import (
    DraftForwardOutput,
    DraftForwardRequest,
    LossOutput,
    TargetSignalRequest,
    TargetRuntimeView,
)


class Eagle3Strategy(TemplateSpecDecodeStrategy):
    """EAGLE3 strategy with online TTT-style draft training."""

    REQUIRED_ROLLOUT_ENGINE = "vllm"

    def __init__(self):
        super().__init__()
        self.ttt_steps = 1
        self.pad_token_id = 0
        self.ignore_index = -100
        self.aux_hidden_layers: list[int] = [1, -2, -1]

    def _resolve_aux_hidden_layers(self, target_config, strategy_cfg: dict[str, Any]) -> list[int]:
        aux_hidden_layers = strategy_cfg.get("aux_hidden_layers", None)
        if aux_hidden_layers is None and self._draft_config_obj is not None:
            draft_cfg = self._draft_config_obj
            aux_hidden_layers = getattr(draft_cfg, "aux_hidden_layers", None)
            if aux_hidden_layers is None:
                eagle3_cfg = getattr(draft_cfg, "eagle3_config", None)
                if isinstance(eagle3_cfg, dict):
                    aux_hidden_layers = eagle3_cfg.get("aux_hidden_layers", None)

        if aux_hidden_layers is None:
            num_hidden_layers = getattr(target_config, "num_hidden_layers", None)
            if not isinstance(num_hidden_layers, int):
                raise ValueError(
                    "Failed to resolve aux_hidden_layers because target model config does not provide "
                    f"`num_hidden_layers`: {target_config!r}"
                )
            aux_hidden_layers = [1, num_hidden_layers // 2 - 1, num_hidden_layers - 4]

        resolved = [int(x) for x in aux_hidden_layers]
        if len(resolved) != 3:
            raise ValueError(
                "aux_hidden_layers is expected to contain exactly 3 layers for EAGLE3, "
                f"but got {resolved}."
            )
        return resolved

    def _apply_draft_freeze_from_config(self) -> None:
        if self.draft_model is None:
            return
        draft_cfg = self._draft_config_obj
        freeze_embed_tokens = bool(getattr(draft_cfg, "freeze_reference_embed_tokens", False))
        freeze_lm_head = bool(getattr(draft_cfg, "freeze_reference_lm_head", False))
        if not (freeze_embed_tokens or freeze_lm_head):
            return

        draft_model = self._unwrap_model(self.draft_model)
        if freeze_embed_tokens:
            embed_tokens = getattr(draft_model, "embed_tokens", None)
            if embed_tokens is None and hasattr(draft_model, "get_input_embeddings"):
                embed_tokens = draft_model.get_input_embeddings()
            if embed_tokens is None or not hasattr(embed_tokens, "weight"):
                raise ValueError(
                    "Eagle3Strategy configured freeze_reference_embed_tokens=True, "
                    "but draft model has no accessible embedding weight."
                )
            embed_tokens.weight.requires_grad = False

        if freeze_lm_head:
            lm_head = getattr(draft_model, "lm_head", None)
            if lm_head is None and hasattr(draft_model, "get_output_embeddings"):
                lm_head = draft_model.get_output_embeddings()
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise ValueError(
                    "Eagle3Strategy configured freeze_reference_lm_head=True, "
                    "but draft model has no accessible lm_head weight."
                )
            lm_head.weight.requires_grad = False

    def initialize(self, target_model, spec_decode_cfg: dict[str, Any], runtime_ctx) -> None:
        super().initialize(target_model=target_model, spec_decode_cfg=spec_decode_cfg, runtime_ctx=runtime_ctx)
        cfg = self.strategy_cfg

        self.ttt_steps = int(cfg.get("ttt_steps", cfg.get("length", 1)))
        self.ttt_steps = max(self.ttt_steps, 1)
        self.ignore_index = int(cfg.get("ignore_index", -100))

        target_config = getattr(self._unwrap_model(self.target_model), "config", None)
        default_pad_id = getattr(target_config, "pad_token_id", 0) or 0
        self.pad_token_id = int(cfg.get("pad_token_id", default_pad_id))
        self.aux_hidden_layers = self._resolve_aux_hidden_layers(target_config=target_config, strategy_cfg=cfg)
        self._apply_draft_freeze_from_config()

    def bind_draft_module(self, draft_module) -> None:
        super().bind_draft_module(draft_module=draft_module)
        self._apply_draft_freeze_from_config()

    def build_draft_module(self, target_model, strategy_cfg: dict[str, Any]):
        return super().build_draft_module(target_model=target_model, strategy_cfg=strategy_cfg)

    def get_target_signal_request(self) -> TargetSignalRequest:
        return TargetSignalRequest(
            hidden_layers=list(self.aux_hidden_layers),
            include_input_embeddings=False,
        )

    def _resolve_hidden_tensor(self, target_view: TargetRuntimeView, layer_id: int):
        tensor = target_view.hidden_by_layer.get(layer_id, None)
        if tensor is not None:
            return tensor
        if layer_id < 0 and target_view.hidden_by_layer:
            sorted_ids = sorted(target_view.hidden_by_layer.keys())
            idx = len(sorted_ids) + layer_id
            if 0 <= idx < len(sorted_ids):
                return target_view.hidden_by_layer[sorted_ids[idx]]
        return None

    @staticmethod
    def _shift_left_2d(tensor: torch.Tensor, fill_value: int | float) -> torch.Tensor:
        shifted = torch.roll(tensor, shifts=-1, dims=1)
        shifted[:, -1] = fill_value
        return shifted

    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        input_ids = target_view.input_ids
        if input_ids is None:
            raise ValueError("Eagle3Strategy requires input_ids in target_view.")

        hidden_tensors = []
        missing_layers: list[int] = []
        for layer_id in self.aux_hidden_layers:
            hidden = self._resolve_hidden_tensor(target_view, layer_id)
            if hidden is None:
                missing_layers.append(layer_id)
                continue
            hidden_tensors.append(hidden)
        if missing_layers:
            raise ValueError(
                "Eagle3Strategy requires teacher hidden states for all configured aux layers, "
                f"but missing {missing_layers} from {sorted(target_view.hidden_by_layer.keys())}."
            )

        base_hidden_size = hidden_tensors[0].shape[-1]
        if any(t.shape[-1] != base_hidden_size for t in hidden_tensors):
            raise ValueError(
                "Eagle3Strategy requires all selected teacher hidden states to share the same hidden size."
            )
        concat_hidden = torch.cat(hidden_tensors, dim=-1)
        expected_concat_size = base_hidden_size * 3
        if concat_hidden.shape[-1] != expected_concat_size:
            raise ValueError(
                f"Eagle3Strategy expects concatenated teacher hidden size {expected_concat_size}, "
                f"but got {concat_hidden.shape[-1]}."
            )
        labels = self._resolve_labels(target_view)
        if labels is None:
            labels = input_ids
        loss_mask = self._resolve_loss_mask(target_view=target_view)
        attention_mask = target_view.attention_mask
        position_ids = target_view.position_ids
        input_embeddings = target_view.input_embeddings

        # In Megatron no-padding path, captured teacher hidden may exclude the terminal token state
        # while input_ids still includes it. Align once at source for consistent TTT behavior.
        hidden_seq_len = concat_hidden.shape[1]
        input_seq_len = input_ids.shape[1]
        if input_seq_len != hidden_seq_len:
            if input_seq_len == hidden_seq_len + 1:
                input_ids = input_ids[:, :hidden_seq_len]
                if labels is not None:
                    labels = labels[:, :hidden_seq_len]
                if loss_mask is not None:
                    loss_mask = loss_mask[:, :hidden_seq_len]
                if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
                    attention_mask = attention_mask[:, :hidden_seq_len]
                if position_ids is not None and torch.is_tensor(position_ids):
                    if position_ids.ndim >= 2:
                        position_ids = position_ids[:, :hidden_seq_len]
                    elif position_ids.ndim == 1:
                        position_ids = position_ids[:hidden_seq_len]
                if input_embeddings is not None and torch.is_tensor(input_embeddings) and input_embeddings.ndim >= 2:
                    input_embeddings = input_embeddings[:, :hidden_seq_len, ...]
            else:
                raise ValueError(
                    "Eagle3Strategy found sequence mismatch between input_ids and teacher hidden states. "
                    f"input_ids.shape={tuple(input_ids.shape)}, teacher_hidden.shape={tuple(concat_hidden.shape)}. "
                    "Only a +1 terminal-token mismatch is supported; other mismatches indicate upstream "
                    "target hidden capture/aggregation issues."
                )

        phase = target_view.backend_payload.get("phase", None)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "labels": labels,
            "concat_hidden": concat_hidden,
            "input_embeddings": input_embeddings,
            "phase": phase,
        }

    def build_draft_forward_request(
        self,
        teacher_signals: dict[str, Any],
    ) -> DraftForwardRequest:
        return DraftForwardRequest(
            kwargs={
                "input_ids": teacher_signals["input_ids"],
                "attention_mask": teacher_signals["attention_mask"],
                "position_ids": teacher_signals["position_ids"],
                "input_embeddings": teacher_signals["input_embeddings"],
                "teacher_hidden": teacher_signals["concat_hidden"],
                "labels": teacher_signals["labels"],
                "loss_mask": teacher_signals["loss_mask"],
                "ttt_steps": self.ttt_steps,
                "pad_token_id": self.pad_token_id,
            }
        )

    @staticmethod
    def _normalize_logits_layout(logits: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape [B, S, V] or [S, B, V], got {tuple(logits.shape)}")
        if logits.shape[0] == batch_size and logits.shape[1] == seq_len:
            return logits
        if logits.shape[0] == seq_len and logits.shape[1] == batch_size:
            return logits.transpose(0, 1).contiguous()
        return logits

    def forward_draft(self, draft_request: DraftForwardRequest) -> DraftForwardOutput:
        request_kwargs = draft_request.kwargs
        input_ids: torch.Tensor = request_kwargs["input_ids"]
        attention_mask = request_kwargs.get("attention_mask", None)
        position_ids = request_kwargs.get("position_ids", None)
        input_embeddings: torch.Tensor | None = request_kwargs.get("input_embeddings", None)
        hidden_states: torch.Tensor = request_kwargs["teacher_hidden"]
        labels: torch.Tensor = request_kwargs["labels"]
        loss_mask = request_kwargs.get("loss_mask", None)
        ttt_steps = int(request_kwargs.get("ttt_steps", self.ttt_steps))
        pad_token_id = int(request_kwargs.get("pad_token_id", self.pad_token_id))

        logits_per_step: list[torch.Tensor] = []
        labels_per_step: list[torch.Tensor] = []
        masks_per_step: list[torch.Tensor | None] = []
        raw_outputs: list[Any] = []

        step_input_ids = input_ids
        step_labels = labels
        step_loss_mask = loss_mask

        for step_idx in range(ttt_steps):
            step_embeddings = input_embeddings
            if hidden_states.ndim != 3:
                raise ValueError(
                    f"Expected teacher hidden states shape [B, S, 3H] before draft projection, "
                    f"got {tuple(hidden_states.shape)}"
                )
            draft_kwargs = {
                "input_ids": step_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "hidden_states": hidden_states,
            }
            if step_embeddings is not None:
                step_embeddings = step_embeddings.to(hidden_states.dtype)
                draft_kwargs["inputs_embeds"] = step_embeddings
                draft_kwargs["input_embeds"] = step_embeddings

            raw_output = self.draft_model(**draft_kwargs)
            parsed = self._normalize_forward_output(raw_output)
            raw_outputs.append(raw_output)

            next_hidden = parsed.hidden_states if parsed.hidden_states is not None else hidden_states
            logits = parsed.logits
            if logits is None:
                raise ValueError("Eagle3Strategy requires draft model forward to return logits.")
            logits = self._normalize_logits_layout(
                logits, batch_size=step_input_ids.shape[0], seq_len=step_input_ids.shape[1]
            )

            logits_per_step.append(logits)
            labels_per_step.append(step_labels)
            masks_per_step.append(step_loss_mask)
            hidden_states = next_hidden

            if step_idx != ttt_steps - 1:
                step_input_ids = self._shift_left_2d(step_input_ids, fill_value=pad_token_id)
                step_labels = self._shift_left_2d(step_labels, fill_value=self.ignore_index)
                if step_loss_mask is not None:
                    fill = 0 if step_loss_mask.dtype != torch.bool else False
                    step_loss_mask = self._shift_left_2d(step_loss_mask, fill_value=fill)

        return DraftForwardOutput(
            logits=logits_per_step[-1],
            hidden_states=hidden_states,
            raw_output=raw_outputs[-1] if raw_outputs else None,
            extras={
                "logits_per_step": logits_per_step,
                "labels_per_step": labels_per_step,
                "loss_masks_per_step": masks_per_step,
            },
        )

    def compute_draft_loss(
        self,
        draft_output: DraftForwardOutput,
        teacher_signals: dict[str, Any],
    ) -> LossOutput:
        logits_per_step = draft_output.extras.get("logits_per_step", None) if draft_output.extras else None
        if logits_per_step is None:
            logits_per_step = [draft_output.logits]

        labels_per_step = draft_output.extras.get("labels_per_step", None) if draft_output.extras else None
        masks_per_step = draft_output.extras.get("loss_masks_per_step", None) if draft_output.extras else None

        losses: list[torch.Tensor] = []
        accs: list[torch.Tensor] = []
        aux_losses: dict[str, torch.Tensor] = {}

        fallback_labels = teacher_signals["labels"]
        fallback_mask = teacher_signals.get("loss_mask", None)
        for idx, logits_i in enumerate(logits_per_step):
            if logits_i is None:
                continue
            labels_i = fallback_labels if labels_per_step is None else labels_per_step[idx]
            mask_i = fallback_mask if masks_per_step is None else masks_per_step[idx]
            loss_i, acc_i = self._next_token_ce(
                logits=logits_i,
                labels=labels_i,
                loss_mask=mask_i,
                ignore_index=self.ignore_index,
            )
            losses.append(loss_i)
            accs.append(acc_i)
            aux_losses[f"ttt_step_{idx}_loss"] = loss_i.detach()

        if not losses:
            zero = fallback_labels.new_zeros((), dtype=torch.float32)
            metrics = {"speculator_loss": 0.0}
            phase = teacher_signals.get("phase", None)
            if phase:
                metrics[f"{phase}/speculator_loss"] = 0.0
            return LossOutput(total_loss=zero, metrics=metrics)

        total_loss = torch.stack(losses).mean()
        mean_acc = torch.stack(accs).mean() if accs else total_loss.detach().new_zeros(())
        metrics = {
            "speculator_loss": float(total_loss.detach().float().item()),
            "speculator_acc": float(mean_acc.detach().float().item()),
        }
        phase = teacher_signals.get("phase", None)
        if phase:
            metrics[f"{phase}/speculator_loss"] = metrics["speculator_loss"]
            metrics[f"{phase}/speculator_acc"] = metrics["speculator_acc"]
        return LossOutput(total_loss=total_loss, metrics=metrics, aux_losses=aux_losses)
