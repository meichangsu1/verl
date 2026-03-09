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

import copy
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from verl.trainer.speculators.draft_auto import AutoDraftModel
from verl.trainer.speculators.strategy_interface import (
    BaseSpecDecodeStrategy,
    DraftModelSpec,
    DraftForwardOutput,
    DraftForwardRequest,
    LossOutput,
    StrategyRuntimeContext,
    TargetRuntimeView,
)


class TemplateSpecDecodeStrategy(BaseSpecDecodeStrategy):
    """
    Template-method strategy with default AutoModel loading and loss workflow.
    """

    REQUIRED_ROLLOUT_ENGINE = "vllm"

    def __init__(self):
        self.target_model = None
        self.draft_model = None
        self.spec_decode_cfg: dict[str, Any] = {}
        self.strategy_cfg: dict[str, Any] = {}
        self.runtime_ctx: Optional[StrategyRuntimeContext] = None
        self._draft_config_obj = None
        self.loss_cfg: dict[str, Any] = {}
        self._enable_draft_module: bool = True
        self._target_embed_tokens = None
        self._target_lm_head = None

    @staticmethod
    def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "items"):
            return dict(value)
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")

    def _unwrap_model(self, model):
        backend = str(getattr(self.runtime_ctx, "backend", "")).lower()
        if backend == "megatron":
            try:
                from verl.utils.megatron_utils import unwrap_model as megatron_unwrap_model

                return megatron_unwrap_model(model)
            except Exception:
                pass
        if backend in {"fsdp", "fsdp2"}:
            try:
                from verl.utils.fsdp_utils import unwrap_model as fsdp_unwrap_model

                return fsdp_unwrap_model(model)
            except Exception:
                pass
        return model

    def _find_target_embed_tokens(self):
        model = self._unwrap_model(self.target_model)
        if hasattr(model, "get_input_embeddings"):
            module = model.get_input_embeddings()
            if module is not None:
                return module
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        if hasattr(model, "embed_tokens"):
            return model.embed_tokens
        return None

    def _find_target_lm_head(self):
        model = self._unwrap_model(self.target_model)
        if hasattr(model, "get_output_embeddings"):
            module = model.get_output_embeddings()
            if module is not None:
                return module
        if hasattr(model, "lm_head"):
            return model.lm_head
        if hasattr(model, "output_layer"):
            return model.output_layer
        return None

    @staticmethod
    def _infer_vocab_size_from_model(model) -> Optional[int]:
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
        return None

    @staticmethod
    def _normalize_path(path_value: Any) -> Optional[str]:
        if not isinstance(path_value, str):
            return None
        value = path_value.strip()
        return value if value else None

    def _resolve_target_model_path(self, target_config) -> Optional[str]:
        runtime_path = self._normalize_path(getattr(self.runtime_ctx, "target_model_path", None))
        if runtime_path is not None:
            return runtime_path
        for attr in ("_name_or_path", "name_or_path"):
            cfg_path = self._normalize_path(getattr(target_config, attr, None))
            if cfg_path is not None:
                return cfg_path
        return None

    def _resolve_draft_model_spec(self) -> DraftModelSpec:
        raw = self.spec_decode_cfg.get("draft_model", None)
        if raw is None:
            raise ValueError(
                f"{self.__class__.__name__} requires model.spec_decode.draft_model for AutoModel loading."
            )
        cfg = self._as_dict(raw, name="model.spec_decode.draft_model")
        path = cfg.get("path", None)
        init_mode = str(cfg.get("init", "pretrained")).strip().lower() or "pretrained"
        if init_mode not in {"pretrained", "from_config"}:
            raise ValueError(
                f"model.spec_decode.draft_model.init must be one of ['pretrained', 'from_config'], got: {init_mode}"
            )
        return DraftModelSpec(
            path=path,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            init=init_mode,
        )

    def initialize(
        self,
        target_model,
        spec_decode_cfg: dict[str, Any],
        runtime_ctx: StrategyRuntimeContext,
    ) -> None:
        self.target_model = target_model
        self.spec_decode_cfg = self._as_dict(spec_decode_cfg, name="spec_decode_cfg")
        strategy_cfg = self.spec_decode_cfg.get("strategy_config", self.spec_decode_cfg.get("config", {}))
        self.strategy_cfg = self._as_dict(strategy_cfg, name="model.spec_decode.strategy_config")
        self.loss_cfg = self._as_dict(self.spec_decode_cfg.get("loss", {}), name="model.spec_decode.loss")
        self.runtime_ctx = runtime_ctx
        self._enable_draft_module = bool(getattr(runtime_ctx, "enable_draft_module", True))

        self._target_embed_tokens = self._find_target_embed_tokens()
        self._target_lm_head = self._find_target_lm_head()

        if self._enable_draft_module:
            if self.draft_model is None:
                self.draft_model = self.build_draft_module(target_model=target_model, strategy_cfg=self.strategy_cfg)
            if self.draft_model is None:
                raise ValueError(f"{self.__class__.__name__} failed to build draft model.")
            self._draft_config_obj = getattr(self.draft_model, "config", self._draft_config_obj)
        else:
            self.draft_model = None
            self._draft_config_obj = None

    def bind_draft_module(self, draft_module) -> None:
        if draft_module is None:
            raise ValueError("bind_draft_module received None draft_module")
        self.draft_model = draft_module
        if self._draft_config_obj is None:
            self._draft_config_obj = getattr(draft_module, "config", None)

    def build_draft_module(self, target_model, strategy_cfg: dict[str, Any]):
        del target_model, strategy_cfg
        spec = self._resolve_draft_model_spec()

        target_cfg = getattr(self._unwrap_model(self.target_model), "config", None)
        target_cfg_copy = copy.deepcopy(target_cfg) if target_cfg is not None else None

        if spec.init == "pretrained":
            base_config = None if spec.path else target_cfg_copy
            model = AutoDraftModel.from_pretrained(
                spec.path,
                trust_remote_code=spec.trust_remote_code,
                torch_dtype=self.runtime_ctx.torch_dtype,
                config=base_config,
            )
        elif spec.init == "from_config":
            AutoDraftModel.maybe_load_local_plugins(spec.path)
            draft_config = None
            if spec.path:
                try:
                    draft_config = AutoConfig.from_pretrained(spec.path, trust_remote_code=spec.trust_remote_code)
                except Exception:
                    draft_config = None
            if draft_config is None:
                draft_config = target_cfg_copy
            if draft_config is None:
                raise ValueError(
                    "draft_model.init=from_config requires either a valid draft_model.path config "
                    "or a target model config for fallback initialization."
                )
            explicit_reference_path = self._normalize_path(getattr(draft_config, "reference_model_path", None))
            if explicit_reference_path is None:
                target_model_path = self._resolve_target_model_path(target_cfg_copy)
                if target_model_path is not None:
                    setattr(draft_config, "reference_model_path", target_model_path)
            model = AutoDraftModel.from_config(
                draft_config,
                trust_remote_code=spec.trust_remote_code,
                torch_dtype=self.runtime_ctx.torch_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported draft_model.init={spec.init}. Expected one of ['pretrained', 'from_config']."
            )
        self._draft_config_obj = getattr(model, "config", None)
        return model

    def compute_step_loss(
        self,
        target_view: TargetRuntimeView,
    ) -> LossOutput:
        teacher_signals = self.extract_teacher_signals(target_view=target_view)
        draft_request = self.build_draft_forward_request(teacher_signals=teacher_signals)
        draft_output = self.forward_draft(draft_request=draft_request)
        loss_output = self.compute_draft_loss(
            draft_output=draft_output,
            teacher_signals=teacher_signals,
        )
        if "speculator_loss" not in loss_output.metrics:
            loss_output.metrics["speculator_loss"] = float(loss_output.total_loss.detach().float().item())
        return loss_output

    @staticmethod
    def _normalize_forward_output(raw_output) -> DraftForwardOutput:
        if isinstance(raw_output, DraftForwardOutput):
            return raw_output
        if torch.is_tensor(raw_output):
            return DraftForwardOutput(hidden_states=raw_output, raw_output=raw_output)
        if isinstance(raw_output, (tuple, list)):
            if not raw_output:
                return DraftForwardOutput(raw_output=raw_output)
            first = raw_output[0]
            if torch.is_tensor(first):
                return DraftForwardOutput(logits=first, raw_output=raw_output)
            return DraftForwardOutput(raw_output=raw_output)
        if isinstance(raw_output, dict):
            logits = raw_output.get("logits", None)
            hidden = raw_output.get("hidden_states", raw_output.get("last_hidden_state", None))
            if isinstance(hidden, (tuple, list)) and hidden:
                hidden = hidden[-1]
            return DraftForwardOutput(logits=logits, hidden_states=hidden, raw_output=raw_output)

        logits = getattr(raw_output, "logits", None)
        hidden = getattr(raw_output, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(raw_output, "hidden_states", None)
            if isinstance(hidden, (tuple, list)) and hidden:
                hidden = hidden[-1]
        return DraftForwardOutput(logits=logits, hidden_states=hidden, raw_output=raw_output)

    def forward_draft(self, draft_request: DraftForwardRequest) -> DraftForwardOutput:
        if self.draft_model is None:
            raise RuntimeError("Draft model is not initialized")
        raw_output = self.draft_model(*draft_request.args, **draft_request.kwargs)
        return self._normalize_forward_output(raw_output)

    def get_draft_trainable_params(self):
        if self.draft_model is None:
            return []
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()

        def collect(module: Optional[torch.nn.Module]):
            if module is None:
                return
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)

        collect(self.draft_model)
        return params

    def get_draft_module(self):
        return self.draft_model

    def get_draft_config_obj(self):
        return self._draft_config_obj

    def register_trainable_module(self, module: Optional[torch.nn.Module]) -> None:
        del module
        raise RuntimeError(
            "Strategy-owned trainable modules are not allowed. "
            "Please move trainable components into draft model implementation."
        )

    @staticmethod
    def _resolve_labels(target_view: TargetRuntimeView) -> Optional[torch.Tensor]:
        if target_view.labels is not None:
            return target_view.labels
        return target_view.input_ids

    @staticmethod
    def _resolve_loss_mask(target_view: TargetRuntimeView) -> Optional[torch.Tensor]:
        return target_view.loss_mask

    @staticmethod
    def _next_token_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape [B, S, V], got {tuple(logits.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"Expected labels shape [B, S], got {tuple(labels.shape)}")

        seq = min(logits.shape[1], labels.shape[1])
        if seq < 2:
            zero = logits.new_zeros(())
            return zero, zero

        logits_shifted = logits[:, : seq - 1, :].contiguous()
        labels_shifted = labels[:, 1:seq].contiguous()
        flat_logits = logits_shifted.view(-1, logits_shifted.shape[-1])
        flat_labels = labels_shifted.view(-1)

        per_token = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
        valid = (flat_labels != ignore_index).float()
        if loss_mask is not None:
            mask_seq = min(loss_mask.shape[1], seq)
            mask_shifted = loss_mask[:, 1:mask_seq].contiguous().view(-1).float()
            valid = valid * mask_shifted
            if mask_seq < seq:
                valid = valid[: per_token.numel()]

        denom = valid.sum().clamp_min(1e-6)
        loss = (per_token * valid).sum() / denom

        with torch.no_grad():
            preds = flat_logits.argmax(dim=-1)
            correct = ((preds == flat_labels).float() * valid).sum()
            acc = correct / denom
        return loss, acc

    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        del target_view
        raise NotImplementedError

    def build_draft_forward_request(
        self,
        teacher_signals: dict[str, Any],
    ) -> DraftForwardRequest:
        del teacher_signals
        raise NotImplementedError

    def compute_draft_loss(
        self,
        draft_output: DraftForwardOutput,
        teacher_signals: dict[str, Any],
    ) -> LossOutput:
        del draft_output, teacher_signals
        raise NotImplementedError
