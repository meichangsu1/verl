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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class StrategyRuntimeContext:
    backend: str
    device_name: str
    torch_dtype: torch.dtype
    supports_packed_seq: bool = False
    supports_input_embeddings: bool = True
    supports_multi_layer_hidden: bool = True
    enable_draft_module: bool = True
    target_model_path: Optional[str] = None


@dataclass
class TargetRuntimeView:
    """Runtime teacher-signal container passed from engine to spec strategy.

    Field formats:
    - input_ids: token ids for the current micro-batch, usually shape [B, S].
    - attention_mask: optional mask aligned with input_ids; common shapes are
      [B, S] or backend-specific expanded mask tensors.
    - position_ids: optional position ids aligned with input_ids, typically
      [B, S] (or [S] for degenerate single-batch paths).
    - loss_mask: optional token-level supervision mask, usually [B, S].
    - labels: optional token labels for CE-style objectives, usually [B, S].
    - hidden_by_layer: mapping {layer_id -> hidden tensor}; each value is
      usually dense [B, S, H] and already detached from teacher autograd.
    - input_embeddings: optional teacher input embeddings, usually [B, S, H].
    - packed_seq_params: optional packed-sequence metadata used by backends
      that run with no-padding/packed layouts.
    - raw_output: raw target model forward output for debugging/compatibility.
    - backend_payload: extra backend metadata (for example {"phase": "train"}).
    """
    input_ids: Any
    attention_mask: Optional[Any]
    position_ids: Optional[Any]
    loss_mask: Optional[Any]
    labels: Optional[Any] = None
    hidden_by_layer: dict[int, Any] = field(default_factory=dict)
    input_embeddings: Optional[Any] = None
    packed_seq_params: Optional[Any] = None
    raw_output: Any = None
    backend_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSignalRequest:
    hidden_layers: list[int] = field(default_factory=lambda: [-1])
    include_input_embeddings: bool = False
    reuse_target_lm_head_module: bool = False


@dataclass
class DraftModelSpec:
    path: Optional[str] = None
    trust_remote_code: bool = False
    init: str = "pretrained"


@dataclass
class DraftForwardRequest:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftForwardOutput:
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    raw_output: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)


class BaseSpecDecodeStrategy(ABC):
    """Unified spec decode strategy contract for both FSDP and Megatron engines."""

    @abstractmethod
    def initialize(
        self,
        target_model,
        spec_decode_cfg: dict[str, Any],
        runtime_ctx: StrategyRuntimeContext,
    ) -> None:
        """Initialize strategy state and build/bind draft model if needed."""
        raise NotImplementedError

    @abstractmethod
    def bind_draft_module(self, draft_module) -> None:
        """Bind wrapped draft module after engine-level wrapping."""
        raise NotImplementedError

    @abstractmethod
    def compute_step_loss(
        self,
        target_view: TargetRuntimeView,
    ) -> LossOutput:
        """Compute one training/eval step loss."""
        raise NotImplementedError

    @abstractmethod
    def get_draft_trainable_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_draft_module(self):
        raise NotImplementedError

    @abstractmethod
    def get_draft_config_obj(self):
        raise NotImplementedError

    def get_target_signal_request(self) -> TargetSignalRequest:
        return TargetSignalRequest(hidden_layers=[-1])

    def get_required_rollout_engine(self) -> str:
        return "vllm"

    @abstractmethod
    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_draft_forward_request(
        self,
        teacher_signals: dict[str, Any],
    ) -> DraftForwardRequest:
        raise NotImplementedError

    @abstractmethod
    def forward_draft(
        self,
        draft_request: DraftForwardRequest,
    ) -> DraftForwardOutput:
        raise NotImplementedError

    @abstractmethod
    def compute_draft_loss(
        self,
        draft_output: DraftForwardOutput,
        teacher_signals: dict[str, Any],
    ) -> LossOutput:
        raise NotImplementedError
