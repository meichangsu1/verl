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

from verl.trainer.speculators.base_strategy import TemplateSpecDecodeStrategy
from verl.trainer.speculators.strategy_interface import (
    DraftForwardOutput,
    DraftForwardRequest,
    LossOutput,
    TargetRuntimeView,
)


class MLPStrategy(TemplateSpecDecodeStrategy):
    """MLP strategy interface skeleton."""

    REQUIRED_ROLLOUT_ENGINE = "vllm"

    def build_draft_module(self, target_model, strategy_cfg: dict[str, Any]):
        del target_model, strategy_cfg
        raise NotImplementedError("Interface skeleton only: MLP draft build logic is intentionally empty.")

    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        del target_view
        raise NotImplementedError("Interface skeleton only: MLP teacher extraction logic is intentionally empty.")

    def build_draft_forward_request(
        self,
        teacher_signals: dict[str, Any],
    ) -> DraftForwardRequest:
        del teacher_signals
        raise NotImplementedError("Interface skeleton only: MLP draft forward request logic is intentionally empty.")

    def compute_draft_loss(
        self,
        draft_output: DraftForwardOutput,
        teacher_signals: dict[str, Any],
    ) -> LossOutput:
        del draft_output, teacher_signals
        raise NotImplementedError("Interface skeleton only: MLP draft loss logic is intentionally empty.")
