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

import dataclasses
import inspect

from verl.trainer.speculators.strategy_interface import BaseSpecDecodeStrategy, TargetRuntimeView


def test_target_runtime_view_has_labels_field():
    field_names = [field.name for field in dataclasses.fields(TargetRuntimeView)]
    assert "labels" in field_names


def test_base_strategy_contract_no_batch_or_mode():
    compute_step_loss_sig = inspect.signature(BaseSpecDecodeStrategy.compute_step_loss)
    assert list(compute_step_loss_sig.parameters.keys()) == ["self", "target_view"]

    extract_teacher_sig = inspect.signature(BaseSpecDecodeStrategy.extract_teacher_signals)
    assert list(extract_teacher_sig.parameters.keys()) == ["self", "target_view"]

    build_forward_sig = inspect.signature(BaseSpecDecodeStrategy.build_draft_forward_request)
    assert list(build_forward_sig.parameters.keys()) == ["self", "teacher_signals"]

    forward_draft_sig = inspect.signature(BaseSpecDecodeStrategy.forward_draft)
    assert list(forward_draft_sig.parameters.keys()) == ["self", "draft_request"]

    compute_loss_sig = inspect.signature(BaseSpecDecodeStrategy.compute_draft_loss)
    assert list(compute_loss_sig.parameters.keys()) == ["self", "draft_output", "teacher_signals"]
