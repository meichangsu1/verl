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

import importlib

import pytest

from verl.trainer.speculators.strategy_registry import (
    SPEC_DECODE_STRATEGY_REGISTRY,
    build_spec_decode_strategy,
    ensure_builtin_spec_decode_strategies_registered,
)


class _MockModelConfig:
    def __init__(self, spec_decode):
        self.spec_decode = spec_decode


def test_builtin_registry_contains_all_spec_decode_strategies():
    ensure_builtin_spec_decode_strategies_registered()
    registered = set(SPEC_DECODE_STRATEGY_REGISTRY.keys())
    assert {"lstm", "mlp", "eagle3", "dflash"}.issubset(registered)


def test_build_spec_decode_strategy_from_name():
    model_config = _MockModelConfig(spec_decode={"strategy": {"name": "eagle3"}})
    strategy = build_spec_decode_strategy(model_config)
    assert strategy.__class__.__name__ == "Eagle3Strategy"


def test_compat_shims_are_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("verl.trainer.speculators.interface")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("verl.trainer.speculators.registry")
