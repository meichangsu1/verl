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

"""Spec decode strategy package."""

from .base_strategy import TemplateSpecDecodeStrategy
from .draft_auto import AutoDraftModel, DraftPretrainedConfig, DraftPreTrainedModel, register_draft_model
from .strategy_interface import (
    BaseSpecDecodeStrategy,
    DraftForwardOutput,
    DraftForwardRequest,
    DraftModelSpec,
    LossOutput,
    StrategyRuntimeContext,
    TargetRuntimeView,
    TargetSignalRequest,
)

from .strategy_registry import (
    SPEC_DECODE_STRATEGY_REGISTRY,
    build_spec_decode_strategy,
    ensure_builtin_spec_decode_strategies_registered,
    get_required_rollout_engine,
    has_spec_decode,
    make_speculator_metadata,
    register_spec_decode_strategy,
    resolve_spec_decode_block,
    validate_strategy_rollout_engine,
)
from .engine_helpers import (
    EngineSpecDecodeBundle,
    build_engine_spec_decode_bundle,
    build_strategy_runtime_context,
    resolve_spec_decode_config,
)

ensure_builtin_spec_decode_strategies_registered()

__all__ = [
    "SPEC_DECODE_STRATEGY_REGISTRY",
    "register_spec_decode_strategy",
    "ensure_builtin_spec_decode_strategies_registered",
    "build_spec_decode_strategy",
    "validate_strategy_rollout_engine",
    "make_speculator_metadata",
    "get_required_rollout_engine",
    "resolve_spec_decode_block",
    "has_spec_decode",
    "EngineSpecDecodeBundle",
    "build_engine_spec_decode_bundle",
    "build_strategy_runtime_context",
    "resolve_spec_decode_config",
    "BaseSpecDecodeStrategy",
    "TemplateSpecDecodeStrategy",
    "DraftModelSpec",
    "DraftForwardRequest",
    "DraftForwardOutput",
    "LossOutput",
    "TargetSignalRequest",
    "TargetRuntimeView",
    "StrategyRuntimeContext",
    "DraftPretrainedConfig",
    "DraftPreTrainedModel",
    "AutoDraftModel",
    "register_draft_model",
]
