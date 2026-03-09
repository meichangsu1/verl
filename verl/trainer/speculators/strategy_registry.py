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

from verl.utils.import_utils import load_class_from_fqn, load_extern_object

SPEC_DECODE_STRATEGY_REGISTRY: dict[str, type] = {}
_BUILTINS_REGISTERED = False


def register_spec_decode_strategy(name: str, cls: type) -> None:
    if not name:
        raise ValueError("Spec decode strategy name must be non-empty")
    SPEC_DECODE_STRATEGY_REGISTRY[name.lower()] = cls


def _register_builtin_spec_decode_strategies() -> None:
    from .dflash_strategy import DFlashStrategy
    from .eagle3_strategy import Eagle3Strategy
    from .lstm_strategy import LSTMStrategy
    from .mlp_strategy import MLPStrategy

    register_spec_decode_strategy("lstm", LSTMStrategy)
    register_spec_decode_strategy("mlp", MLPStrategy)
    register_spec_decode_strategy("eagle3", Eagle3Strategy)
    register_spec_decode_strategy("dflash", DFlashStrategy)


def ensure_builtin_spec_decode_strategies_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _register_builtin_spec_decode_strategies()
    _BUILTINS_REGISTERED = True


def get_spec_decode_strategy_cls(name: str) -> type:
    ensure_builtin_spec_decode_strategies_registered()
    key = name.lower()
    if key not in SPEC_DECODE_STRATEGY_REGISTRY:
        available = sorted(SPEC_DECODE_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown spec decode strategy '{name}'. Available strategies: {available}")
    return SPEC_DECODE_STRATEGY_REGISTRY[key]


def _load_custom_strategy(strategy_config: Any):
    if isinstance(strategy_config, str):
        if "." not in strategy_config:
            return get_spec_decode_strategy_cls(strategy_config)
        return load_class_from_fqn(strategy_config, description="spec decode strategy")

    if hasattr(strategy_config, "get"):
        path = strategy_config.get("path", None)
        name = strategy_config.get("name", None)
        if path and name:
            return load_extern_object(path, name)
        if name:
            return get_spec_decode_strategy_cls(name)
        fqn = strategy_config.get("fqn", None)
        if fqn:
            return load_class_from_fqn(fqn, description="spec decode strategy")
    return None


def resolve_spec_decode_block(model_config: Any) -> dict[str, Any] | None:
    if model_config is None:
        return None
    spec_decode = getattr(model_config, "spec_decode", None)
    if spec_decode is None:
        return None
    if not hasattr(spec_decode, "get"):
        raise ValueError("model.spec_decode must be a mapping")
    return spec_decode


def has_spec_decode(model_config: Any) -> bool:
    return resolve_spec_decode_block(model_config) is not None


def build_spec_decode_strategy(model_config: Any):
    ensure_builtin_spec_decode_strategies_registered()
    spec_decode = resolve_spec_decode_block(model_config)
    if spec_decode is None:
        raise ValueError("model.spec_decode is not configured")
    strategy_cfg = spec_decode.get("strategy", None)
    if strategy_cfg is None:
        raise ValueError("model.spec_decode.strategy is required")
    strategy_cls = _load_custom_strategy(strategy_cfg)
    if strategy_cls is None:
        raise ValueError("model.spec_decode.strategy is invalid. Use strategy.name/fqn/path+name or string alias.")
    return strategy_cls()


def _normalize_rollout_engine(engine: Any) -> str | None:
    if engine is None:
        return None
    normalized = str(engine).strip().lower()
    return normalized if normalized else None


def validate_strategy_rollout_engine(
    spec_decode: dict[str, Any] | None,
    strategy: Any,
    rollout_engine: str | None = None,
) -> str:
    get_required_fn = getattr(strategy, "get_required_rollout_engine", None)
    required_engine = _normalize_rollout_engine(get_required_fn() if get_required_fn else "vllm")
    if not required_engine:
        raise ValueError(f"Spec decode strategy {strategy.__class__.__name__} returned empty rollout engine")

    expected_engine = spec_decode.get("rollout_engine", None) if hasattr(spec_decode, "get") else None
    expected_engine = _normalize_rollout_engine(expected_engine)
    if expected_engine is not None and expected_engine != required_engine:
        raise ValueError(
            f"model.spec_decode.rollout_engine={expected_engine} conflicts with strategy requirement: {required_engine}"
        )

    rollout_engine = _normalize_rollout_engine(rollout_engine)
    if rollout_engine is not None and rollout_engine != required_engine:
        raise ValueError(
            f"Spec decode strategy {strategy.__class__.__name__} requires rollout engine "
            f"'{required_engine}', but got '{rollout_engine}'."
        )
    return required_engine


def make_speculator_metadata(required_rollout_engine: str | None) -> dict[str, Any] | None:
    normalized = _normalize_rollout_engine(required_rollout_engine)
    if normalized is None:
        return None
    return {"required_rollout_engine": normalized}


def get_required_rollout_engine(model_config: Any, rollout_engine: str | None = None) -> str | None:
    spec_decode = resolve_spec_decode_block(model_config)
    if spec_decode is None:
        return None
    strategy = build_spec_decode_strategy(model_config)
    return validate_strategy_rollout_engine(spec_decode=spec_decode, strategy=strategy, rollout_engine=rollout_engine)


__all__ = [
    "SPEC_DECODE_STRATEGY_REGISTRY",
    "register_spec_decode_strategy",
    "ensure_builtin_spec_decode_strategies_registered",
    "get_spec_decode_strategy_cls",
    "build_spec_decode_strategy",
    "validate_strategy_rollout_engine",
    "make_speculator_metadata",
    "get_required_rollout_engine",
    "resolve_spec_decode_block",
    "has_spec_decode",
]
