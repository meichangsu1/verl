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

from dataclasses import dataclass
from typing import Any

import torch

from verl.trainer.speculators.strategy_interface import BaseSpecDecodeStrategy, StrategyRuntimeContext
from verl.trainer.speculators.strategy_registry import build_spec_decode_strategy, resolve_spec_decode_block
from verl.utils.device import get_device_name


@dataclass
class EngineSpecDecodeBundle:
    config: dict[str, Any]
    strategy: BaseSpecDecodeStrategy
    runtime_context: StrategyRuntimeContext


def resolve_spec_decode_config(model_config: Any, *, require: bool = False) -> dict[str, Any] | None:
    spec_decode = resolve_spec_decode_block(model_config)
    if require and spec_decode is None:
        raise ValueError("model.spec_decode is required for spec decode engine.")
    return spec_decode


def build_strategy_runtime_context(
    *,
    backend: str,
    torch_dtype: torch.dtype,
    supports_packed_seq: bool,
    supports_input_embeddings: bool = True,
    supports_multi_layer_hidden: bool = True,
    target_model_path: str | None = None,
) -> StrategyRuntimeContext:
    return StrategyRuntimeContext(
        backend=backend,
        device_name=get_device_name(),
        torch_dtype=torch_dtype,
        supports_packed_seq=supports_packed_seq,
        supports_input_embeddings=supports_input_embeddings,
        supports_multi_layer_hidden=supports_multi_layer_hidden,
        target_model_path=target_model_path,
    )


def build_engine_spec_decode_bundle(
    *,
    model_config: Any,
    backend: str,
    torch_dtype: torch.dtype,
    supports_packed_seq: bool,
    require: bool = False,
) -> EngineSpecDecodeBundle | None:
    spec_decode = resolve_spec_decode_config(model_config=model_config, require=require)
    if spec_decode is None:
        return None

    target_model_path = getattr(model_config, "local_path", None) or getattr(model_config, "path", None)
    if target_model_path is not None:
        target_model_path = str(target_model_path)

    strategy = build_spec_decode_strategy(model_config)
    runtime_context = build_strategy_runtime_context(
        backend=backend,
        torch_dtype=torch_dtype,
        supports_packed_seq=supports_packed_seq,
        target_model_path=target_model_path,
    )
    return EngineSpecDecodeBundle(config=spec_decode, strategy=strategy, runtime_context=runtime_context)
