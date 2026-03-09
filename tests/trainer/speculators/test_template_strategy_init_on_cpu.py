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

import pytest
import torch

from verl.trainer.speculators.base_strategy import TemplateSpecDecodeStrategy
from verl.trainer.speculators.strategy_interface import (
    DraftForwardOutput,
    DraftForwardRequest,
    LossOutput,
    StrategyRuntimeContext,
    TargetRuntimeView,
)


class _ConcreteStrategy(TemplateSpecDecodeStrategy):
    def extract_teacher_signals(self, target_view: TargetRuntimeView) -> dict[str, Any]:
        del target_view
        return {}

    def build_draft_forward_request(self, teacher_signals: dict[str, Any]) -> DraftForwardRequest:
        del teacher_signals
        return DraftForwardRequest()

    def compute_draft_loss(self, draft_output: DraftForwardOutput, teacher_signals: dict[str, Any]) -> LossOutput:
        del draft_output, teacher_signals
        return LossOutput(total_loss=torch.zeros(()))


class _DummyConfig:
    def __init__(self, model_type="dummy", vocab_size=32):
        self.model_type = model_type
        self.vocab_size = vocab_size


class _DummyTargetModel:
    def __init__(self):
        self.config = _DummyConfig(model_type="target_model_type", vocab_size=128)


class _DummyDraftModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class _DummyDraftModelWithParams(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(8, 4, bias=False)


def _runtime_ctx(target_model_path: str | None = None) -> StrategyRuntimeContext:
    return StrategyRuntimeContext(
        backend="cpu",
        device_name="cpu",
        torch_dtype=torch.float32,
        target_model_path=target_model_path,
    )


def test_init_pretrained_by_default(monkeypatch):
    strategy = _ConcreteStrategy()
    called = {"from_pretrained": 0, "from_config": 0}

    def _from_pretrained(*args, **kwargs):
        del args, kwargs
        called["from_pretrained"] += 1
        return _DummyDraftModel(config=_DummyConfig(model_type="loaded"))

    def _from_config(*args, **kwargs):
        del args, kwargs
        called["from_config"] += 1
        raise AssertionError("from_config should not be called for default init mode")

    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_pretrained", _from_pretrained)
    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_config", _from_config)

    strategy.initialize(
        target_model=_DummyTargetModel(),
        spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model"}},
        runtime_ctx=_runtime_ctx(),
    )

    assert called["from_pretrained"] == 1
    assert called["from_config"] == 0


def test_init_from_config_uses_autoconfig(monkeypatch):
    strategy = _ConcreteStrategy()
    called = {"from_pretrained": 0, "from_config": 0, "plugins": None}
    loaded_cfg = _DummyConfig(model_type="draft_model_type")

    def _load_plugins(path):
        called["plugins"] = path

    def _auto_config(path, trust_remote_code):
        del trust_remote_code
        assert path == "/tmp/draft-model"
        return loaded_cfg

    def _from_pretrained(*args, **kwargs):
        del args, kwargs
        called["from_pretrained"] += 1
        raise AssertionError("from_pretrained should not be called for init=from_config")

    def _from_config(config, **kwargs):
        del kwargs
        called["from_config"] += 1
        assert config is loaded_cfg
        return _DummyDraftModel(config=config)

    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.maybe_load_local_plugins", _load_plugins)
    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoConfig.from_pretrained", _auto_config)
    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_pretrained", _from_pretrained)
    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_config", _from_config)

    strategy.initialize(
        target_model=_DummyTargetModel(),
        spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model", "init": "from_config"}},
        runtime_ctx=_runtime_ctx(target_model_path="/tmp/target-model"),
    )

    assert called["from_pretrained"] == 0
    assert called["from_config"] == 1
    assert called["plugins"] == "/tmp/draft-model"
    assert getattr(loaded_cfg, "reference_model_path", None) == "/tmp/target-model"


def test_init_from_config_falls_back_to_target_config(monkeypatch):
    strategy = _ConcreteStrategy()
    target_model = _DummyTargetModel()
    captured = {"config": None}

    def _auto_config(*args, **kwargs):
        del args, kwargs
        raise OSError("missing local config")

    def _from_config(config, **kwargs):
        del kwargs
        captured["config"] = config
        return _DummyDraftModel(config=config)

    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoConfig.from_pretrained", _auto_config)
    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_config", _from_config)

    strategy.initialize(
        target_model=target_model,
        spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model", "init": "from_config"}},
        runtime_ctx=_runtime_ctx(),
    )

    assert captured["config"] is not None
    assert captured["config"] is not target_model.config
    assert captured["config"].model_type == target_model.config.model_type


def test_invalid_draft_model_init_raises():
    strategy = _ConcreteStrategy()

    with pytest.raises(ValueError, match="draft_model.init"):
        strategy.initialize(
            target_model=_DummyTargetModel(),
            spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model", "init": "invalid"}},
            runtime_ctx=_runtime_ctx(),
        )


def test_get_draft_trainable_params_only_from_draft(monkeypatch):
    strategy = _ConcreteStrategy()

    def _from_pretrained(*args, **kwargs):
        del args, kwargs
        return _DummyDraftModelWithParams(config=_DummyConfig(model_type="loaded"))

    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_pretrained", _from_pretrained)
    strategy.initialize(
        target_model=_DummyTargetModel(),
        spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model"}},
        runtime_ctx=_runtime_ctx(),
    )

    params = list(strategy.get_draft_trainable_params())
    draft_params = [p for p in strategy.get_draft_module().parameters() if p.requires_grad]
    assert len(params) == len(draft_params)
    assert {id(p) for p in params} == {id(p) for p in draft_params}


def test_runtime_ctx_disable_draft_build(monkeypatch):
    strategy = _ConcreteStrategy()
    called = {"from_pretrained": 0}

    def _from_pretrained(*args, **kwargs):
        del args, kwargs
        called["from_pretrained"] += 1
        return _DummyDraftModelWithParams(config=_DummyConfig(model_type="loaded"))

    monkeypatch.setattr("verl.trainer.speculators.base_strategy.AutoDraftModel.from_pretrained", _from_pretrained)
    runtime_ctx = _runtime_ctx()
    runtime_ctx.enable_draft_module = False
    strategy.initialize(
        target_model=_DummyTargetModel(),
        spec_decode_cfg={"draft_model": {"path": "/tmp/draft-model"}},
        runtime_ctx=runtime_ctx,
    )

    assert called["from_pretrained"] == 0
    assert strategy.get_draft_module() is None
    assert strategy.get_draft_trainable_params() == []
