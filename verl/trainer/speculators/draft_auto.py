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

import hashlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


class DraftPretrainedConfig(PretrainedConfig):
    """Base config class for draft models managed by AutoDraftModel."""

    model_type = "verl_draft"


class DraftPreTrainedModel(PreTrainedModel):
    """Base model class for draft models managed by AutoDraftModel."""

    config_class = DraftPretrainedConfig
    base_model_prefix = "draft_model"


class AutoDraftModel:
    """Auto loader for draft models with registry-first, HF AutoModel fallback semantics."""

    _MODEL_TYPE_REGISTRY: dict[str, type[PreTrainedModel]] = {}
    _LOADED_LOCAL_PLUGIN_DIRS: set[str] = set()

    @classmethod
    def register(cls, model_type: str, model_cls: type[PreTrainedModel]) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        if not issubclass(model_cls, PreTrainedModel):
            raise TypeError(
                f"Draft model class must inherit PreTrainedModel, got {model_cls.__name__}"
            )
        cls._MODEL_TYPE_REGISTRY[model_type] = model_cls

    @staticmethod
    def _to_local_dir(path: str) -> Path | None:
        local_path = Path(path).expanduser()
        if local_path.is_dir():
            return local_path.resolve()
        return None

    @staticmethod
    def _iter_local_plugin_files(model_dir: Path) -> list[Path]:
        files: list[Path] = []
        config_py = model_dir / "config.py"
        if config_py.is_file():
            files.append(config_py)
        files.extend(sorted(p for p in model_dir.glob("modeling*.py") if p.is_file()))
        return files

    @staticmethod
    def _ensure_local_package(model_dir: Path) -> str:
        digest = hashlib.md5(str(model_dir.resolve()).encode("utf-8")).hexdigest()
        package_name = f"verl_draft_local_pkg_{digest}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(model_dir)]
            package.__file__ = str(model_dir / "__init__.py")
            sys.modules[package_name] = package
        return package_name

    @staticmethod
    def _import_module_from_file(py_file: Path, *, package_name: str):
        abs_path = str(py_file.resolve())
        module_name = f"{package_name}.{py_file.stem}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {abs_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _register_config_class(cls, config_cls: type[PretrainedConfig]) -> None:
        model_type = getattr(config_cls, "model_type", None)
        if not isinstance(model_type, str) or not model_type:
            return
        try:
            AutoConfig.register(model_type, config_cls)
        except TypeError:
            try:
                AutoConfig.register(model_type, config_cls, exist_ok=True)
            except Exception:
                pass
        except ValueError:
            # model_type already registered by transformers or other modules.
            pass

    @classmethod
    def _register_symbols_from_module(cls, module: Any) -> None:
        for symbol in vars(module).values():
            if not inspect.isclass(symbol):
                continue
            if issubclass(symbol, PretrainedConfig) and symbol is not PretrainedConfig:
                cls._register_config_class(symbol)
            if issubclass(symbol, PreTrainedModel) and symbol is not PreTrainedModel:
                config_cls = getattr(symbol, "config_class", None)
                model_type = getattr(config_cls, "model_type", None) if inspect.isclass(config_cls) else None
                if isinstance(model_type, str) and model_type:
                    cls._MODEL_TYPE_REGISTRY[model_type] = symbol

    @classmethod
    def _maybe_load_local_plugins(cls, pretrained_model_name_or_path: str) -> None:
        model_dir = cls._to_local_dir(pretrained_model_name_or_path)
        if model_dir is None:
            return
        dir_key = str(model_dir)
        if dir_key in cls._LOADED_LOCAL_PLUGIN_DIRS:
            return
        package_name = cls._ensure_local_package(model_dir)
        for py_file in cls._iter_local_plugin_files(model_dir):
            module = cls._import_module_from_file(py_file, package_name=package_name)
            cls._register_symbols_from_module(module)
        cls._LOADED_LOCAL_PLUGIN_DIRS.add(dir_key)

    @classmethod
    def maybe_load_local_plugins(cls, pretrained_model_name_or_path: Optional[str]) -> None:
        if not pretrained_model_name_or_path:
            return
        cls._maybe_load_local_plugins(pretrained_model_name_or_path)

    @staticmethod
    def _maybe_load_config_from_path(
        pretrained_model_name_or_path: str,
        *,
        trust_remote_code: bool,
    ) -> PretrainedConfig | None:
        try:
            return AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        except Exception:
            return None

    @staticmethod
    def _from_hf_config(config: PretrainedConfig, *, trust_remote_code: bool, **config_kwargs):
        try:
            return AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )
        except TypeError:
            try:
                return AutoModelForCausalLM.from_config(config, **config_kwargs)
            except Exception:
                return AutoModel.from_config(config, trust_remote_code=trust_remote_code, **config_kwargs)
        except Exception:
            try:
                return AutoModel.from_config(config, trust_remote_code=trust_remote_code, **config_kwargs)
            except TypeError:
                return AutoModel.from_config(config, **config_kwargs)

    @staticmethod
    def _from_hf_pretrained(
        pretrained_model_name_or_path: str,
        *,
        trust_remote_code: bool,
        torch_dtype: torch.dtype | None,
        **model_kwargs,
    ):
        try:
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        except Exception:
            return AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

    @classmethod
    def _from_registered_model(
        cls,
        *,
        model_cls: type[PreTrainedModel],
        pretrained_model_name_or_path: str,
        config: PretrainedConfig,
        torch_dtype: torch.dtype | None,
        model_kwargs: dict[str, Any],
    ):
        return model_cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        *,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
        **config_kwargs,
    ):
        model_type = getattr(config, "model_type", None)
        if model_type in cls._MODEL_TYPE_REGISTRY:
            model = cls._MODEL_TYPE_REGISTRY[model_type](config, **config_kwargs)
            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
            return model

        model = cls._from_hf_config(
            config,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str],
        *,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
        config: PretrainedConfig | None = None,
        **model_kwargs,
    ):
        has_path = bool(pretrained_model_name_or_path)
        # Rule: if path is provided, always load pretrained weights.
        if has_path:
            assert pretrained_model_name_or_path is not None
            cls._maybe_load_local_plugins(pretrained_model_name_or_path)

            if config is None:
                config = cls._maybe_load_config_from_path(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                )
            if config is not None:
                model_type = getattr(config, "model_type", None)
                if model_type in cls._MODEL_TYPE_REGISTRY:
                    return cls._from_registered_model(
                        model_cls=cls._MODEL_TYPE_REGISTRY[model_type],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        model_kwargs=model_kwargs,
                    )
            return cls._from_hf_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        # No path: initialize from provided config.
        if config is None:
            raise ValueError(
                "draft_model.path is not provided and no config is available. "
                "Please provide model.spec_decode.draft_model.path or a valid draft config."
            )
        return cls.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )


def register_draft_model(model_type: str):
    """Decorator for registering a DraftPreTrainedModel by config.model_type."""

    def _decorator(model_cls: type[PreTrainedModel]):
        AutoDraftModel.register(model_type=model_type, model_cls=model_cls)
        return model_cls

    return _decorator


__all__ = [
    "DraftPretrainedConfig",
    "DraftPreTrainedModel",
    "AutoDraftModel",
    "register_draft_model",
]
