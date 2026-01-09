# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Speculator adapter interface and factory.
This keeps trainers decoupled from concrete speculator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from verl.utils.import_utils import load_class_from_fqn, load_extern_object


class SpeculatorAdapter(ABC):
    has_speculator: bool
    freeze_base_model: bool

    @abstractmethod
    def build_and_attach(self, model, attach_to_model: bool = True):
        """Build and optionally attach a speculator module to the model."""

    @abstractmethod
    def get_optimizer_params(self, fsdp_model):
        """Return the parameters to optimize."""

    @abstractmethod
    def compute_speculator_loss(
        self,
        fsdp_model,
        input_ids,
        attention_mask=None,
        position_ids=None,
        loss_mask=None,
        hidden_states=None,
        spec_logits=None,
    ):
        """Compute speculator loss."""

    @abstractmethod
    def save_checkpoint(self, fsdp_model, local_global_step_folder: str):
        """Save speculator checkpoint if applicable."""

    @abstractmethod
    def load_checkpoint(self, fsdp_model, checkpoint_path: str, logger):
        """Load speculator checkpoint if applicable."""


def _load_custom_adapter(speculator_adapter_config: Any):
    if isinstance(speculator_adapter_config, str):
        return load_class_from_fqn(speculator_adapter_config, description="speculator adapter")

    if hasattr(speculator_adapter_config, "get"):
        fqn = speculator_adapter_config.get("fqn", None)
        if fqn:
            return load_class_from_fqn(fqn, description="speculator adapter")
        path = speculator_adapter_config.get("path", None)
        name = speculator_adapter_config.get("name", None)
        if path and name:
            return load_extern_object(path, name)

    return None


def build_speculator_adapter(
    config,
    model_config,
    device_name,
    device_mesh,
    torch_dtype,
):
    speculator_adapter_config = None
    if config is not None and hasattr(config, "model"):
        speculator_adapter_config = getattr(config.model, "speculator_adapter", None)
    if speculator_adapter_config is None and model_config is not None:
        speculator_adapter_config = getattr(model_config, "speculator_adapter", None)

    adapter_cls = _load_custom_adapter(speculator_adapter_config)
    if adapter_cls is None:
        raise ValueError(
            "speculator_adapter is not configured. Please set config.model.speculator_adapter "
            "to a valid adapter class or module path."
        )

    return adapter_cls(
        config=config,
        model_config=model_config,
        device_name=device_name,
        device_mesh=device_mesh,
        torch_dtype=torch_dtype,
    )
