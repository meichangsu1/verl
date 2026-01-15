# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os

import torch
import torch.distributed as dist
from torch import nn

from verl.models.speculator.mlp_speculator import MLPSpeculator, MLPSpeculatorConfig
from verl.trainer.speculators.interface import SpeculatorAdapter
from verl.utils.fsdp_utils import fsdp_version


class MLPSpeculatorAdapter(SpeculatorAdapter):
    def __init__(
        self,
        config,
        model_config,
        device_name,
        device_mesh,
        torch_dtype,
    ):
        self.config = config
        self.model_config = model_config
        self.device_name = device_name
        self.device_mesh = device_mesh
        self.torch_dtype = torch_dtype

        speculator_config = None
        if self.config is not None and hasattr(self.config, "model"):
            speculator_config = getattr(self.config.model, "speculator", None)
        if speculator_config is None:
            speculator_config = getattr(self.model_config, "speculator", None)
        self.speculator_config = speculator_config
        self.has_speculator = self.speculator_config is not None

        if self.config is not None and hasattr(self.config, "model"):
            self.freeze_base_model = getattr(self.config.model, "freeze_base_model", True)
        else:
            self.freeze_base_model = True

        self.speculator = None

    def build_and_attach(self, model, attach_to_model: bool = True):
        if not self.has_speculator:
            return None

        hf_config = self.model_config.hf_config if hasattr(self.model_config, "hf_config") else self.model_config
        hidden_size = hf_config.hidden_size
        vocab_size = hf_config.vocab_size

        speculator_config_dict = {
            "n_predict": self.speculator_config.get("n_predict", 5),
            "emb_dim": self.speculator_config.get("emb_dim", hidden_size),
            "inner_dim": self.speculator_config.get("inner_dim", hidden_size),
            "vocab_size": vocab_size,
            "tie_weights": self.speculator_config.get("tie_weights", False),
            "scale_input": self.speculator_config.get("scale_input", False),
        }

        base_model_name_or_path = None
        if self.config is not None and hasattr(self.config, "model"):
            base_model_name_or_path = getattr(self.config.model, "path", None)
        if base_model_name_or_path is None:
            base_model_name_or_path = getattr(self.model_config, "local_path", None)
        if base_model_name_or_path is None:
            base_model_name_or_path = "unknown"

        speculator_config_dict["base_model_name_or_path"] = base_model_name_or_path
        config_obj = MLPSpeculatorConfig(**speculator_config_dict)
        self.speculator = MLPSpeculator(config_obj)

        if attach_to_model:
            model.speculator = self.speculator

        if self.freeze_base_model:
            for param in model.parameters():
                param.requires_grad = False
            for param in self.speculator.parameters():
                param.requires_grad = True

        self.speculator.to(device=self.device_name, dtype=self.torch_dtype)
        self.speculator.reset_parameters()

        if self.device_mesh.get_rank() == 0:
            print(f"Created MLP speculator with config: {speculator_config_dict}")
            print(f"Freeze base model: {self.freeze_base_model}")

        return self.speculator

    def get_optimizer_params(self, fsdp_model):
        if self.has_speculator and self.freeze_base_model:
            speculator_module = self._get_speculator_module(fsdp_model)
            if speculator_module is not None:
                return speculator_module.parameters()
        return fsdp_model.parameters()

    def _get_speculator_module(self, fsdp_model):
        if fsdp_model is not None and hasattr(fsdp_model, "speculator"):
            return fsdp_model.speculator
        if self.speculator is not None:
            return self.speculator
        return None

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
        if not self.has_speculator:
            return torch.tensor(0.0, device=self.device_name)

        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return torch.tensor(0.0, device=self.device_name)

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        if hidden_states is None:
            with torch.no_grad():
                hidden_out = fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = hidden_out.hidden_states[-1]
        else:
            hidden = hidden_states
        if spec_logits is None:
            spec_logits = self.compute_speculator_logits(fsdp_model, input_ids, hidden)

        n_predict = speculator_module.n_predict
        vocab_size = spec_logits.size(-1)

        spec_loss_accum = 0.0
        loss_mask_matrix = loss_mask.reshape(input_ids.size(0), -1)
        for i in range(n_predict):
            start = i + 2
            length = spec_logits.size(2)
            max_len = min(
                length,
                input_ids.size(1) - start,
                loss_mask_matrix.size(1) - start,
            )
            if max_len <= 0:
                continue
            targets = input_ids[:, start : start + max_len]

            logits_i = spec_logits[i][:, :max_len, :].reshape(-1, vocab_size)
            labels_i = targets.reshape(-1)

            ce_i = loss_fct(logits_i, labels_i)
            mask_i = loss_mask_matrix[:, start : start + max_len].reshape(-1)
            ce_i = ce_i * mask_i
            spec_loss_accum += ce_i.sum() / mask_i.sum().clamp(min=1)

        spec_loss = spec_loss_accum / n_predict
        return spec_loss

    def compute_speculator_logits(self, fsdp_model, input_ids, hidden_states):
        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return None

        if hasattr(hidden_states, "to_local"):
            try:
                from torch.distributed.tensor import DTensor

                if isinstance(hidden_states, DTensor):
                    hidden_states = hidden_states.to_local()
            except Exception:
                hidden_states = hidden_states.to_local()

        if hasattr(input_ids, "to_local"):
            try:
                from torch.distributed.tensor import DTensor

                if isinstance(input_ids, DTensor):
                    input_ids = input_ids.to_local()
            except Exception:
                input_ids = input_ids.to_local()

        n_predict = speculator_module.n_predict
        hidden = hidden_states[:, : -(n_predict + 1), :]
        seq_ids = input_ids[:, 1:]
        pad_ids = torch.zeros(input_ids.size(0), n_predict, dtype=seq_ids.dtype, device=seq_ids.device)
        spec_inds = torch.cat([seq_ids, pad_ids], dim=1)

        spec_logits = speculator_module(hidden, spec_inds)
        return spec_logits

    def save_checkpoint(self, fsdp_model, local_global_step_folder: str):
        if not self.has_speculator:
            return

        speculator_dir = os.path.join(local_global_step_folder, "speculator")
        os.makedirs(speculator_dir, exist_ok=True)

        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return

        state_dict_path = os.path.join(speculator_dir, "pytorch_model.bin")
        fsdp_ver = fsdp_version(fsdp_model)
        if fsdp_ver == 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

            if hasattr(speculator_module, "_fsdp_wrapped_module"):
                FSDP.set_state_dict_type(
                    speculator_module,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                )
                state_dict = speculator_module.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state_dict, state_dict_path)
                    print(f"Saved speculator state dict to {state_dict_path}")
                FSDP.set_state_dict_type(
                    speculator_module,
                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=FullStateDictConfig(),
                )
            else:
                with FSDP.summon_full_params(speculator_module, writeback=False):
                    state_dict = speculator_module.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state_dict, state_dict_path)
                    print(f"Saved speculator state dict to {state_dict_path}")
        elif fsdp_ver == 2:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
            state_dict = get_model_state_dict(fsdp_model, submodules={speculator_module}, options=options)
            if dist.get_rank() == 0:
                torch.save(state_dict, state_dict_path)
                print(f"Saved speculator state dict to {state_dict_path}")
        else:
            state_dict = speculator_module.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, state_dict_path)
                print(f"Saved speculator state dict to {state_dict_path}")

        if torch.distributed.get_rank() == 0:
            speculator_module.config.save(speculator_dir)
            print(f"Saved speculator config to {os.path.join(speculator_dir, 'config.json')}")

    def load_checkpoint(self, fsdp_model, checkpoint_path: str, logger):
        if not self.has_speculator:
            return

        speculator_dir = os.path.join(checkpoint_path, "speculator")
        state_dict_path = os.path.join(speculator_dir, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            if dist.get_rank() == 0:
                logger.warning(
                    f"Warning: No speculator checkpoint found at {state_dict_path}, starting from scratch"
                )
            return

        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return

        fsdp_ver = fsdp_version(fsdp_model)
        if fsdp_ver == 1:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

            if hasattr(speculator_module, "_fsdp_wrapped_module"):
                FSDP.set_state_dict_type(
                    speculator_module,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                )
                speculator_module.load_state_dict(state_dict)
                FSDP.set_state_dict_type(
                    speculator_module,
                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=FullStateDictConfig(),
                )
            else:
                with FSDP.summon_full_params(speculator_module, writeback=True):
                    speculator_module.load_state_dict(state_dict)
        elif fsdp_ver == 2:
            state_dict = torch.load(state_dict_path, map_location="cpu") if dist.get_rank() == 0 else {}
            try:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
            except Exception:
                from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,
                broadcast_from_rank0=True,
                strict=False,
            )
            set_model_state_dict(
                fsdp_model,
                model_state_dict={speculator_module: state_dict},
                options=options,
            )
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            speculator_module.load_state_dict(state_dict)

        if dist.get_rank() == 0:
            logger.info(f"Successfully loaded speculator checkpoint from {state_dict_path}")
