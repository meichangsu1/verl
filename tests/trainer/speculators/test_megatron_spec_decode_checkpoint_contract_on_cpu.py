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

from pathlib import Path


def _read_megatron_impl() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / "verl/workers/engine/megatron/transformer_impl.py").read_text(encoding="utf-8")


def test_save_checkpoint_saves_draft_before_super():
    content = _read_megatron_impl()
    save_marker = "def save_checkpoint("
    save_idx = content.index(save_marker)
    draft_call_idx = content.index("self._save_draft_checkpoint(local_path=local_path)", save_idx)
    super_call_idx = content.index("super().save_checkpoint(", save_idx)
    assert draft_call_idx < super_call_idx


def test_draft_optimizer_checkpoint_roundtrip_contract_exists():
    content = _read_megatron_impl()
    assert '"optimizer.pt"' in content
    assert 'optimizer_payload["optimizer"] = self.draft_optimizer.state_dict()' in content
    assert 'optimizer_state = optimizer_payload.get("optimizer", None)' in content


def test_draft_uses_megatron_optimizer_contract():
    content = _read_megatron_impl()
    assert "get_megatron_optimizer(model=self.draft_chunks" in content
    assert "register_megatron_training_hooks(self.draft_chunks, self.draft_optimizer)" in content
    assert "torch.optim.AdamW(" not in content
    assert "def _sync_draft_gradients" not in content


def test_pipeline_hidden_aggregation_contract_exists():
    content = _read_megatron_impl()
    assert "broadcast_from_megatron_pp" in content
    assert "all_gather_object(" in content
    assert "required_abs_layer_ids" in content
