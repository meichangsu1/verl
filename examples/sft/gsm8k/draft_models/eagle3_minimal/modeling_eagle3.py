from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

from verl.trainer.speculators.draft_auto import DraftPreTrainedModel, register_draft_model

from .config import Eagle3DraftConfig

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None


class Eagle3DecoderLayer(nn.Module):
    """
    Minimal Eagle3-like decoder layer.

    Reference idea from SpecForge eagle3:
    - normalize input embedding and hidden states separately
    - concatenate [input_emb, hidden]
    - apply attention + MLP with residual paths
    """

    def __init__(self, config: Eagle3DraftConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        intermediate_size = int(config.intermediate_size)
        eps = float(config.rms_norm_eps)

        self.hidden_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.input_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.fuse_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(
        self,
        *,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        fused = torch.cat([self.input_norm(input_emb), self.hidden_norm(hidden_states)], dim=-1)
        fused = self.fuse_proj(fused)

        key_padding_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            key_padding_mask = attention_mask <= 0

        attn_out, _ = self.self_attn(
            query=fused,
            key=fused,
            value=fused,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attn_norm(hidden_states))
        return hidden_states


@register_draft_model("eagle3_draft")
class Eagle3DraftModel(DraftPreTrainedModel):
    """
    Minimal Eagle3 draft model for VERL spec decode training.

    Expected by Eagle3Strategy:
    - accepts `hidden_states` (concatenated 3 * hidden_size teacher states) in forward kwargs
    - returns hidden states with shape [B, S, H]
    """

    config_class = Eagle3DraftConfig

    def __init__(self, config: Eagle3DraftConfig) -> None:
        super().__init__(config)
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)

        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden_size, padding_idx=config.pad_token_id)
        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3DecoderLayer(config) for _ in range(int(config.num_hidden_layers))])
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, int(config.vocab_size), bias=False)
        self.post_init()
        reference_model_path = getattr(config, "reference_model_path", None)
        if not reference_model_path:
            reference_model_path = getattr(config, "target_model_path", None)
        if reference_model_path:
            self._setup_embeddings_and_lm_heads(
                model_path=str(reference_model_path),
                trust_remote_code=bool(getattr(config, "reference_trust_remote_code", False)),
                freeze_embed_tokens=bool(getattr(config, "freeze_reference_embed_tokens", False)),
                freeze_lm_head=bool(getattr(config, "freeze_reference_lm_head", False)),
            )

    @staticmethod
    def _match_key_by_suffix(keys: list[str], suffix: str) -> Optional[str]:
        matches = [key for key in keys if key == suffix or key.endswith(f".{suffix}")]
        if not matches:
            return None
        matches.sort(key=lambda key: (key.count("."), len(key), key))
        return matches[0]

    @staticmethod
    def _normalize_state_dict(payload):
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict", payload)
            if isinstance(state_dict, dict):
                return state_dict
        return None

    @classmethod
    def _load_model_layers(cls, layer_suffixes: list[str], model_path: str) -> dict[str, torch.Tensor]:
        remaining = list(layer_suffixes)
        loaded: dict[str, torch.Tensor] = {}
        path = Path(model_path)

        def _load_from_safetensors(file_path: Path) -> None:
            if safe_open is None or not file_path.exists():
                return
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                for suffix in list(remaining):
                    matched = cls._match_key_by_suffix(keys, suffix)
                    if matched is None:
                        continue
                    loaded[suffix] = handle.get_tensor(matched)
                    remaining.remove(suffix)

        def _load_from_bin(file_path: Path) -> None:
            if not file_path.exists():
                return
            payload = torch.load(str(file_path), map_location="cpu", weights_only=False)
            state_dict = cls._normalize_state_dict(payload)
            if state_dict is None:
                return
            keys = list(state_dict.keys())
            for suffix in list(remaining):
                matched = cls._match_key_by_suffix(keys, suffix)
                if matched is None:
                    continue
                loaded[suffix] = state_dict[matched]
                remaining.remove(suffix)

        safetensors_index = path / "model.safetensors.index.json"
        bin_index = path / "pytorch_model.bin.index.json"

        if safetensors_index.exists():
            weight_map = json.loads(safetensors_index.read_text(encoding="utf-8")).get("weight_map", {})
            if isinstance(weight_map, dict):
                files = {str(filename) for filename in weight_map.values()}
                for filename in sorted(files):
                    _load_from_safetensors(path / filename)

        if remaining and bin_index.exists():
            weight_map = json.loads(bin_index.read_text(encoding="utf-8")).get("weight_map", {})
            if isinstance(weight_map, dict):
                files = {str(filename) for filename in weight_map.values()}
                for filename in sorted(files):
                    _load_from_bin(path / filename)

        if remaining:
            candidate_files = [
                path / "model.safetensors",
                path / "pytorch_model.bin",
                *sorted(path.glob("*.safetensors")),
                *sorted(path.glob("pytorch_model-*.bin")),
            ]
            seen: set[Path] = set()
            for file_path in candidate_files:
                if file_path in seen:
                    continue
                seen.add(file_path)
                if file_path.suffix == ".safetensors":
                    _load_from_safetensors(file_path)
                elif file_path.suffix == ".bin":
                    _load_from_bin(file_path)
                if not remaining:
                    break

        if remaining:
            raise KeyError(
                f"Failed to locate layer weights {remaining} under {model_path}. "
                "Expected keys ending with requested suffixes."
            )

        return loaded

    def _setup_embeddings_and_lm_heads(
        self,
        *,
        model_path: str,
        trust_remote_code: bool,
        freeze_embed_tokens: bool,
        freeze_lm_head: bool,
    ) -> None:
        source_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if hasattr(source_config, "text_config"):
            source_config = source_config.text_config

        source_hidden_size = int(getattr(source_config, "hidden_size", -1))
        if source_hidden_size != int(self.config.hidden_size):
            raise ValueError(
                f"Source hidden size {source_hidden_size} does not match draft hidden size {self.config.hidden_size}."
            )

        weights = self._load_model_layers(
            ["embed_tokens.weight", "lm_head.weight"],
            model_path=model_path,
        )
        embed_tokens_weight = weights["embed_tokens.weight"]
        lm_head_weight = weights.get("lm_head.weight", embed_tokens_weight)

        expected_embed_shape = tuple(self.embed_tokens.weight.shape)
        if tuple(embed_tokens_weight.shape) != expected_embed_shape:
            raise ValueError(
                f"Loaded embed_tokens shape {tuple(embed_tokens_weight.shape)} does not match {expected_embed_shape}."
            )

        expected_lm_head_shape = tuple(self.lm_head.weight.shape)
        if tuple(lm_head_weight.shape) != expected_lm_head_shape:
            raise ValueError(
                f"Loaded lm_head shape {tuple(lm_head_weight.shape)} does not match {expected_lm_head_shape}."
            )

        self.embed_tokens.weight.data.copy_(embed_tokens_weight.to(dtype=self.embed_tokens.weight.dtype))
        self.lm_head.weight.data.copy_(lm_head_weight.to(dtype=self.lm_head.weight.dtype))

        if freeze_embed_tokens:
            self.embed_tokens.weight.requires_grad = False
        if freeze_lm_head:
            self.lm_head.weight.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del position_ids, kwargs

        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Eagle3DraftModel requires input_ids or input_embeds.")
            input_embeds = self.embed_tokens(input_ids)

        if hidden_states is None:
            raise ValueError("Eagle3DraftModel requires `hidden_states` with shape [B, S, 3 * hidden_size].")
        expected_hidden_size = int(self.config.hidden_size) * 3
        if hidden_states.shape[-1] != expected_hidden_size:
            raise ValueError(
                f"Eagle3DraftModel expects teacher hidden size {expected_hidden_size} (3 * hidden_size), "
                f"but got {hidden_states.shape[-1]}."
            )
        hidden = self.fc(hidden_states)
        for layer in self.layers:
            hidden = layer(input_emb=input_embeds, hidden_states=hidden, attention_mask=attention_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return {
            "logits": logits,
            "last_hidden_state": hidden,
            "hidden_states": (hidden,),
        }

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))
