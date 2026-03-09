from __future__ import annotations

from typing import Any, Optional

from verl.trainer.speculators.draft_auto import DraftPretrainedConfig


class Eagle3DraftConfig(DraftPretrainedConfig):
    """
    Minimal Eagle3 draft config.

    This config is designed for training-time draft module usage in VERL spec decode.
    """

    model_type = "eagle3_draft"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 32,
        rms_norm_eps: float = 1e-6,
        aux_hidden_layers: Optional[list[int]] = None,
        reference_trust_remote_code: bool = False,
        freeze_reference_embed_tokens: bool = False,
        freeze_reference_lm_head: bool = False,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.rms_norm_eps = float(rms_norm_eps)
        self.aux_hidden_layers = list(aux_hidden_layers) if aux_hidden_layers is not None else []
        self.reference_trust_remote_code = bool(reference_trust_remote_code)
        self.freeze_reference_embed_tokens = bool(freeze_reference_embed_tokens)
        self.freeze_reference_lm_head = bool(freeze_reference_lm_head)
