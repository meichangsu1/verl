# Eagle3 Minimal Draft Template

This directory is a minimal local draft-model package for VERL spec decode.

## Files

- `config.py`: `Eagle3DraftConfig` (`model_type = "eagle3_draft"`)
- `modeling_eagle3.py`: `Eagle3DraftModel` registered by `@register_draft_model("eagle3_draft")`
- `config.json`: model structure config

The draft model owns all trainable parameters for hidden projection and logits head.
`strategy` only assembles teacher signals and does not create trainable layers.

## Usage

Set your training config:

```yaml
model:
  spec_decode:
    strategy:
      name: eagle3
    draft_model:
      path: /absolute/path/to/examples/sft/gsm8k/draft_models/eagle3_minimal
      trust_remote_code: false
```

At load time, `AutoDraftModel` will auto-discover `config.py` and `modeling*.py` from this path.

## Notes

- This template only keeps the Eagle3 structure idea (fuse input embedding + hidden, then decoder blocks).
- The teacher hidden input contract is fixed to `3 * hidden_size` (concatenate three selected teacher layers), then project to `hidden_size` before decoder processing.
- It intentionally does not copy SpecForge distributed or custom attention backends.
- Draft embedding/lm_head bootstrap defaults to target model path (resolved by strategy during `init=from_config`).
- Optional knobs in draft `config.json`:
  - `freeze_reference_embed_tokens` / `freeze_reference_lm_head`: whether to freeze loaded weights
  - `reference_trust_remote_code`: trust-remote-code flag used for loading bootstrap weights
