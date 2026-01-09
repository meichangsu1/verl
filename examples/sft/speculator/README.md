# Speculator Training Guide

This guide explains how to train a language model with a speculator for speculative decoding in VERL.

## Overview

Speculative decoding is a technique that uses a smaller, faster "speculator" model to predict multiple tokens ahead, which are then verified by the larger base model. This can significantly speed up inference while maintaining the same output quality.

In VERL, speculator training is integrated into the SFT (Supervised Fine-Tuning) pipeline, allowing you to train a speculator alongside or instead of fine-tuning the base model.

## Training Script

The main training script is `run_speculator_training.sh`. Here's how to use it:

```bash
# Basic usage
./run_speculator_training.sh <nproc_per_node> <save_path> [other_configs...]

# Example with 2 GPUs
./run_speculator_training.sh 2 ./speculator-checkpoints \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet
```

## Speculator Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.speculator.n_predict` | int | 5 | Number of tokens to predict ahead (speculation depth) |
| `model.speculator.inner_dim` | int/str | 1024 | Hidden dimension for speculator internal layers. Can be a single integer or dot-separated list for multiple layers (e.g., "1024.2048") |
| `model.speculator.emb_dim` | int/str | 1024 | Embedding dimension for speculator. Can be a single integer or dot-separated list |
| `model.speculator.proj_dim` | int/str | 1024 | Projection dimension for speculator. Can be a single integer or dot-separated list |
| `model.speculator.scale_input` | bool | false | Whether to scale input hidden states with LayerNorm |
| `model.speculator.tie_weights` | bool | true | Whether to share weights across prediction heads |
| `model.speculator.tie_lstm_embs` | bool | false | Whether to share LSTM embeddings (only for 'sum_lstm' method) |
| `model.speculator.method` | str | 'sum_rnn' | Speculator architecture: 'sum_rnn' (simpler) or 'sum_lstm' (more complex) |
| `model.speculator_loss_coeff` | float | 1.0 | Weight for speculator loss relative to base model loss |
| `model.freeze_base_model` | bool | true | Freeze base model parameters, only train speculator |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optim.lr` | float | 1e-4 | Learning rate for speculator training |
| `trainer.total_epochs` | int | 1 | Number of training epochs |
| `trainer.save_freq` | int | 1000 | Save checkpoint every N steps |
| `trainer.test_freq` | int | 500 | Run validation every N steps |
| `data.micro_batch_size_per_gpu` | int | 4 | Micro-batch size per GPU |

## Architecture Details

### Speculator Types

1. **sum_rnn**: Simpler RNN-based architecture with weight sharing
2. **sum_lstm**: More complex LSTM-based architecture with separate gates

### Training Modes

1. **Base Model Frozen (Recommended)**: Only train the speculator while keeping the base model frozen. This is the default mode (`model.freeze_base_model=true`).

2. **Joint Training**: Train both base model and speculator together. Set `model.freeze_base_model=false`.

## Checkpoint Saving

When training with speculator, checkpoints are saved in a special format:

```
checkpoint_dir/
├── pytorch_model.bin      # Speculator weights only
├── config.json            # Speculator configuration
└── ...                    # Other checkpoint files
```

This is similar to LoRA checkpoints - only the speculator parameters are saved, not the entire base model.

## Inference with Speculator

After training, you can use the speculator for speculative decoding during inference. The trained speculator will be automatically loaded with the base model when using the `language_model_with_speculator` engine type.

## Example Configurations

### Basic Speculator for Qwen2.5-0.5B

```bash
./run_speculator_training.sh 2 ./speculator-qwen \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    model.speculator.n_predict=3 \
    model.speculator.inner_dim=512 \
    model.speculator.emb_dim=512 \
    model.speculator.proj_dim=512 \
    model.speculator.method=sum_rnn
```

### Advanced LSTM Speculator

```bash
./run_speculator_training.sh 4 ./speculator-lstm \
    model.partial_pretrain=meta-llama/Llama-2-7b-hf \
    model.speculator.n_predict=5 \
    model.speculator.inner_dim="1024.2048" \
    model.speculator.emb_dim="1024.2048" \
    model.speculator.proj_dim="1024.2048" \
    model.speculator.method=sum_lstm \
    model.speculator.tie_lstm_embs=true \
    optim.lr=5e-5 \
    trainer.total_epochs=3
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `data.micro_batch_size_per_gpu` or use gradient accumulation.
2. **Slow Training**: The speculator adds computational overhead. Consider using a simpler architecture (`sum_rnn`) or reducing `n_predict`.
3. **Poor Speculator Performance**: Increase speculator dimensions or try the `sum_lstm` method.

### Debug Tips

- Set `trainer.logger=console` to see detailed logs
- Monitor `speculator_loss` metric during training
- Start with small `n_predict` (2-3) and increase gradually

## References

1. [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318)
2. [VERL Documentation](https://github.com/volcengine/verl)
3. [Arctic Speculator Implementation](https://github.com/snowflake-arctic/speculator)
