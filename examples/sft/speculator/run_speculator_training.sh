#!/bin/bash

# Speculator Training Script for VERL
# This script demonstrates how to train a language model with a speculator for speculative decoding

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_speculator_training.sh <nproc_per_node> <save_path> [other_configs...]"
    echo ""
    echo "Example:"
    echo "  ./run_speculator_training.sh 2 ./speculator-checkpoints \\"
    echo "    data.train_files=\$HOME/data/gsm8k/train.parquet \\"
    echo "    data.val_files=\$HOME/data/gsm8k/test.parquet"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=speculator-training \
    trainer.experiment_name=speculator-qwen-2.5-0.5b \
    trainer.logger=console \
    trainer.total_epochs=1 \
    model.speculator.n_predict=5 \
    model.speculator.inner_dim=1024 \
    model.speculator.emb_dim=1024 \
    model.speculator.proj_dim=1024 \
    model.speculator.scale_input=false \
    model.speculator.tie_weights=true \
    model.speculator.tie_lstm_embs=false \
    model.speculator.method=sum_rnn \
    model.speculator_loss_coeff=1.0 \
    model.freeze_base_model=true \
    trainer.save_freq=1000 \
    trainer.test_freq=500 \
    $@

# Explanation of speculator parameters:
# - n_predict: Number of tokens to predict ahead (speculation depth)
# - inner_dim: Hidden dimension for speculator internal layers
# - emb_dim: Embedding dimension for speculator
# - proj_dim: Projection dimension for speculator
# - scale_input: Whether to scale input hidden states
# - tie_weights: Whether to share weights across prediction heads
# - tie_lstm_embs: Whether to share LSTM embeddings
# - method: Speculator architecture ('sum_rnn' or 'sum_lstm')
# - speculator_loss_coeff: Weight for speculator loss relative to base loss
# - freeze_base_model: Freeze base model parameters, only train speculator

# Notes:
# 1. The speculator is trained alongside the base model during SFT
# 2. Base model parameters are frozen by default when speculator is present
# 3. Checkpoints will save only the speculator parameters (similar to LoRA)
# 4. For inference, both base model and speculator are needed
