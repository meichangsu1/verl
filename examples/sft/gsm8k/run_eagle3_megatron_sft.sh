#!/usr/bin/env bash
set -xeuo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: run_eagle3_megatron_sft.sh <nproc_per_node> <save_path> <draft_model_dir> [extra_hydra_overrides...]"
    exit 1
fi

NPROC_PER_NODE=$1
SAVE_PATH=$2
DRAFT_DIR=$3
shift 3

if [ ! -d "$DRAFT_DIR" ]; then
    echo "draft_model_dir does not exist: $DRAFT_DIR"
    exit 1
fi

DRAFT_DIR="$(cd "$DRAFT_DIR" && pwd)"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
MESSAGES_KEY=${MESSAGES_KEY:-messages}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-4096}
MAX_LENGTH=${MAX_LENGTH:-2048}

LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-50}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}

PROJECT_NAME=${PROJECT_NAME:-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eagle3-megatron-sft}
LOGGER=${LOGGER:-console}

DRAFT_INIT=${DRAFT_INIT:-from_config}
DRAFT_TRUST_REMOTE_CODE=${DRAFT_TRUST_REMOTE_CODE:-false}
TTT_STEPS=${TTT_STEPS:-1}
AUX_HIDDEN_LAYERS=${AUX_HIDDEN_LAYERS:-"[1,-2,-1]"}
REUSE_TARGET_LM_HEAD=${REUSE_TARGET_LM_HEAD:-false}

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    -m verl.trainer.sft_trainer \
    engine=megatron \
    optim=megatron \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.messages_key="$MESSAGES_KEY" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    data.pad_mode=no_padding \
    data.max_length="$MAX_LENGTH" \
    data.num_workers=4 \
    model.path="$MODEL_PATH" \
    model.trust_remote_code=true \
    model.use_remove_padding=true \
    model.spec_decode.strategy.name=eagle3 \
    model.spec_decode.draft_model.path="$DRAFT_DIR" \
    model.spec_decode.draft_model.init="$DRAFT_INIT" \
    model.spec_decode.draft_model.trust_remote_code="$DRAFT_TRUST_REMOTE_CODE" \
    model.spec_decode.strategy_config.ttt_steps="$TTT_STEPS" \
    model.spec_decode.strategy_config.aux_hidden_layers="$AUX_HIDDEN_LAYERS" \
    model.spec_decode.strategy_config.reuse_target_lm_head="$REUSE_TARGET_LM_HEAD" \
    engine.tensor_model_parallel_size="$TP_SIZE" \
    engine.pipeline_model_parallel_size="$PP_SIZE" \
    engine.virtual_pipeline_model_parallel_size="$VPP_SIZE" \
    engine.context_parallel_size="$CP_SIZE" \
    engine.use_mbridge=true \
    optim.lr="$LR" \
    optim.min_lr="$MIN_LR" \
    optim.lr_warmup_steps="$WARMUP_STEPS" \
    optim.lr_decay_style=cosine \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="$LOGGER" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.resume_mode=disable \
    "$@"
