set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_speculator_engine.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.micro_batch_size_per_gpu=1 \
  model.path=/path/to/your/model \
  model.use_remove_padding=false \
  ++model.speculator_adapter.fqn=verl.trainer.speculators.lstm_adapter.LSTMSpeculatorAdapter \
  ++model.speculator.n_predict=3 \
  ++model.speculator.method=sum_lstm \
  ++model.speculator.tie_lstm_embs=true \
  ++model.speculator.tie_weights=true \
  engine=megatron \
  optim=megatron \
  engine.tensor_model_parallel_size=1 \
  engine.pipeline_model_parallel_size=1 \
  engine.virtual_pipeline_model_parallel_size=null \
  engine.context_parallel_size=1 \
  engine.use_mbridge=true \
  trainer.total_training_steps=2 \
  trainer.total_epochs=1 \
  trainer.logger=console \
  trainer.default_local_dir=/model/ljl/output/lstm
