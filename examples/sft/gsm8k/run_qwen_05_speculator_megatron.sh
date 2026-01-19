torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files=/model/ljl/arctic-traing-datasets/data/train.parquet \
  data.val_files=/model/ljl/arctic-traing-datasets/data/test.parquet \
  ++data.pad_mode=right \
  data.micro_batch_size_per_gpu=1 \
  model.path=/model/ljl/Qwen3MoeCustom3  \
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
