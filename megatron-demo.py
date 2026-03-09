# minimal_megatron_train.py
import os
from functools import partial
from pathlib import Path

import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

TP_SIZE = 2
PP_SIZE = 1
SEQ_LEN = 128
VOCAB_SIZE = 32000
MICRO_BS = 4
TRAIN_STEPS = 20
LR = 3e-4


def initialize_distributed(tp_size: int, pp_size: int) -> torch.device:
    # 允许重复运行时先清理 model parallel 状态
    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )
    return torch.device("cuda", local_rank)


def build_model(device: torch.device) -> torch.nn.Module:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
    ).to(device)

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    model = DistributedDataParallel(config=cfg, ddp_config=ddp_cfg, module=model)
    return model


def build_optimizer(model: torch.nn.Module):
    # 这里的 model_chunks 就是 [当前 rank 持有的 chunk]
    optim_cfg = OptimizerConfig(
        optimizer="adam",
        lr=LR,
        min_lr=LR * 0.1,
        clip_grad=1.0,
        weight_decay=0.1,
        use_distributed_optimizer=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    optimizer = get_megatron_optimizer(config=optim_cfg, model_chunks=[model])
    return optimizer


def random_data_iterator(device: torch.device):
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(MICRO_BS, -1)
    # Megatron 常用 mask 语义: True 表示 mask 掉（上三角）
    attention_mask = torch.triu(
        torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.bool, device=device),
        diagonal=1,
    )
    while True:
        tokens = torch.randint(0, VOCAB_SIZE, (MICRO_BS, SEQ_LEN), device=device, dtype=torch.long)
        labels = tokens.clone()
        loss_mask = torch.ones((MICRO_BS, SEQ_LEN), device=device, dtype=torch.float32)
        yield {
            "tokens": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def forward_step_func(data_iterator, model):
    data = next(data_iterator)

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        losses = output_tensor.float().view(-1)
        mask = loss_mask.float().view(-1)
        loss = torch.sum(losses * mask) / torch.clamp(mask.sum(), min=1.0)
        return loss, {"lm_loss": loss}

    output_tensor = model(
        data["tokens"],
        data["position_ids"],
        data["attention_mask"],
        labels=data["labels"],
    )
    return output_tensor, partial(loss_func, data["loss_mask"])


def save_distributed_checkpoint(checkpoint_dir: str, model: torch.nn.Module):
    base_model = model.module if hasattr(model, "module") else model
    sharded_state_dict = base_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_dir)


def load_distributed_checkpoint(checkpoint_dir: str, model: torch.nn.Module):
    base_model = model.module if hasattr(model, "module") else model
    sharded_state_dict = base_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_dir)
    base_model.load_state_dict(checkpoint)


def main():
    device = initialize_distributed(TP_SIZE, PP_SIZE)
    model_parallel_cuda_manual_seed(1234)

    model = build_model(device)
    optimizer = build_optimizer(model)
    data_iter = random_data_iterator(device)
    forward_backward_func = get_forward_backward_func()

    rank = torch.distributed.get_rank()

    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=model,
            num_microbatches=1,
            seq_length=SEQ_LEN,
            micro_batch_size=MICRO_BS,
            decoder_seq_length=SEQ_LEN,
            forward_only=False,
        )

        finalize_model_grads([model])
        update_successful, grad_norm, _ = optimizer.step()

        if rank == 0:
            print(f"step={step} update={update_successful} grad_norm={grad_norm} losses={losses_reduced}")

    ckpt_dir = str(Path.cwd() / "mcore_ckpt")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    save_distributed_checkpoint(ckpt_dir, model)
    load_distributed_checkpoint(ckpt_dir, model)

    if rank == 0:
        print(f"checkpoint save/load OK: {ckpt_dir}")

    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
