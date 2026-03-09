# Spec Decode Training (Strategy Architecture)

本文档描述当前 VERL 中投机解码草稿模型训练架构（FSDP + Megatron 统一）。

## 目标

- 训练流程与算法解耦：Engine 负责训练循环，Strategy 负责编排算法差异。
- 接入新草稿模型低成本：用户只需提供
  1. 草稿模型结构/配置（支持 `name/fqn/path+name` 加载）
  2. loss 计算逻辑（通过 strategy 钩子实现）
- 统一配置入口：`model.spec_decode.*`。

## 当前目录结构

```text
verl/trainer/speculators/
├── strategy_interface.py      # 统一接口与数据契约
├── base_strategy.py           # 模板方法与通用 CE loss
├── strategy_registry.py       # strategy 注册与加载
├── lstm_strategy.py           # LSTM strategy
├── mlp_strategy.py            # MLP strategy
├── eagle3_strategy.py         # EAGLE3 strategy
└── dflash_strategy.py         # DFlash strategy
```

## 核心职责划分

### 1) Engine（训练循环层）

- 文件：
  - `verl/workers/engine/fsdp/transformer_impl.py`
  - `verl/workers/engine/megatron/transformer_impl.py`
- 职责：
  - 运行 target forward
  - 按 `model.spec_decode.target_signals` 抽取 teacher signals，构造 `TargetRuntimeView`
  - 调用 `strategy.compute_step_loss(...)`
  - backward / optimizer step
  - 保存与加载 `speculator/` checkpoint

### 2) Strategy（算法编排层）

- 统一接口：`BaseSpecDecodeStrategy`
- 模板实现：`TemplateSpecDecodeStrategy`
- 子类只实现差异钩子：
  - `build_draft_module`
  - `extract_teacher_signals`
  - `build_draft_inputs`
  - `compute_draft_loss`

### 3) Draft Module（模型结构层）

- 仅负责 `nn.Module` 本身（结构、forward、参数初始化）
- 由 Strategy 在 `initialize(...)` 中构建并绑定

## 数据契约

### `TargetRuntimeView`

Engine 传给 Strategy 的 target 运行时视图，包含：

- `input_ids / attention_mask / position_ids / loss_mask`
- `hidden_by_layer`（按 `target_signals.hidden_layers` 裁剪后的多层 hidden）
- `last_hidden`
- `input_embeddings`（可选）
- `packed_seq_params`（Megatron packed 序列场景）

### `LossOutput`

Strategy 返回统一结构：

- `total_loss: torch.Tensor`
- `metrics: dict[str, float]`
- `aux_losses: dict[str, torch.Tensor]`

## 配置

统一使用：

```yaml
model:
  use_remove_padding: false

  spec_decode:
    strategy:
      name: lstm   # 或 fqn/path+name

    draft_model:   # 可选：纯 AutoModel 草稿模型
      path: /path/to/draft_hf_dir
      auto_class: AutoModel            # AutoModel | AutoModelForCausalLM
      trust_remote_code: true
      init: pretrained                 # pretrained | from_config

    strategy_config:
      n_predict: 3
      method: sum_lstm
      tie_lstm_embs: true
      tie_weights: true

    loss:
      name: next_n_ce
      config: {}

    target_signals:
      hidden_layers: [-1]
      include_input_embeddings: false
```

说明：

- 旧键 `model.speculator` / `model.speculator_adapter` 已移除。
- 如果提供旧键会直接报错，要求迁移到 `model.spec_decode`。
- `strategy_config` 等价于旧 `config`，两者都支持；优先读取 `strategy_config`。
- 当配置 `draft_model` 时，strategy 默认会走 AutoModel 加载草稿模型，teacher-draft 组装逻辑仍在 strategy 内。




1. FSDP Engine：初始化 + 训练一步（spec decode）
```mermaid
sequenceDiagram
    participant TW as TrainingWorker
    participant E as FSDPEngineWithLMHeadAndSpeculator
    participant SR as StrategyRegistry
    participant S as SpecDecodeStrategy
    participant T as Target(FSDP)
    participant D as Draft(FSDP/FSDP2)
    participant O as Optimizer(draft params)

    TW->>E: reset()/initialize()
    E->>E: _build_module()
    E->>SR: build_spec_decode_strategy(model_config)
    SR-->>E: strategy
    E->>S: initialize(target_model, spec_decode_cfg, runtime_ctx)
    S->>S: build_draft_module()
    S->>S: freeze target(default=true)
    S-->>E: draft_module
    E->>T: wrap target with FSDP
    E->>D: wrap draft separately
    E->>S: bind_draft_module(D)
    E->>S: get_draft_trainable_params()
    S-->>E: draft params
    E->>O: build_optimizer(draft params)

    TW->>E: train_batch(data, loss_fn)
    E->>E: optimizer_zero_grad()
    E->>E: forward_backward_batch()

    loop micro_batch
        E->>T: forward(output_hidden_states=True)
        T-->>E: base_out(hidden_states,...)
        E->>E: _build_target_runtime_view()
        E->>S: compute_step_loss(batch, target_view, "train")
        S->>S: extract_teacher_signals()
        S->>S: build_draft_inputs()
        S->>D: forward_draft(...)
        D-->>S: draft_output
        S->>S: compute_draft_loss(...)
        S-->>E: LossOutput(total_loss, metrics)
        E->>E: total_loss.backward()
    end

    E->>O: step()  %% 只更新draft参数

```

2. Megatron Engine：初始化 + 训练一步（spec decode）


DraftOptimizer
Draft(MegatronDDP)
SpecDecodeStrategy
StrategyRegistry
Target Megatron Module(list[chunks])
MegatronEngineWithLMHeadAndSpeculator
TrainingWorker
DraftOptimiz

```mermaid
sequenceDiagram
    participant TW as TrainingWorker
    participant E as MegatronEngineWithLMHeadAndSpeculator
    participant T as Target Megatron Module(list[chunks])
    participant SR as StrategyRegistry
    participant S as SpecDecodeStrategy
    participant D as Draft(MegatronDDP)
    participant DO as DraftOptimizer

    TW->>E: reset()/initialize()
    E->>E: super().initialize()  %% 先建teacher模块/teacher优化器
    E->>E: _build_isolated_draft()
    E->>E: _find_local_post_process_stage()
    E->>SR: build_spec_decode_strategy(model_config)
    SR-->>E: strategy
    E->>S: initialize(target_post_process_stage, spec_decode_cfg, runtime_ctx)
    S-->>E: draft_module
    E->>D: MegatronDDP wrap + broadcast_params()
    E->>S: bind_draft_module(D)
    E->>E: _build_draft_optimizer_and_scheduler()

    TW->>E: train_batch(data, loss_fn)
    E->>E: optimizer_zero_grad()  %% teacher + draft清零
    E->>E: forward_backward_batch()  %% PP调度

    loop micro_batch (PP schedule)
        E->>T: forward_step() -> model_forward_with_hidden
        T-->>E: output(hidden_states, packed_seq_params)
        E->>E: postprocess_micro_batch_func(...)
        E->>E: _build_target_runtime_view()
        E->>S: compute_step_loss(batch, target_view, "train")
        S->>D: draft forward + loss
        S-->>E: spec_loss
        E-->>E: scaled_loss = spec_loss * num_micro_batch
    end

    E->>DO: step()  %% 当前仅step draft优化器

```


3. PPO/GRPO 链路里与 spec decode 的关系（当前实现）

```mermaid
sequenceDiagram
    participant R as RayPPOTrainer
    participant AR as ActorRolloutRefWorker
    participant TW as TrainingWorker(actor)
    participant E as Engine

    R->>AR: update_actor(batch)
    AR->>TW: train_mini_batch()
    loop mini_batch
        TW->>E: train_batch(...)
    end

    Note over AR,TW: 当前默认 actor model_type = language_model
    Note over AR,TW: 默认不会进入 language_model_with_speculator 引擎

    R->>AR: update_weights()
    AR->>E: get_per_tensor_param()
    E-->>AR: target weights only
    AR->>Rollout: update_weights(target only)
    Note over AR,Rollout: 当前不在线同步draft权重

```

-----


0302


```mermaid
classDiagram
direction LR

class TargetSignalRequest {
  +hidden_layers: list[int]
  +include_input_embeddings: bool
  +reuse_target_embed_tokens_module: bool
  +reuse_target_lm_head_module: bool
}

class TargetRuntimeView {
  +input_ids
  +attention_mask
  +position_ids
  +loss_mask
  +hidden_by_layer: dict[int, Any]
  +last_hidden
  +input_embeddings
  +packed_seq_params
  +raw_output
}

class LossOutput {
  +total_loss: Tensor
  +metrics: dict[str,float]
  +aux_losses: dict[str,Tensor]
}

class BaseSpecDecodeStrategy {
  <<abstract>>
  +initialize(target_model, spec_decode_cfg, runtime_ctx)
  +bind_draft_module(draft_module)
  +compute_step_loss(batch, target_view, mode)
  +get_draft_trainable_params()
  +get_draft_module()
  +get_draft_config_obj()
  +get_target_signal_request()
  +get_required_rollout_engine()
  +extract_teacher_signals(batch, target_view, mode)
  +build_draft_inputs(batch, teacher, mode)
  +compute_draft_loss(draft_output, batch, teacher, mode)
}

class TemplateSpecDecodeStrategy {
  +REQUIRED_ROLLOUT_ENGINE
  +compute_step_loss(...)
  +forward_draft(...)
  +get_required_rollout_engine()
  +_compute_next_n_ce_loss(...)
}

class LSTMStrategy
class MLPStrategy
class Eagle3Strategy
class DFlashStrategy

BaseSpecDecodeStrategy <|-- TemplateSpecDecodeStrategy
TemplateSpecDecodeStrategy <|-- LSTMStrategy
TemplateSpecDecodeStrategy <|-- MLPStrategy
TemplateSpecDecodeStrategy <|-- Eagle3Strategy
TemplateSpecDecodeStrategy <|-- DFlashStrategy

class StrategyRegistry {
  <<module>>
  +build_spec_decode_strategy(model_config)
  +validate_strategy_rollout_engine(spec_decode, strategy, rollout_engine)
  +get_required_rollout_engine(model_config, rollout_engine)
  +make_speculator_metadata(required_rollout_engine)
}

class FSDPEngineSpec {
  +spec_decode_strategy
  +target_signal_request
  +required_rollout_engine
  +_build_module()
  +forward_step(...)
}

class MegatronEngineSpec {
  +spec_decode_strategy
  +target_signal_request
  +required_rollout_engine
  +_build_isolated_draft()
  +forward_step(...)
  +postprocess_micro_batch_func(...)
}

class FSDPCheckpointManager {
  +set_speculator(..., speculator_metadata)
  +save_speculator_checkpoint()
}

class MegatronCheckpointManager {
  +set_speculator(..., speculator_metadata)
  +save_speculator_checkpoint()
}

class FSDPActorRolloutRefWorker["ActorRolloutRefWorker (fsdp_workers)"] {
  +_build_rollout()
}

class MegatronActorRolloutRefWorker["ActorRolloutRefWorker (megatron_workers)"] {
  +_build_rollout()
}

FSDPEngineSpec --> BaseSpecDecodeStrategy : uses
MegatronEngineSpec --> BaseSpecDecodeStrategy : uses
FSDPEngineSpec --> TargetSignalRequest
MegatronEngineSpec --> TargetSignalRequest
BaseSpecDecodeStrategy --> TargetRuntimeView
BaseSpecDecodeStrategy --> LossOutput

FSDPEngineSpec --> FSDPCheckpointManager : set_speculator(metadata)
MegatronEngineSpec --> MegatronCheckpointManager : set_speculator(metadata)

FSDPActorRolloutRefWorker ..> StrategyRegistry : get_required_rollout_engine(...)
MegatronActorRolloutRefWorker ..> StrategyRegistry : get_required_rollout_engine(...)
FSDPEngineSpec ..> StrategyRegistry : validate_strategy_rollout_engine(...)
MegatronEngineSpec ..> StrategyRegistry : validate_strategy_rollout_engine(...)

```


```mermaid
sequenceDiagram
participant W as Worker(_build_rollout)
participant R as strategy_registry
participant S as Strategy实例

W->>R: get_required_rollout_engine(model_config, rollout.name)
R->>R: resolve_spec_decode_block()
alt spec_decode 未开启
  R-->>W: None
else spec_decode 开启
  R->>R: build_spec_decode_strategy()
  R->>S: get_required_rollout_engine()
  R->>R: validate_strategy_rollout_engine(...)
  R-->>W: required_engine
end
W->>W: mismatch -> raise ValueError

```


```mermaid
sequenceDiagram
participant E as FSDPEngineWithLMHeadAndSpeculator
participant R as strategy_registry
participant S as Strategy
participant M as TargetModel(FSDP)
participant C as FSDPCheckpointManager

E->>R: build_spec_decode_strategy(model_config)
R-->>E: S
E->>S: initialize(target_model=M, spec_decode_cfg, runtime_ctx)
E->>R: validate_strategy_rollout_engine(spec_decode, S)
R-->>E: required_rollout_engine
E->>S: get_target_signal_request()
E->>S: get_draft_module()
E->>S: bind_draft_module(fsdp_wrapped_draft)

E->>R: make_speculator_metadata(required_rollout_engine)
R-->>E: {"required_rollout_engine": "..."}
E->>C: set_speculator(speculator_module, config_obj, metadata)

loop each train step
  E->>M: forward(output_hidden_states=True)
  E->>E: _build_target_runtime_view(...)
  E->>S: compute_step_loss(batch, target_view, "train")
  S->>S: extract_teacher_signals -> build_draft_inputs -> forward_draft -> compute_draft_loss
  S-->>E: LossOutput
  E-->>E: 返回 total_loss + metrics(train/speculator_loss)
end

```



```mermaid
sequenceDiagram
participant E as MegatronEngineWithLMHeadAndSpeculator
participant R as strategy_registry
participant S as Strategy
participant T as Target(Megatron PP)
participant PP as PP Group
participant C as MegatronCheckpointManager

E->>R: build_spec_decode_strategy(model_config)
R-->>E: S
E->>S: initialize(target_stage, spec_decode_cfg, runtime_ctx)
E->>R: validate_strategy_rollout_engine(spec_decode, S)
R-->>E: required_rollout_engine
E->>S: get_target_signal_request()
E->>E: _refresh_requested_hidden_plan()

E->>R: make_speculator_metadata(required_rollout_engine)
R-->>E: {"required_rollout_engine": "..."}
E->>C: set_speculator(speculator_module, config_obj, host_model, metadata)

loop each micro-batch
  E->>T: forward_step() with local layer hooks
  T-->>E: output(hidden_states/packed_seq_params/...)
  E->>PP: broadcast_from_megatron_pp(local_captured_hidden)
  PP-->>E: gathered_hidden_on_last_stage
  E->>E: _build_target_runtime_view(output + hidden_by_layer)
  E->>S: compute_step_loss(batch, target_view, "train")
  S-->>E: LossOutput
  E-->>E: postprocess_micro_batch_func 返回 scaled_loss
end

```

不同backend engine 、rollout engine下权重更新逻辑：

核心是两层解耦：训练引擎负责导出 teacher 权重，rollout 引擎负责接收并装载。

训练侧（按 engine 区分）

FSDP 路径：在 worker 里从 FSDP 模型取 state_dict（或 LoRA 参数），再做 convert_weight_keys，最后调用 rollout.update_weights(...)。见 fsdp_workers.py:681、fsdp_workers.py:704、fsdp_workers.py:751。
Megatron 路径：在 worker 里通过 bridge 导出权重（export_weights/export_hf_weights/export_adapter_weights），再调用 rollout.update_weights(...)。见 megatron_workers.py:715、megatron_workers.py:748。
通用 engine worker 还有一条分支：若 checkpoint backend 不是 naive，不直接喂 rollout，而是 checkpoint_engine.send_weights(...)。见 engine_workers.py:579。
rollout 侧（按 rollout 引擎区分）

统一接口都是 BaseRollout.update_weights(weights, **kwargs)。见 base.py:52。
vLLM：走 CUDA IPC bucket，远端执行 update_weights_from_ipc。见 vllm_rollout.py:146。
SGLang：按 bucket 切分后调用 sgl_update_weights。见 sglang_rollout.py:179。
TRTLLM：把 tensor 转 IPC handle，汇总后调用 server update_weights。见 trtllm_rollout.py:378。
结论

“不同 engine”的差异主要在teacher 权重如何导出/命名转换。
“不同 rollout 引擎”的差异主要在传输与装载协议。
两者通过统一的 update_weights 契约对接。
另外 spec decode 下还会做 strategy -> required_rollout_engine 校验，防止不兼容组合。见 strategy_registry.py:145。
