
```mermaid
sequenceDiagram
    participant Script as 训练脚本
    participant Main as main_ppo.py
    participant Runner as TaskRunner(Ray)
    participant Trainer as RayPPOTrainer
    participant Rollout as ActorRolloutWG
    participant Ref as RefPolicyWG
    participant Critic as CriticWG

    Script->>Main: python -m verl.trainer.main_ppo
    Main->>Main: run_ppo(config), ray.init()
    Main->>Runner: runner.run(config)

    Runner->>Runner: add_actor_rollout_worker/add_critic_worker
    Runner->>Trainer: 构造 RayPPOTrainer(...)
    Runner->>Trainer: init_workers()
    Runner->>Trainer: fit()

    loop 每个训练 step
        Trainer->>Rollout: generate_sequences(batch)
        Rollout-->>Trainer: responses + rollout_log_probs

        Trainer->>Trainer: reward 计算
        Trainer->>Rollout: compute_log_prob(old_log_prob) [PPO infer]
        Rollout-->>Trainer: old_log_probs (+entropy)

        opt use_reference_policy
            Trainer->>Ref: compute_ref_log_prob
            Ref-->>Trainer: ref_log_prob
        end

        opt use_critic
            Trainer->>Critic: infer values
            Critic-->>Trainer: values
        end

        Trainer->>Trainer: compute_advantage(...)
        Trainer->>Trainer: adv_estimator == GRPO -> compute_grpo_outcome_advantage

        opt use_critic
            Trainer->>Critic: update_critic
            Critic-->>Trainer: critic metrics
        end

        Trainer->>Rollout: update_actor (PPO update)
        Rollout-->>Trainer: actor metrics

        Trainer->>Trainer: save ckpt / update rollout weights / log metrics
    end

```




```mermaid
sequenceDiagram
    autonumber
    participant T as SFTTrainer/RayTrainer
    participant W as TrainingWorker
    participant E as MegatronEngine
    participant U as megatron_utils
    participant F as gptmodel_forward_no_padding
    participant G as GPTModel(_postprocess patched)
    participant M as MTP layers
    participant L as sft_loss
    participant O as Megatron Optimizer

    Note over T,O: 初始化阶段
    T->>W: reset()
    W->>E: initialize()
    E->>U: check_mtp_config(model_config, engine_config)
    E->>E: _build_megatron_module()
    alt model.mtp.enable == true
        E->>U: patch_engine_mtp(module, model_config)
        Note over U,G: patch GPTModel._postprocess
        opt model.mtp.detach_encoder == true
            U->>M: patch_mtp_layer_get_embeddings()
        end
    end
    E->>E: _build_optimizer() / _build_lr_scheduler()

    Note over T,O: 训练阶段（每个 global batch）
    T->>W: train_mini_batch(data)
    W->>E: train_batch(data)
    E->>E: optimizer_zero_grad()
    E->>E: forward_backward_batch(data)

    loop 每个 micro-batch
        E->>E: forward_step(batch)
        E->>F: forward_fn(..., enable_mtp=model.mtp.enable_train)
        alt enable_mtp=true 且 post_process=true
            F->>G: model(..., labels, loss_mask)
            Note over F,G: labels/loss_mask 预处理后传入模型
        else enable_mtp=false
            F->>G: model(...)（不传 mtp labels/loss_mask）
        end

        G->>G: patched _postprocess()
        opt mtp_in_postprocess
            G->>M: self.mtp(...)
        end
        opt config.mtp_num_layers > 0
            alt 新版 Megatron
                G->>M: process_mtp_loss(...)
            else 兼容 fallback
                loop mtp_layer = 1..N
                    G->>M: roll labels/loss_mask + compute mtp loss
                    M-->>G: MTPLossAutoScaler 注入到 hidden_states
                end
            end
        end
        G-->>F: logits [B,S,V]
        F-->>E: output

        E->>L: loss_function(model_output, data)
        L-->>E: lm_loss + metrics
        E->>E: backward (pipeline schedule)
    end

    E->>U: get_megatron_mtp_loss(n_micro_batch)
    U-->>E: mtp_losses/*
    E->>O: optimizer.step()（主干+MTP参数一起更新）
    E->>O: lr_scheduler.step()
    E-->>W: loss + metrics(含 mtp_losses/*)
    W-->>T: 日志上报

```






```mermaid
sequenceDiagram
    autonumber
    participant U as torchrun/sft_trainer
    participant W as TrainingWorker
    participant E as MegatronEngineWithLMHeadAndSpeculator
    participant T as Target Megatron Module(chunks, all TP ranks)
    participant O as Draft Owner Rank(tp0, cp0, last-pp-stage)
    participant S as Eagle3Strategy
    participant D as Eagle3 DraftModel(MCore DDP, owner only)

    Note over U,E: 初始化阶段
    U->>W: 创建 TrainingWorkerConfig(model_type=language_model_with_speculator)
    W->>E: reset() -> initialize()
    E->>E: _bootstrap_spec_decode()
    E->>T: super.initialize() (构建/包裹 target + optimizer)
    E->>E: _validate_pp_spec_decode_support()
    Note over E: PP>1 直接 fail-fast（当前不支持）
    E->>E: _is_draft_active_on_rank() -> 仅 owner=true
    E->>S: initialize(target_model, spec_decode_cfg, runtime_ctx)
    opt owner rank
        S->>D: build_draft_module(AutoDraftModel.from_pretrained/init)
        E->>E: _wrap_draft_with_megatron_ddp()
        E->>E: _build_draft_optimizer()
        E->>S: bind_draft_module(draft_model)
    end

    Note over U,E: 单次训练步（含 micro-batch）
    U->>W: train_mini_batch(data)
    W->>E: train_batch(data)
    E->>E: optimizer_zero_grad()
    E->>E: forward_backward_batch(...)

    loop 每个 micro-batch（Megatron schedule）
        E->>E: forward_step(batch)
        E->>E: _resolve_target_signal_request()

        E->>T: 在请求层注册 forward_hook
        E->>T: gptmodel_forward_no_padding(...)
        T-->>E: hook 回传各层 hidden（常见 [S,B,H]）
        E->>T: remove hooks
        E->>E: hidden postprocess + dense + detach
        alt TP>1 且 sequence_parallel=true
            E->>T: all_reduce(presence) + all_gather(hidden shard)
            T-->>O: owner 拼接为完整序列 hidden
        else TP=1 或无需 gather
            E->>O: owner 直接使用本地 hidden
        end
        E->>E: 构建 hidden_states_map（仅 owner 非空）

        alt owner rank
            E->>E: _build_target_runtime_view(...)
            E->>S: compute_step_loss(target_view)

            S->>S: extract_teacher_signals()
            S->>S: build_draft_forward_request()

            loop ttt_steps
                S->>D: forward(input_emb + projected_hidden/fused_hidden)
                D-->>S: hidden/logits
                S->>S: 准备下一步 TTT 输入（roll input/label/mask）
            end

            S->>S: compute_draft_loss(多步 CE 汇总)
            S-->>E: LossOutput(total_loss, metrics)
            E->>E: scaled_loss = loss * num_micro_batch
        else 非 owner rank
            E->>E: 返回 zero-loss 占位（维持 pipeline 反向契约）
        end
    end

    E->>E: optimizer_step()
    Note over E: target optimizer 在所有 rank 更新；draft optimizer 仅 owner 更新
    E-->>W: loss + metrics
    W-->>U: 日志/监控输出

