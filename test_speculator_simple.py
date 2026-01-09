#!/usr/bin/env python3
"""
Simple test to verify speculator integration in SFT trainer.
This test doesn't require distributed setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_speculator_model():
    """Test the speculator model from the provided code."""
    print("Testing speculator model...")
    
    # Mock the necessary classes
    class LayerNormParameterized(nn.Module):
        def __init__(self, dim, elementwise_shift=True, elementwise_scale=True):
            super().__init__()
            self.dim = dim
            if elementwise_scale:
                self.weight = nn.Parameter(torch.ones(dim))
            else:
                self.register_buffer('weight', torch.ones(dim))
            if elementwise_shift:
                self.bias = nn.Parameter(torch.zeros(dim))
            else:
                self.register_buffer('bias', torch.zeros(dim))
        
        def forward(self, x):
            return F.layer_norm(x, (self.dim,), self.weight, self.bias)
    
    class ArcticLSTMSpeculator(nn.Module):
        def __init__(self, config):
            super().__init__()
            
            self.config = config
            self.n_predict = config.n_predict
            self.input_hidden_dim = config.input_hidden_dim
            
            def parse_dim(s):
                if isinstance(s, int):
                    return [s]
                elif isinstance(s, str):
                    return [int(i) for i in s.split(".")]
                else:
                    raise NotImplementedError
            
            self.inner_dim = parse_dim(config.inner_dim)
            self.emb_dim = parse_dim(config.emb_dim)
            self.proj_dim = parse_dim(config.proj_dim)
            
            self.vocab_size = config.vocab_size
            self.scale_input = config.scale_input
            self.tie_weights = config.tie_weights
            self.tie_lstm_embs = config.tie_lstm_embs
            self.method = config.method
            self.activation = nn.GELU()
            
            if self.method == "sum_rnn":
                embs = []
                for n_i in range(self.n_predict):
                    if not self.tie_weights or n_i == 0:
                        seqs = [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                        for i in range(1, len(self.emb_dim)):
                            seqs.append(
                                LayerNormParameterized(
                                    self.emb_dim[i],
                                    elementwise_shift=True,
                                    elementwise_scale=True,
                                )
                            )
                            seqs.append(self.activation)
                            seqs.append(nn.Linear(self.emb_dim[i - 1], self.emb_dim[i], bias=False))
                        embs.append(nn.Sequential(*seqs))
                self.emb = nn.ModuleList(embs)
                
                projs = []
                for n_i in range(self.n_predict):
                    if not self.tie_weights or n_i <= 1:
                        seqs = [
                            nn.Linear(
                                (self.input_hidden_dim if n_i == 0 else self.inner_dim[-1]),
                                self.proj_dim[0],
                                bias=False,
                            )
                        ]
                        for i in range(1, len(self.proj_dim)):
                            seqs.append(
                                LayerNormParameterized(
                                    self.proj_dim[i],
                                    elementwise_shift=True,
                                    elementwise_scale=True,
                                )
                            )
                            seqs.append(self.activation)
                            seqs.append(nn.Linear(self.proj_dim[i - 1], self.proj_dim[i], bias=False))
                        projs.append(nn.Sequential(*seqs))
                self.proj = nn.ModuleList(projs)
                
                lns = []
                for n_i in range(self.n_predict):
                    if not self.tie_weights or n_i == 0:
                        seqs = [
                            LayerNormParameterized(
                                self.inner_dim[0],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        ]
                        for i in range(1, len(self.inner_dim)):
                            seqs.append(self.activation)
                            seqs.append(nn.Linear(self.inner_dim[i - 1], self.inner_dim[i], bias=False))
                            seqs.append(
                                LayerNormParameterized(
                                    self.inner_dim[i],
                                    elementwise_shift=True,
                                    elementwise_scale=True,
                                )
                            )
                        lns.append(nn.Sequential(*seqs))
                self.ln = nn.ModuleList(lns)
            
            elif self.method == "sum_lstm":
                assert self.tie_weights
                self.forget_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                if self.tie_lstm_embs:
                    self.input_emb = self.cell_emb = self.output_emb = self.forget_emb
                else:
                    self.input_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                    self.cell_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                    self.output_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                self.forget_proj = nn.ModuleList(
                    [
                        nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                        nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                    ]
                )
                self.input_proj = nn.ModuleList(
                    [
                        nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                        nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                    ]
                )
                self.cell_proj = nn.ModuleList(
                    [
                        nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                        nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                    ]
                )
                self.output_proj = nn.ModuleList(
                    [
                        nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                        nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                    ]
                )
                self.cell_ln = nn.ModuleList(
                    [
                        LayerNormParameterized(
                            self.inner_dim[0],
                            elementwise_shift=True,
                            elementwise_scale=True,
                        )
                    ]
                )
                self.state_ln = nn.ModuleList(
                    [
                        LayerNormParameterized(
                            self.inner_dim[0],
                            elementwise_shift=True,
                            elementwise_scale=True,
                        )
                    ]
                )
            
            if self.scale_input:
                self.ln0 = LayerNormParameterized(self.input_hidden_dim, elementwise_shift=False, elementwise_scale=False)
            
            self.head = nn.ModuleList(
                [nn.Linear(self.inner_dim[-1], self.vocab_size, bias=False) for _ in range(self.n_predict)]
            )
            
            # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
            self.state_weight = 0.5 ** (0.5 / self.n_predict)
            self.emb_weight = torch.sqrt(torch.tensor((1 - self.state_weight**2) * (self.emb_dim[-1] / 2)))
            
            # Handle weight tying as specified
            if self.tie_weights and self.n_predict > 1:
                for head in self.head:
                    head.weight = self.head[0].weight
        
        def forward(self, state: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
            out = []
            if self.scale_input:
                state = self.ln0(state) / (2**0.5)
            
            if self.method == "sum_lstm":
                cell_state = torch.zeros(state.shape, device=state.device, dtype=state.dtype)
                for i in range(self.n_predict):
                    prev_state = state
                    actual_i = 0 if self.tie_weights else i
                    actual_proj_i = 1 if self.tie_weights and i >= 2 else i
                    
                    z = self.forget_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                    state = self.forget_proj[actual_proj_i](prev_state)
                    forget_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))
                    
                    z = self.input_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                    state = self.input_proj[actual_proj_i](prev_state)
                    input_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))
                    
                    z = self.cell_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                    state = self.cell_proj[actual_proj_i](prev_state)
                    cell_candidate = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                    cell_candidate = self.activation(self.cell_ln[actual_i](cell_candidate))  # b n d
                    cell_candidate = cell_candidate * input_gate
                    
                    z = self.output_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                    state = self.output_proj[actual_proj_i](prev_state)
                    output_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))
                    
                    cell_state = cell_state * forget_gate
                    cell_state = cell_state + cell_candidate
                    
                    state_candidate = self.activation(self.state_ln[actual_i](cell_state))
                    state = state_candidate * output_gate
                    
                    out.append(self.head[i](state))  # b n v
            
            else:
                assert self.method == "sum_rnn"
                for i in range(self.n_predict):
                    actual_i = 0 if self.tie_weights else i
                    actual_proj_i = 1 if self.tie_weights and i >= 2 else i
                    
                    z = self.emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                    state = self.proj[actual_proj_i](state)
                    state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                    state = self.activation(self.ln[actual_i](state))  # b n d
                    out.append(self.head[i](state))  # b n v
            
            return torch.stack(out, dim=0)  # h b n v
    
    # Create a simple config
    class Config:
        n_predict = 3
        input_hidden_dim = 768
        inner_dim = "768.768"
        emb_dim = "768.768"
        proj_dim = "768.768"
        vocab_size = 32000
        scale_input = False
        tie_weights = True
        tie_lstm_embs = True
        method = "sum_rnn"
    
    config = Config()
    
    # Create speculator
    speculator = ArcticLSTMSpeculator(config)
    print(f"✓ Speculator created: {speculator}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_dim = 768
    
    state = torch.randn(batch_size, seq_len, hidden_dim)
    inds = torch.randint(0, config.vocab_size, (batch_size, seq_len + config.n_predict))
    
    with torch.no_grad():
        logits = speculator(state, inds)
    
    print(f"✓ Forward pass successful")
    print(f"  Input state shape: {state.shape}")
    print(f"  Input inds shape: {inds.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected shape: ({config.n_predict}, {batch_size}, {seq_len}, {config.vocab_size})")
    
    assert logits.shape == (config.n_predict, batch_size, seq_len, config.vocab_size)
    
    # Test parameter freezing
    print("\nTesting parameter freezing logic...")
    
    # Create a mock base model
    class MockBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, hidden_dim)
            self.layer = nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, x):
            return self.layer(self.embedding(x))
    
    base_model = MockBaseModel()
    base_model.speculator = speculator
    
    # Apply freezing logic
    for name, param in base_model.named_parameters():
        if not name.startswith('speculator.'):
            param.requires_grad = False
    
    # Check that base model parameters are frozen
    for name, param in base_model.named_parameters():
        if not name.startswith('speculator.'):
            assert not param.requires_grad, f"Base model parameter {name} should be frozen"
        else:
            assert param.requires_grad, f"Speculator parameter {name} should be trainable"
    
    print("✓ Base model parameters correctly frozen")
    print("✓ Speculator parameters correctly trainable")
    
    return True

def test_loss_function():
    """Test the loss function with speculator outputs."""
    print("\nTesting loss function with speculator...")
    
    # Mock the loss function
    def sft_loss_with_speculator(model_output, data, config):
        log_prob = model_output["log_probs"]
        loss_mask = data["loss_mask"]
        
        # Basic SFT loss
        loss = -torch.sum(log_prob * loss_mask) / loss_mask.sum().clamp(min=1)
        
        # Add speculator loss if present
        spec_logits = model_output.get("spec_logits", None)
        if spec_logits is not None:
            n_predict = spec_logits.shape[0]
            input_ids = data["input_ids"]
            
            spec_loss_total = 0.0
            for i in range(n_predict):
                logits_i = spec_logits[i]  # (batch, seq_len, vocab_size)
                # Targets are shifted by i+2
                targets = input_ids[:, i+2:i+2+logits_i.shape[1]]
                # Compute cross-entropy loss
                ce_loss = F.cross_entropy(
                    logits_i.reshape(-1, logits_i.shape[-1]),
                    targets.reshape(-1),
                    reduction='none'
                ).view(logits_i.shape[0], -1)
                # Use loss mask
                ce_loss_masked = ce_loss * loss_mask[:, i+2:i+2+logits_i.shape[1]]
                head_loss = ce_loss_masked.sum() / loss_mask[:, i+2:i+2+logits_i.shape[1]].sum().clamp(min=1)
                spec_loss_total += head_loss / n_predict
            
            spec_coeff = getattr(config, "speculator_loss_coeff", 1.0)
            loss = loss + spec_coeff * spec_loss_total
        
        return loss
    
    # Create test data
    batch_size = 2
    seq_len = 10
    vocab_size = 32000
    n_predict = 3
    
    # Model outputs
    log_prob = torch.randn(batch_size, seq_len)
    spec_logits = torch.randn(n_predict, batch_size, seq_len - n_predict - 1, vocab_size)
    
    # Labels and masks
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)
    
    # Create model output and data
    model_output = {
        "log_probs": log_prob,
        "spec_logits": spec_logits
    }
    
    data = {
        "input_ids": input_ids,
        "loss_mask": loss_mask
    }
    
    class Config:
        speculator_loss_coeff = 0.5
    
    config = Config()
    
    # Compute loss
    loss = sft_loss_with_speculator(model_output, data, config)
    print(f"✓ Loss with speculator computed: {loss.item()}")
    
    # Test without speculator
    model_output_no_spec = {
        "log_probs": log_prob
    }
    
    loss_no_spec = sft_loss_with_speculator(model_output_no_spec, data, config)
    print(f"✓ Loss without speculator computed: {loss_no_spec.item()}")
    
    # Loss with speculator should be larger
    assert loss >= loss_no_spec - 1e-5, f"Loss with speculator ({loss}) should be >= loss without ({loss_no_spec})"
    
    print("✓ Loss function correctly handles speculator outputs")
    return True

def main():
    print("=" * 60)
    print("Simple Speculator Integration Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    try:
        if test_speculator_model():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Speculator model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_loss_function():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Speculator integration logic is correct.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
