#!/usr/bin/env python3
"""
Test to compare our loss implementation with ArcticLSTMSpeculatorTrainer loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_loss_calculation():
    """Test that our loss calculation matches the reference implementation."""
    print("Testing loss calculation comparison...")
    
    # Simulate reference implementation
    batch_size = 2
    seq_len = 10
    vocab_size = 32000
    n_predict = 3
    hidden_dim = 768
    
    # Create dummy data
    torch.manual_seed(42)
    
    # Reference implementation inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + n_predict))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len + n_predict))
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Simulate speculator output
    spec_logits = torch.randn(n_predict, batch_size, seq_len - n_predict - 1, vocab_size)
    
    # Reference implementation loss calculation
    def reference_loss(preds, labels):
        losses = []
        loss_fn = nn.CrossEntropyLoss()
        
        for i in range(preds.size(0)):
            targ = labels[:, i + 2 : preds.size(2) + i + 2]  # b n
            loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
            losses.append(loss)
        
        loss = sum(losses)
        return loss
    
    ref_loss = reference_loss(spec_logits, labels)
    print(f"Reference loss: {ref_loss.item()}")
    
    # Our implementation loss calculation
    def our_loss(preds, input_ids, loss_mask=None):
        n_predict = preds.shape[0]
        batch_size, seq_len = input_ids.shape
        
        # For non-remove_padding mode
        spec_loss_total = 0.0
        for i in range(n_predict):
            logits_i = preds[i]  # (batch, n, vocab_size)
            # n = seq_len - n_predict - 1 (since hidden_states was sliced)
            start = i + 2
            end = start + logits_i.shape[1]
            targets = input_ids[:, start:end]  # (batch, n)
            
            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(
