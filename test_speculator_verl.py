#!/usr/bin/env python3
"""
Test script to verify speculator integration in VERL.
"""

import torch
import torch.nn as nn
from dataclasses import make_dataclass

def test_speculator_creation():
    """Test that speculator can be created and works."""
    print("Testing speculator creation...")
    
    # Import the speculator module
    from verl.models.transformers.speculator import create_speculator_from_config
    
    # Create config
    speculator_config = {
        'n_predict': 3,
        'input_hidden_dim': 768,
        'inner_dim': "768.768",
        'emb_dim': "768.768",
        'proj_dim': "768.768",
        'vocab_size': 32000,
        'scale_input': False,
        'tie_weights': True,
        'tie_lstm_embs': True,
        'method': 'sum_rnn'
    }
    
    # Create speculator
    speculator = create_speculator_from_config(speculator_config)
    print(f"✓ Speculator created: {speculator}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_dim = 768
    
    state = torch.randn(batch_size, seq_len, hidden_dim)
    inds = torch.randint(0, 32000, (batch_size, seq_len + speculator.n_predict))
    
    with torch.no_grad():
        logits = speculator(state, inds)
    
    print(f"✓ Forward pass successful")
    print(f"  Input state shape: {state.shape}")
    print(f"  Input inds shape: {inds.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected shape: ({speculator.n_predict}, {batch_size}, {seq_len}, 32000)")
    
    assert logits.shape == (speculator.n_predict, batch_size, seq_len, 32000)
    return True

def test_engine_registration():
    """Test that the speculator engine is registered."""
    print("\nTesting engine registration...")
    
    from verl.workers.engine.fsdp.transformer_impl import EngineRegistry
    
    # Check if speculator engine is registered
    registry = EngineRegistry._registry
    print(f"Engine registry keys: {list(registry.keys())}")
    
    # Look for speculator engine
    speculator_found = False
    for key in registry.keys():
        if 'speculator' in str(key).lower():
            speculator_found = True
            print(f"✓ Found speculator engine: {key}")
            break
    
    if not speculator_found:
        print("✗ Speculator engine not found in registry")
        return False
        
    return True

def test_loss_function():
    """Test that loss function handles speculator outputs."""
    print("\nTesting loss function...")
    
    from verl.workers.utils.losses import sft_loss
    
    # Create a mock config
    class MockConfig:
        speculator_loss_coeff = 0.5
    
    config = MockConfig()
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    vocab_size = 32000
    n_predict = 3
    
    # Test with speculator outputs
    log_prob = torch.randn(batch_size, seq_len)
    spec_logits = torch.randn(n_predict, batch_size, seq_len - n_predict - 1, vocab_size)
    
    # Create model output
    model_output = {
        "log_probs": log_prob,
        "spec_logits": spec_logits
    }
    
    # Create data
    import torch.nested as nested
    from tensordict import TensorDict
    
    # Create nested tensors for remove_padding mode
    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int64)
    input_ids_values = torch.randint(0, vocab_size, (10,))
    input_ids = nested.nested_tensor_from_jagged(input_ids_values, cu_seqlens)
    
    loss_mask_values = torch.ones(10, dtype=torch.float32)
    loss_mask = nested.nested_tensor_from_jagged(loss_mask_values, cu_seqlens)
    
    data = TensorDict({
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "dp_size": 1,
        "batch_num_tokens": 10.0,
    }, batch_size=[])
    
    # Compute loss
    loss, metrics = sft_loss(config, model_output, data)
    print(f"✓ Loss with speculator computed: {loss.item()}")
    
    # Test without speculator
    model_output_no_spec = {
        "log_probs": log_prob
    }
    
    loss_no_spec, metrics_no_spec = sft_loss(config, model_output_no_spec, data)
    print(f"✓ Loss without speculator computed: {loss_no_spec.item()}")
    
    print("✓ Loss function correctly handles speculator outputs")
    return True

def main():
    print("=" * 60)
    print("Testing Speculator Integration in VERL")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_speculator_creation():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Speculator creation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_engine_registration():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Engine registration test failed: {e}")
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
        print("✓ All tests passed! Speculator integration is working.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
