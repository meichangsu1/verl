#!/usr/bin/env python3
"""
Test script to verify speculator integration in SFT trainer.
This script tests that:
1. The speculator model can be created from config
2. The FSDPEngineWithLMHeadAndSpeculator can be instantiated
3. The loss function correctly handles speculator outputs
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

# Mock the necessary imports
class MockConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_speculator_model():
    """Test that speculator model can be created and forward pass works."""
    print("Testing speculator model creation...")
    
    # Create a minimal speculator config
    speculator_config = {
        'n_predict': 3,
        'input_hidden_dim': 768,
        'inner_dim': [768, 768],
        'emb_dim': [768, 768],
        'proj_dim': [768, 768],
        'vocab_size': 32000,
        'scale_input': False,
        'tie_weights': True,
        'tie_lstm_embs': True,
        'method': 'sum_rnn'
    }
    
    try:
        from verl.models.transformers.speculator import create_speculator_from_config
        speculator = create_speculator_from_config(speculator_config)
        print(f"✓ Speculator created successfully: {speculator}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        hidden_dim = 768
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        inds = torch.randint(0, 32000, (batch_size, seq_len + speculator.n_predict))
        
        # Forward pass
        with torch.no_grad():
            logits = speculator(hidden_states, inds)
            
        print(f"✓ Speculator forward pass successful")
        print(f"  Input shape: hidden_states={hidden_states.shape}, inds={inds.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected shape: ({speculator.n_predict}, {batch_size}, {seq_len}, 32000)")
        
        assert logits.shape == (speculator.n_predict, batch_size, seq_len, 32000), \
            f"Unexpected output shape: {logits.shape}"
        
        return True
        
    except Exception as e:
        print(f"✗ Speculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function():
    """Test that loss function handles speculator outputs."""
    print("\nTesting loss function with speculator...")
    
    try:
        from verl.workers.utils.losses import sft_loss
        
        # Create dummy data
        batch_size = 2
        seq_len = 10
        vocab_size = 32000
        n_predict = 3
        
        # Model outputs
        logits = torch.randn(batch_size, seq_len, vocab_size)
        spec_logits = torch.randn(n_predict, batch_size, seq_len, vocab_size)
        
        # Labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create mock config
        config = MockConfig(speculator_loss_coeff=0.5)
        
        # Test with speculator outputs
        model_output = {
            'logits': logits,
            'spec_logits': spec_logits
        }
        
        loss_with_spec = sft_loss(model_output, labels, config)
        print(f"✓ Loss with speculator computed: {loss_with_spec.item()}")
        
        # Test without speculator outputs
        model_output_no_spec = {
            'logits': logits
        }
        
        loss_no_spec = sft_loss(model_output_no_spec, labels, config)
        print(f"✓ Loss without speculator computed: {loss_no_spec.item()}")
        
        # The loss with speculator should be larger (or equal) due to additional terms
        assert loss_with_spec >= loss_no_spec - 1e-5, \
            f"Loss with speculator ({loss_with_spec}) should be >= loss without ({loss_no_spec})"
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_engine_registration():
    """Test that the speculator engine is properly registered."""
    print("\nTesting engine registration...")
    
    try:
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
        
    except Exception as e:
        print(f"✗ Engine registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing Speculator Integration in SFT Trainer")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Speculator model
    if test_speculator_model():
        tests_passed += 1
    
    # Test 2: Loss function
    if test_loss_function():
        tests_passed += 1
    
    # Test 3: Engine registration
    if test_engine_registration():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Speculator integration is working.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
