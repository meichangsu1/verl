#!/usr/bin/env python3
"""
Test script for speculator SFT training integration
"""

import os
import tempfile
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Mock the necessary components for testing
class MockTokenizer:
    def __init__(self):
        self.vocab_size = 32000
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.randint(0, 100, (2, 128))}

class MockDataset:
    def __init__(self):
        self.length = 100
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 100, (128,)),
            "attention_mask": torch.ones(128),
            "labels": torch.randint(0, 100, (128,))
        }

def test_speculator_creation():
    """Test that speculator is created correctly"""
    print("Testing speculator creation...")
    
    # Create a minimal config
    config_dict = {
        "model": {
            "partial_pretrain": "test_model",
            "trust_remote_code": False,
            "fsdp_config": {
                "model_dtype": "fp32",
                "wrap_policy": "transformer_auto_wrap_policy",
                "cpu_offload": False,
                "offload_params": False,
            },
            "strategy": "fsdp",
            "enable_gradient_checkpointing": False,
            "speculator": {
                "n_predict": 3,
                "inner_dim": "768",
                "emb_dim": "768",
                "proj_dim": "768",
                "scale_input": False,
                "tie_weights": False,
                "tie_lstm_embs": False,
                "method": "sum_rnn",
            },
            "freeze_base_model": True,
            "speculator_loss_coeff": 1.0,
            "lora_rank": 0,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_adapter_path": None,
            "use_liger": False,
            "external_lib": None,
        },
        "data": {
            "train_batch_size": 2,
            "micro_batch_size_per_gpu": 1,
            "max_length": 128,
            "balance_dp_token": False,
            "chat_template": None,
            "train_files": ["test.parquet"],
            "val_files": ["test.parquet"],
            "custom_cls": {"path": None, "name": None},
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-4,
            "lr_warmup_steps_ratio": 0.1,
            "lr_scheduler": "cosine",
            "clip_grad": 1.0,
        },
        "trainer": {
            "device": "cpu",
            "default_local_dir": tempfile.mkdtemp(),
            "total_epochs": 1,
            "test_freq": 10,
            "save_freq": 10,
            "project_name": "test",
            "experiment_name": "test",
            "logger": "wandb",
        },
        "ulysses_sequence_parallel_size": 1,
        "use_remove_padding": False,
    }
    
    config = DictConfig(config_dict)
    
    # Mock device mesh
    class MockDeviceMesh:
        def __init__(self):
            self._rank = 0
            self._size = 1
            
        def get_rank(self):
            return self._rank
            
        def size(self, dim=0):
            return self._size
            
        def get_local_rank(self, dim):
            return 0
            
    device_mesh = MockDeviceMesh()
    ulysses_device_mesh = MockDeviceMesh()
    
    # Mock tokenizer and dataset
    tokenizer = MockTokenizer()
    train_dataset = MockDataset()
    val_dataset = MockDataset()
    
    # Import the trainer
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
        
        # Create trainer
        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        
        # Check if speculator was created
        assert hasattr(trainer, 'has_speculator'), "Trainer should have has_speculator attribute"
        assert trainer.has_speculator, "Trainer should have speculator enabled"
        assert hasattr(trainer, 'speculator'), "Trainer should have speculator attribute"
        assert trainer.speculator is not None, "Speculator should not be None"
        
        print("✓ Speculator creation test passed!")
        
        # Test speculator forward pass
        batch_size = 2
        seq_len = 128
        hidden_dim = 768
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        # Test speculator forward
        with torch.no_grad():
            output = trainer.speculator(hidden_states, input_ids)
            assert output.shape[0] == config.model.speculator.n_predict, \
                f"Output should have {config.model.speculator.n_predict} prediction heads"
            print("✓ Speculator forward pass test passed!")
            
        # Test loss computation
        dummy_batch = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "loss_mask": torch.ones(batch_size, seq_len),
        }
        
        # Test loss computation without backward
        loss = trainer._compute_loss_and_backward(dummy_batch, do_backward=False)
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        print("✓ Loss computation test passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_save_load():
    """Test checkpoint save and load functionality"""
    print("\nTesting checkpoint save and load...")
    
    try:
        import tempfile
        import shutil
        
        # Create a temporary directory for checkpoints
        checkpoint_dir = tempfile.mkdtemp()
        
        # Create a simple speculator
        from verl.models.transformers.speculator import create_speculator_from_config
        
        speculator_config = {
            'n_predict': 3,
            'input_hidden_dim': 768,
            'inner_dim': '768',
            'emb_dim': '768',
            'proj_dim': '768',
            'vocab_size': 32000,
            'scale_input': False,
            'tie_weights': False,
            'tie_lstm_embs': False,
            'method': 'sum_rnn',
        }
        
        speculator = create_speculator_from_config(speculator_config)
        
        # Save checkpoint
        speculator_dir = os.path.join(checkpoint_dir, "speculator")
        os.makedirs(speculator_dir, exist_ok=True)
        
        state_dict_path = os.path.join(speculator_dir, "pytorch_model.bin")
        torch.save(speculator.state_dict(), state_dict_path)
        
        # Load checkpoint
        loaded_state_dict = torch.load(state_dict_path, map_location="cpu")
        
        # Create new speculator and load state
        new_speculator = create_speculator_from_config(speculator_config)
        new_speculator.load_state_dict(loaded_state_dict)
        
        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            speculator.named_parameters(), new_speculator.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"Parameter {name1} doesn't match"
        
        print("✓ Checkpoint save/load test passed!")
        
        # Cleanup
        shutil.rmtree(checkpoint_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running speculator SFT integration tests...")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_speculator_creation()
    test2_passed = test_checkpoint_save_load()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Speculator creation test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Checkpoint save/load test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
