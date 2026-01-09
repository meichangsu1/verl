#!/usr/bin/env python3
"""
Test script for the save_checkpoint method in FSDPEngineWithLMHeadAndSpeculator.
"""

import os
import tempfile
import torch
import json
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_speculator_checkpoint_save():
    print("Testing speculator checkpoint saving...")
    
