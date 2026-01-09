# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPSpeculatorConfig:
    """Configuration for MLP-based speculator."""
    name_or_path: str
    hidden_size: int
    speculator_width: int
    vocab_size: int
    n_speculator_heads: int
    tie_weights: bool = True
    scale_input: bool = False
    activation: str = "gelu"


class MLPSpeculator(nn.Module):
    """MLP-based speculator for speculative decoding."""
    
    def __init__(self, config: MLPSpeculatorConfig):
        super().__init__()
        self.config = config
        
        self.n_predict = config.n_speculator_heads
        self.hidden_size = config.hidden_size
        self.speculator_width = config.speculator_width
        self.vocab_size = config.vocab_size
        self.tie_weights = config.tie_weights
        self.scale_input = config.scale_input
        
        # Activation function
        if config.activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif config.activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        
        # Input scaling layer
        if self.scale_input:
            self.ln_input = nn.LayerNorm(self.hidden_size)
        
        # Projection layers for each head
        self.projections = nn.ModuleList()
        for i in range(self.n_predict):
            if not self.tie_weights or i == 0:
                proj = nn.Sequential(
                    nn.Linear(self.hidden_size, self.speculator_width, bias=False),
                    self.activation,
                    nn.LayerNorm(self.speculator_width),
                    nn.Linear(self.speculator_width, self.speculator_width, bias=False),
                    self.activation,
                    nn.LayerNorm(self.speculator_width),
                )
            else:
                # Share weights with first head
