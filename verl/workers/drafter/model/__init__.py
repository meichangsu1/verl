"""
Modeling layer for Eagle-style drafters.
"""

# Import specific implementations for direct access
from .eagle.llama_eagle3 import LlamaForCausalLMEagle3
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel

# __all__ defines what is exported when someone does 'from modeling import *'
__all__ = [
    "LlamaForCausalLMEagle3",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
