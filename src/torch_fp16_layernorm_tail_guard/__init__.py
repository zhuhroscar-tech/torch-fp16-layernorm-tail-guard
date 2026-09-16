"""torch-fp16-layernorm-tail-guard: guard a real torch CPU float16
nn.LayerNorm bug where exact-constant rows whose length is not a
multiple of 8 get nonzero garbage in the tail instead of exact 0.
"""
from .core import (
    TorchUnavailableError,
    diagnose,
    safe_layer_norm,
)

__all__ = [
    "TorchUnavailableError",
    "diagnose",
    "safe_layer_norm",
]

__version__ = "0.1.0"
