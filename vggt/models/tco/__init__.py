from .config import TCOConfig
from .lora import LoRALinear, TransformerLoRA
from .output import GaussianOutputAdapter


def __getattr__(name):
    if name == "TCODepthAnything3":
        from .model import TCODepthAnything3
        return TCODepthAnything3
    raise AttributeError(name)

__all__ = [
    "GaussianOutputAdapter",
    "LoRALinear",
    "TCOConfig",
    "TCODepthAnything3",
    "TransformerLoRA",
]
