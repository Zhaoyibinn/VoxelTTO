"""LoRA injection utilities used by TCO.

The adapter is intentionally independent from DepthAnything3 and the Gaussian
heads so it can be reused with another transformer backbone later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.base_layer.requires_grad_(False)
        weight = base_layer.weight
        # Keep the trainable adapter and its optimizer state in FP32 even when
        # the frozen transformer runs in BF16/FP16. Small TCO updates would
        # otherwise be quantized directly into low-precision LoRA parameters.
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                device=weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty(
                base_layer.out_features,
                self.rank,
                device=weight.device,
                dtype=torch.float32,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # The first adapted forward is exactly the pretrained model.
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        lora_inputs = self.dropout(inputs.to(dtype=self.lora_A.dtype))
        adapted = F.linear(F.linear(lora_inputs, self.lora_A), self.lora_B)
        # Do not promote the surrounding transformer to FP32: only the LoRA
        # branch and its gradients stay FP32.
        return base + (adapted * self.scaling).to(dtype=base.dtype)


@dataclass(frozen=True)
class InjectedLayer:
    name: str
    parent: nn.Module
    attribute: str
    adapter: LoRALinear


class TransformerLoRA:
    """Inject and manage LoRA layers below a transformer module."""

    def __init__(
        self,
        target_suffixes: Sequence[str] = (
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
            "mlp.w1",
            "mlp.w2",
            "mlp.w3",
            "mlp.w12",
        ),
    ) -> None:
        self.target_suffixes = tuple(target_suffixes)
        self.injected: list[InjectedLayer] = []

    def _matches(self, name: str) -> bool:
        return any(name == suffix or name.endswith(f".{suffix}") for suffix in self.target_suffixes)

    def inject(
        self,
        module: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> list[str]:
        if self.injected:
            raise RuntimeError("LoRA has already been injected")

        # Snapshot first; replacing children while walking named_modules is unsafe.
        candidates: list[tuple[str, nn.Module, str, nn.Linear]] = []
        for parent_name, parent in module.named_modules():
            for attribute, child in parent.named_children():
                full_name = f"{parent_name}.{attribute}" if parent_name else attribute
                if isinstance(child, nn.Linear) and self._matches(full_name):
                    candidates.append((full_name, parent, attribute, child))

        if not candidates:
            raise ValueError(
                "No transformer Linear modules matched LoRA targets: "
                f"{list(self.target_suffixes)}"
            )

        for name, parent, attribute, linear in candidates:
            adapter = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attribute, adapter)
            self.injected.append(InjectedLayer(name, parent, attribute, adapter))
        return [item.name for item in self.injected]

    def parameters(self) -> Iterator[nn.Parameter]:
        for item in self.injected:
            yield item.adapter.lora_A
            yield item.adapter.lora_B

    def reset_parameters(self) -> None:
        for item in self.injected:
            item.adapter.reset_parameters()

    def set_requires_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(enabled)

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for item in self.injected:
            state[f"{item.name}.lora_A"] = item.adapter.lora_A.detach().clone()
            state[f"{item.name}.lora_B"] = item.adapter.lora_B.detach().clone()
        return state

    def remove(self, *, merge: bool = False) -> None:
        for item in self.injected:
            base = item.adapter.base_layer
            if merge:
                with torch.no_grad():
                    delta = item.adapter.lora_B @ item.adapter.lora_A
                    base.weight.add_(delta.to(base.weight.dtype), alpha=item.adapter.scaling)
            setattr(item.parent, item.attribute, base)
        self.injected.clear()
