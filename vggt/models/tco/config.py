from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TCOConfig:
    """Per-scene LoRA test-time optimization configuration."""

    steps: int = 20
    lr: float = 5e-4
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.0
    lambda_pose: float = 1.0
    pose_translation_weight: float = 1.0
    lambda_intrinsics: float = 0.01
    lambda_depth: float = 1.0
    grad_clip: float = 1.0
    reset_each_scene: bool = True
    gradient_checkpointing: bool = True

