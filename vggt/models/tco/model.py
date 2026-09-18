"""DepthAnything3 inheritance layer with scene-local TCO LoRA adaptation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Sequence

import torch
from addict import Dict

from vggt.models.depthanything3 import DepthAnything3
from vggt.utils.cuda_memory_profiler import mark_cuda_memory

from .backbone import TCOBackboneCache, TCOCheckpointBackboneMixin
from .config import TCOConfig
from .constraints import camera_pose_energy, depth_energy, intrinsics_energy
from .lora import TransformerLoRA
from .output import GaussianOutputAdapter


class TCODepthAnything3(TCOCheckpointBackboneMixin, DepthAnything3):
    """VoxelTTO model with TCO adapters on the shared ViT backbone.

    Adapters are injected lazily so an ordinary DA3 checkpoint can be loaded
    before module names change. Only LoRA A/B tensors are optimized per scene.
    """

    def __init__(
        self,
        *args,
        tco_steps: int = 20,
        tco_lr: float = 5e-4,
        tco_lora_rank: int = 4,
        tco_lora_alpha: float = 16.0,
        tco_lora_dropout: float = 0.0,
        tco_lambda_pose: float = 1.0,
        tco_pose_translation_weight: float = 1.0,
        tco_lambda_intrinsics: float = 0.01,
        tco_lambda_depth: float = 1.0,
        tco_grad_clip: float = 1.0,
        tco_reset_each_scene: bool = True,
        tco_gradient_checkpointing: bool = True,
        tco_target_modules: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tco_config = TCOConfig(
            steps=tco_steps,
            lr=tco_lr,
            rank=tco_lora_rank,
            alpha=tco_lora_alpha,
            dropout=tco_lora_dropout,
            lambda_pose=tco_lambda_pose,
            pose_translation_weight=tco_pose_translation_weight,
            lambda_intrinsics=tco_lambda_intrinsics,
            lambda_depth=tco_lambda_depth,
            grad_clip=tco_grad_clip,
            reset_each_scene=tco_reset_each_scene,
            gradient_checkpointing=tco_gradient_checkpointing,
        )
        self.tco_lora = TransformerLoRA(tco_target_modules or TransformerLoRA().target_suffixes)
        self.tco_output = GaussianOutputAdapter()

    def forward_tco_base(self, images, *, forward_dict=None, **kwargs):
        """Call the inherited model without re-entering the TCO wrapper."""
        return super().forward(images, forward_dict=forward_dict, **kwargs)

    def get_tco_backbone(self):
        """Return the shared transformer, isolated for future model ports."""
        backbone = getattr(self.model, "backbone", None)
        transformer = getattr(backbone, "pretrained", None)
        if transformer is None or not hasattr(transformer, "blocks"):
            raise AttributeError("Expected DA3 transformer at model.backbone.pretrained")
        start = max(int(getattr(transformer, "alt_start", 0)), 0)
        if start >= len(transformer.blocks):
            raise ValueError(f"Invalid shared-transformer start {start} for {len(transformer.blocks)} blocks")
        return torch.nn.ModuleDict(
            {str(index): transformer.blocks[index] for index in range(start, len(transformer.blocks))}
        )

    def setup_tco_lora(self) -> list[str]:
        if not self.tco_lora.injected:
            return self.tco_lora.inject(
                self.get_tco_backbone(),
                rank=self.tco_config.rank,
                alpha=self.tco_config.alpha,
                dropout=self.tco_config.dropout,
            )
        if self.tco_config.reset_each_scene:
            self.tco_lora.reset_parameters()
        return [item.name for item in self.tco_lora.injected]


    def _compute_tco_loss(
        self,
        predictions: dict[str, Any],
        *,
        image_hw: tuple[int, int],
        prior_extrinsics: Optional[torch.Tensor],
        prior_intrinsics: Optional[torch.Tensor],
        prior_depths: Optional[torch.Tensor],
        prior_depth_masks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "extrinsics" in predictions:
            reference = predictions["extrinsics"]
        elif "intrinsics" in predictions:
            reference = predictions["intrinsics"]
        else:
            reference = predictions["depth"]
        zero = reference.sum() * 0.0
        losses = {"pose_rotation": zero, "pose_translation": zero, "intrinsics": zero, "depth": zero}

        total = zero
        if prior_extrinsics is not None and self.tco_config.lambda_pose > 0:
            rot, trans = camera_pose_energy(predictions["extrinsics"], prior_extrinsics)
            losses["pose_rotation"], losses["pose_translation"] = rot, trans
            total = total + self.tco_config.lambda_pose * (
                rot + self.tco_config.pose_translation_weight * trans
            )
        if prior_intrinsics is not None and self.tco_config.lambda_intrinsics > 0:
            losses["intrinsics"] = intrinsics_energy(
                predictions["intrinsics"], prior_intrinsics, image_hw
            )
            total = total + self.tco_config.lambda_intrinsics * losses["intrinsics"]
        if prior_depths is not None and self.tco_config.lambda_depth > 0:
            losses["depth"] = depth_energy(predictions["depth"], prior_depths, prior_depth_masks)
            total = total + self.tco_config.lambda_depth * losses["depth"]
        losses["total"] = total
        return total, losses

    def _run_tco(
        self,
        images: torch.Tensor,
        *,
        forward_dict: Optional[dict],
        prior_extrinsics: Optional[torch.Tensor],
        prior_intrinsics: Optional[torch.Tensor],
        prior_depths: Optional[torch.Tensor],
        prior_depth_masks: Optional[torch.Tensor],
        model_kwargs: dict[str, Any],
    ) -> list[dict[str, float]]:
        if prior_extrinsics is None and prior_intrinsics is None and prior_depths is None:
            raise ValueError("TCO is enabled but no camera or depth prior was provided")

        injected_names = self.setup_tco_lora()
        mark_cuda_memory("tco.lora_injected", lora_layers=len(injected_names))
        lora_parameters = list(self.tco_lora.parameters())
        previous_grad_flags = [(parameter, parameter.requires_grad) for parameter in self.parameters()]
        for parameter, _ in previous_grad_flags:
            parameter.requires_grad_(False)
        self.tco_lora.set_requires_grad(True)

        optimizer = torch.optim.Adam(lora_parameters, lr=self.tco_config.lr)
        history: list[dict[str, float]] = []
        try:
            with torch.enable_grad():
                cache = self.prepare_tco_backbone_cache(
                    images,
                    extrinsics=model_kwargs.get("extrinsics"),
                    intrinsics=model_kwargs.get("intrinsics"),
                    normalize_images=model_kwargs.get("normalize_images", True),
                    ref_view_strategy=model_kwargs.get(
                        "ref_view_strategy", "saddle_balanced"
                    ),
                )
                mark_cuda_memory("tco.backbone_cache.end")
                for step in range(self.tco_config.steps):
                    optimizer.zero_grad(set_to_none=True)
                    mark_cuda_memory(f"tco.step_{step}.begin", step=step)
                    try:
                        predictions = self._forward_tco_optimization(
                            cache,
                            require_depth=prior_depths is not None,
                            attn_mask=model_kwargs.get("attn_mask"),
                        )
                    except torch.OutOfMemoryError:
                        mark_cuda_memory(f"tco.step_{step}.forward_oom", step=step)
                        raise
                    mark_cuda_memory(f"tco.step_{step}.forward_end", step=step)
                    loss, losses = self._compute_tco_loss(
                        predictions,
                        image_hw=tuple(images.shape[-2:]),
                        prior_extrinsics=prior_extrinsics,
                        prior_intrinsics=prior_intrinsics,
                        prior_depths=prior_depths,
                        prior_depth_masks=prior_depth_masks,
                    )
                    mark_cuda_memory(f"tco.step_{step}.loss_end", step=step)
                    if not loss.requires_grad:
                        raise RuntimeError("TCO loss has no gradient path to the LoRA adapters")
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite TCO loss at step {step}: {loss.item()}")
                    try:
                        loss.backward()
                    except torch.OutOfMemoryError:
                        mark_cuda_memory(f"tco.step_{step}.backward_oom", step=step)
                        raise
                    mark_cuda_memory(f"tco.step_{step}.backward_end", step=step)
                    torch.nn.utils.clip_grad_norm_(lora_parameters, self.tco_config.grad_clip)
                    optimizer.step()
                    mark_cuda_memory(f"tco.step_{step}.optimizer_end", step=step)
                    history.append({name: float(value.detach().float().cpu()) for name, value in losses.items()})
                    print(
                        f"TCO {step + 1:03d}/{self.tco_config.steps:03d} "
                        f"loss={history[-1]['total']:.6f}"
                    )
                    del predictions, loss, losses
        finally:
            if "cache" in locals():
                del cache
            for parameter, enabled in previous_grad_flags:
                parameter.requires_grad_(enabled)
            self.tco_lora.set_requires_grad(False)
            for parameter in lora_parameters:
                parameter.grad = None

        print(f"TCO adapted {len(injected_names)} transformer Linear layers")
        return history

    def forward(
        self,
        images: torch.Tensor,
        forward_dict: Optional[dict] = None,
        *,
        tco_enabled: bool = True,
        tco_extrinsics: Optional[torch.Tensor] = None,
        tco_intrinsics: Optional[torch.Tensor] = None,
        tco_depths: Optional[torch.Tensor] = None,
        tco_depth_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not tco_enabled or self.tco_config.steps <= 0:
            return super().forward(images, forward_dict=forward_dict, **kwargs)

        # If explicit TCO priors are omitted, camera-conditioning inputs are a
        # convenient fallback. They remain available to the inherited model.
        prior_extrinsics = tco_extrinsics if tco_extrinsics is not None else kwargs.get("extrinsics")
        prior_intrinsics = tco_intrinsics if tco_intrinsics is not None else kwargs.get("intrinsics")
        history = self._run_tco(
            images,
            forward_dict=forward_dict,
            prior_extrinsics=prior_extrinsics,
            prior_intrinsics=prior_intrinsics,
            prior_depths=tco_depths,
            prior_depth_masks=tco_depth_masks,
            model_kwargs=dict(kwargs),
        )

        if images.is_cuda:
            torch.cuda.empty_cache()
        mark_cuda_memory("tco.final_full_forward.begin")
        with torch.no_grad():
            predictions = self.tco_output.final_forward(
                self,
                images,
                forward_dict=forward_dict,
                **kwargs,
            )
        mark_cuda_memory("tco.final_full_forward.end")
        predictions["tco"] = {
            "config": asdict(self.tco_config),
            "history": history,
            "lora_state_dict": self.tco_lora.state_dict(),
        }
        return predictions

    def _forward_tco_optimization(
        self,
        cache: TCOBackboneCache,
        *,
        require_depth: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """TCO-only head path; voxel, Gaussian, and unused DPT work stay out."""
        feats = self.forward_tco_cached_backbone(
            cache,
            require_dense_features=require_depth,
            attn_mask=attn_mask,
        )

        with torch.enable_grad():
            output = Dict()
            if require_depth:
                output = self.model._process_depth_head(feats, cache.height, cache.width)
            output = self.model._process_camera_estimation(
                feats, cache.height, cache.width, output
            )

        predictions = {
            "extrinsics": output.extrinsics[..., :3, :],
            "intrinsics": output.intrinsics,
        }
        if require_depth:
            depth = output.depth
            predictions["depth"] = depth.unsqueeze(-1) if depth.ndim == 4 else depth
            if output.get("depth_conf", None) is not None:
                predictions["depth_conf"] = output.depth_conf
        return predictions
