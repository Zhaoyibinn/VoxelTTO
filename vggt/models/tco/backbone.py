"""Memory-efficient DA3 backbone execution used only by TCO.

This module leaves the original DA3/DINO implementation untouched. The TCO
model inherits this mixin and opts into prefix caching, per-block gradient
checkpointing during scene optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from vggt.models.depth_anything_3.model.reference_view_selector import (
    reorder_by_reference,
    restore_original_order,
    select_reference_view,
)
from vggt.models.depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION


@dataclass
class TCOBackboneCache:
    """Detached state immediately before the first LoRA-enabled block."""

    tokens: torch.Tensor
    local_tokens: torch.Tensor
    pos: Optional[torch.Tensor]
    pos_nodiff: Optional[torch.Tensor]
    camera_token: Optional[torch.Tensor]
    reference_indices: Optional[torch.Tensor]
    batch_size: int
    view_count: int
    height: int
    width: int


class TCOCheckpointBackboneMixin:
    """Run the inherited DA3 transformer without modifying its source code."""

    def _tco_transformer(self):
        backbone = getattr(self.model, "backbone", None)
        transformer = getattr(backbone, "pretrained", None)
        if transformer is None or not hasattr(transformer, "blocks"):
            raise AttributeError("Expected DA3 transformer at model.backbone.pretrained")
        return backbone, transformer

    @staticmethod
    def _run_attention(
        transformer,
        tokens: torch.Tensor,
        block,
        attention_type: str,
        *,
        pos: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        use_checkpoint: bool,
    ) -> torch.Tensor:
        batch_size, view_count = tokens.shape[:2]
        input_dtype = tokens.dtype
        if attention_type == "local":
            tokens = rearrange(tokens, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attention_type == "global":
            tokens = rearrange(tokens, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attention_type}")

        if use_checkpoint and torch.is_grad_enabled():
            def block_forward(input_tokens: torch.Tensor) -> torch.Tensor:
                return block(input_tokens, pos=pos, attn_mask=attn_mask)

            # The cached prefix input is detached. Non-reentrant checkpointing
            # still discovers trainable LoRA parameters inside the block.
            tokens = checkpoint(block_forward, tokens, use_reentrant=False)
        else:
            tokens = block(tokens, pos=pos, attn_mask=attn_mask)

        tokens = tokens.to(dtype=input_dtype)
        if attention_type == "local":
            return rearrange(tokens, "(b s) n c -> b s n c", b=batch_size, s=view_count)
        return rearrange(tokens, "b (s n) c -> b s n c", b=batch_size, s=view_count)

    def prepare_tco_backbone_cache(
        self,
        images: torch.Tensor,
        *,
        extrinsics: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        normalize_images: bool = True,
        ref_view_strategy: str = "saddle_balanced",
    ) -> TCOBackboneCache:
        """Compute the frozen image encoder/prefix once for a TCO scene."""
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = self._maybe_normalize_images(images, normalize_images=normalize_images)
        batch_size, view_count, _, height, width = images.shape
        _, transformer = self._tco_transformer()
        start = max(int(transformer.alt_start), 0)
        if start <= 0 or start >= len(transformer.blocks):
            raise ValueError(
                f"TCO prefix caching requires 0 < alt_start < block count, got "
                f"{start} and {len(transformer.blocks)}"
            )

        with torch.no_grad():
            camera_token = None
            if extrinsics is not None:
                if intrinsics is None:
                    raise ValueError("intrinsics are required when extrinsics are supplied")
                # CameraEnc is computed once and remains constant during TCO.
                camera_token = self.model.cam_enc(
                    extrinsics.float(), intrinsics.float(), (height, width)
                )

            tokens = transformer.prepare_tokens_with_masks(images)
            pos, pos_nodiff = transformer._prepare_rope(
                batch_size, view_count, height, width, tokens.device
            )
            local_tokens = tokens
            reference_indices = None

            for block_index in range(start):
                block = transformer.blocks[block_index]
                local_pos = (
                    None
                    if block_index < transformer.rope_start or transformer.rope is None
                    else pos
                )
                if (
                    block_index == start - 1
                    and tokens.shape[1] >= THRESH_FOR_REF_SELECTION
                    and camera_token is None
                ):
                    reference_indices = select_reference_view(
                        tokens, strategy=ref_view_strategy
                    )
                    tokens = reorder_by_reference(tokens, reference_indices)
                    local_tokens = reorder_by_reference(local_tokens, reference_indices)

                tokens = self._run_attention(
                    transformer,
                    tokens,
                    block,
                    "local",
                    pos=local_pos,
                    attn_mask=None,
                    use_checkpoint=False,
                )
                local_tokens = tokens

        return TCOBackboneCache(
            tokens=tokens.detach(),
            local_tokens=local_tokens.detach(),
            pos=None if pos is None else pos.detach(),
            pos_nodiff=None if pos_nodiff is None else pos_nodiff.detach(),
            camera_token=None if camera_token is None else camera_token.detach(),
            reference_indices=reference_indices,
            batch_size=batch_size,
            view_count=view_count,
            height=height,
            width=width,
        )

    def forward_tco_cached_backbone(
        self,
        cache: TCOBackboneCache,
        *,
        require_dense_features: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """Run the LoRA-enabled suffix with block-wise checkpointing."""
        backbone, transformer = self._tco_transformer()
        start = int(transformer.alt_start)
        configured_layers: Sequence[int] = tuple(backbone.out_layers)
        if any(layer < start for layer in configured_layers):
            raise ValueError(
                "TCO cached backbone requires output layers at or after "
                f"alt_start={start}, got {list(configured_layers)}"
            )
        output_layers = (
            set(configured_layers) if require_dense_features else {max(configured_layers)}
        )

        tokens = cache.tokens
        local_tokens = cache.local_tokens
        if cache.camera_token is not None:
            camera_token = cache.camera_token
        else:
            ref_token = transformer.camera_token[:, :1].expand(
                cache.batch_size, -1, -1
            )
            src_token = transformer.camera_token[:, 1:].expand(
                cache.batch_size, cache.view_count - 1, -1
            )
            camera_token = torch.cat([ref_token, src_token], dim=1)

        camera_token = camera_token.to(cache.tokens)
        # Avoid mutating the cached prefix tensor in place.
        local_tokens = torch.cat([camera_token.unsqueeze(2), local_tokens[:, :, 1:]], dim=2)
        tokens = torch.cat([camera_token.unsqueeze(2), tokens[:, :, 1:]], dim=2)
        outputs = []
        with torch.enable_grad():
            for block_index in range(start, len(transformer.blocks)):
                block = transformer.blocks[block_index]
                if block_index < transformer.rope_start or transformer.rope is None:
                    global_pos, local_pos = None, None
                else:
                    global_pos, local_pos = cache.pos_nodiff, cache.pos

                if block_index % 2 == 1:
                    tokens = self._run_attention(
                        transformer,
                        tokens,
                        block,
                        "global",
                        pos=global_pos,
                        attn_mask=attn_mask,
                        use_checkpoint=self.tco_config.gradient_checkpointing,
                    )
                else:
                    tokens = self._run_attention(
                        transformer,
                        tokens,
                        block,
                        "local",
                        pos=local_pos,
                        attn_mask=None,
                        use_checkpoint=self.tco_config.gradient_checkpointing,
                    )
                    local_tokens = tokens

                if block_index in output_layers:
                    if not require_dense_features:
                        camera_output = (
                            torch.cat([local_tokens[:, :, 0], tokens[:, :, 0]], dim=-1)
                            if transformer.cat_token
                            else tokens[:, :, 0]
                        )
                        if cache.reference_indices is not None:
                            camera_output = restore_original_order(
                                camera_output, cache.reference_indices
                            )
                        outputs.append((camera_output, None))
                        continue

                    output_tokens = (
                        torch.cat([local_tokens, tokens], dim=-1)
                        if transformer.cat_token
                        else tokens
                    )
                    if cache.reference_indices is not None:
                        output_tokens = restore_original_order(
                            output_tokens, cache.reference_indices
                        )
                    outputs.append((output_tokens[:, :, 0], output_tokens))

            if not outputs:
                raise RuntimeError("No configured DA3 output layer was produced")
            if not require_dense_features:
                camera_tokens = outputs[-1][0]
                empty_features = tokens.new_empty(
                    (*camera_tokens.shape[:2], 0, transformer.embed_dim)
                )
                return ((empty_features, camera_tokens),)
            camera_tokens = [output[0] for output in outputs]
            if outputs[0][1].shape[-1] == transformer.embed_dim:
                feature_tokens = [transformer.norm(output[1]) for output in outputs]
            elif outputs[0][1].shape[-1] == transformer.embed_dim * 2:
                feature_tokens = [
                    torch.cat(
                        [
                            output[1][..., : transformer.embed_dim],
                            transformer.norm(output[1][..., transformer.embed_dim :]),
                        ],
                        dim=-1,
                    )
                    for output in outputs
                ]
            else:
                raise ValueError(f"Invalid transformer output shape: {outputs[0][1].shape}")

            patch_start = 1 + transformer.num_register_tokens
            feature_tokens = [output[..., patch_start:, :] for output in feature_tokens]
            return tuple(zip(feature_tokens, camera_tokens))
