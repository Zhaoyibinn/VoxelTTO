"""Output-branch boundary for TCO.

The LoRA optimizer only needs camera/depth predictions. Gaussian construction
is deliberately deferred to the final pass so backbone adaptation and a future
Gaussian head can be migrated independently.
"""

from __future__ import annotations


class GaussianOutputAdapter:
    def optimization_forward(self, model, images, *, forward_dict=None, **kwargs):
        optimization_kwargs = dict(kwargs)
        optimization_kwargs.update(
            infer_gs=False,
            render_gs=False,
            skip_backend=True,
        )
        return model.forward_tco_base(
            images,
            forward_dict=forward_dict,
            **optimization_kwargs,
        )

    def final_forward(self, model, images, *, forward_dict=None, **kwargs):
        final_kwargs = dict(kwargs)
        final_kwargs.update(
            infer_gs=bool(model.infer_gs),
            skip_backend=False,
        )
        return model.forward_tco_base(images, forward_dict=forward_dict, **final_kwargs)

