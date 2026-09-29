"""Gaussian-parameter assembly — the canonical encoder-output -> token-vector mapping.

Lives here (deps: torch only) rather than in data/images.py (which pulls in torchvision/webdataset) so
the lightweight consumers — the diversity objective, eval scripts — can import it without the data
stack. data/images.py re-exports it for backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# The floor width of the smallest splat, in normalised units. Part of the tok9 contract, NOT a
# numerical guard: it is what the renderers add, so anything reading the scale channel must add it.
SCALE_EPS = 1e-4


def activate_scale(log_scale: torch.Tensor) -> torch.Tensor:
    """tok9's RAW ``log_scale`` channel -> the sigma the renderers actually draw with.

    ``assemble_gaussian_params`` below applies no transform, so the token layout carries log_scale
    raw and every renderer entry point activates it itself. That transform used to be written out
    at a dozen call sites; it lives here now so a consumer OUTSIDE a renderer -- a descriptor, a
    statistic, a diagnostic -- cannot silently disagree with what was drawn. Dropping the epsilon
    is the failure that hides: it only moves the smallest splats, which is exactly where the
    high-frequency detail this project keeps chasing lives.
    """
    return F.softplus(log_scale) + SCALE_EPS


def assemble_gaussian_params(enc_out: dict) -> torch.Tensor:
    """Encoder output dict -> the 9-dim param vector [mu(2), log_scale(2), theta(1), color(3),
    opacity(1)] that GaussianJEPA consumes. Returns [B, N, 9]. coords = enc_out['mu']."""
    mu = enc_out["mu"]                                   # [B,N,2]
    ls = enc_out["log_scale"]                            # [B,N,2]
    th = enc_out["theta"]
    th = th.unsqueeze(-1) if th is not None else torch.zeros_like(mu[..., :1])
    col = enc_out["color"]                               # [B,N,3]
    op = enc_out["opacity"]
    op = op.unsqueeze(-1) if op is not None else torch.ones_like(mu[..., :1])
    return torch.cat([mu, ls, th, col, op], dim=-1)      # [B,N,9]
