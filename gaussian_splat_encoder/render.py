"""Order-independent ALPHA-COMPOSITING renderer for 2D gaussian splats.

Replaces the baseline encoder's normalized weighted-sum (`weights /= weights.sum()`), which (1) cancels
opacity out of the result so it can't gate, and (2) forces dense uniform tiling because every pixel is a
convex blend of its nearest gaussians (no "empty"/background state, no reward for growing a splat). Those
two properties are exactly why baseline gaussians are uniform small circles.

This renderer, grounded in GaussianImage (2403.08551) + the order-independent transmittance trick:

    αₙ(x) = oₙ · exp(−½ (x−μₙ)ᵀ Σₙ⁻¹ (x−μₙ))     # oₙ = SEPARATE opacity in (0,1), the gate
    T(x)  = Πₙ (1 − αₙ(x))                          # background transmittance — a PRODUCT, order-free
    C(x)  = (1 − T(x)) · ( Σₙ αₙ cₙ / Σₙ αₙ )  +  T(x) · c_bg

Why this fixes efficiency:
  • A single opaque splat (oₙ→1) at a pixel drives αₙ→1 ⇒ T→0, fg→cₙ: it OWNS a flat region — one big
    splat replaces a tile of small ones.
  • Opacity genuinely gates: oₙ→0 ⇒ αₙ→0 ⇒ no contribution to T or fg ⇒ the splat is OFF. This gives a
    real gradient for an opacity-L1 sparsity penalty to push against (impossible under normalization).
  • Learned background c_bg shows through wherever coverage is low (T→1) ⇒ uncovered/flat regions cost
    ZERO splats. Splats are freed to migrate to detail.
  • Π(1−αₙ) and Σαc are both order-independent ⇒ no depth sort needed (mandatory for a feed-forward set).

Covariance stays in the baseline's scale[2]+theta form (Σ = R S² Rᵀ, PSD by construction) so this drops
into the existing param layout; only the COMPOSITING changes.
"""

from __future__ import annotations

import torch

from .params import activate_scale


def gaussian_falloff(diff: torch.Tensor, scale: torch.Tensor, theta) -> torch.Tensor:
    """exp(−½ dᵀ Σ⁻¹ d) per (pixel, gaussian). diff [B,P,N,2] (pixel−mu); scale [B,N,2] (positive);
    theta [B,N] or None. Returns [B,P,N] in (0,1]."""
    if theta is not None:
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)          # [B,N]
        dy, dx = diff[..., 0], diff[..., 1]                        # [B,P,N]
        rot_y = cos_t.unsqueeze(1) * dy + sin_t.unsqueeze(1) * dx
        rot_x = -sin_t.unsqueeze(1) * dy + cos_t.unsqueeze(1) * dx
        diff = torch.stack([rot_y, rot_x], dim=-1)
    maha = (diff.pow(2) / scale.unsqueeze(1).pow(2)).sum(-1)        # [B,P,N]
    return torch.exp(-0.5 * maha)


def alpha_composite(falloff: torch.Tensor, opacity: torch.Tensor, color: torch.Tensor,
                    bg: torch.Tensor) -> torch.Tensor:
    """Order-independent alpha blend. falloff [B,P,N], opacity [B,N] in (0,1), color [B,N,C], bg [B,C].
    Returns [B,P,C]. T computed in log-space for stability over large N."""
    alpha = (falloff * opacity.unsqueeze(1)).clamp(max=1.0 - 1e-4)  # [B,P,N]
    # T = Π(1−α): the reduction sums up to N≈32k tiny log terms -> do it in fp32 even under bf16 autocast
    # (bf16 accumulation here loses too much precision). exp back to alpha's dtype.
    T = torch.log1p(-alpha.float()).sum(-1).exp().to(alpha.dtype)  # [B,P]  order-free
    asum = alpha.sum(-1).clamp_min(1e-8)                           # [B,P]
    fg = torch.bmm(alpha, color) / asum.unsqueeze(-1)              # [B,P,C] opacity-weighted avg color
    C = (1.0 - T).unsqueeze(-1) * fg + T.unsqueeze(-1) * bg.unsqueeze(1)
    return C


def _pixels(H, W, device, dtype):
    ys = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device, dtype=dtype)
    xs = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gy, gx], dim=-1).reshape(H * W, 2)          # [H*W, 2]


def render_alpha_subsampled(mu, log_scale, color, opacity, bg, H, W, theta=None,
                            n_pixels=10000, pixel_indices=None):
    """Differentiable alpha-composited render at a subsample of pixels (training). opacity & bg REQUIRED
    (opacity is the gate now). Returns (rendered [B,C,n_px], px_idx [B,n_px] flat indices)."""
    B, N, C = color.shape
    device, dtype = mu.device, mu.dtype
    scale = activate_scale(log_scale)
    total = H * W
    if pixel_indices is not None:
        px_idx = pixel_indices
    else:
        n_px = min(n_pixels, total)
        px_idx = torch.randint(0, total, (B, n_px), device=device)
    py = ((px_idx // W).to(dtype) + 0.5) / H
    px = ((px_idx % W).to(dtype) + 0.5) / W
    pixels = torch.stack([py, px], dim=-1)                          # [B,n_px,2]
    diff = pixels.unsqueeze(2) - mu.unsqueeze(1)                    # [B,n_px,N,2]
    falloff = gaussian_falloff(diff, scale, theta)
    rendered = alpha_composite(falloff, opacity, color, bg)         # [B,n_px,C]
    return rendered.permute(0, 2, 1), px_idx


def render_alpha_full(mu, log_scale, color, opacity, bg, H, W, theta=None, chunk_pixels=4096):
    """Full-res alpha-composited render (eval / perceptual loss). chunk_pixels>0 limits VRAM. Returns
    [B,C,H,W]."""
    B, N, C = color.shape
    device, dtype = mu.device, mu.dtype
    scale = activate_scale(log_scale)
    pixels = _pixels(H, W, device, dtype).unsqueeze(0).expand(B, -1, -1)   # [B,H*W,2]
    total = H * W
    if chunk_pixels and total > chunk_pixels:
        out = []
        for s in range(0, total, chunk_pixels):
            e = min(s + chunk_pixels, total)
            diff = pixels[:, s:e].unsqueeze(2) - mu.unsqueeze(1)
            falloff = gaussian_falloff(diff, scale, theta)
            out.append(alpha_composite(falloff, opacity, color, bg))
        rendered = torch.cat(out, dim=1)
    else:
        diff = pixels.unsqueeze(2) - mu.unsqueeze(1)
        falloff = gaussian_falloff(diff, scale, theta)
        rendered = alpha_composite(falloff, opacity, color, bg)
    return rendered.reshape(B, H, W, C).permute(0, 3, 1, 2)
