"""Tile-fused Triton renderer for the order-independent 2D-gaussian alpha compositing in render.py.

The reference renderer materializes a [B, P, N] falloff tensor (P pixels x N gaussians), which is the
memory wall that forces n_px subsampling during fitting/training (a 32k-gaussian full-res 768px render
would need 590k*32k floats). This kernel instead accumulates, per pixel, three running reductions over
gaussians and never stores the [P,N] tile -> peak memory is O(P*C), so full-res differentiable rendering
becomes possible. That kills the pixel-sampling under-determination directly (see the ceil768 result:
the fitter ceiling was flat in count because it was starved of pixel supervision, not splats).

Math reproduced EXACTLY from render.py (validated numerically in _selftest_*):
    a_n   = clamp(opacity_n * exp(-1/2 * maha_n),  max=1-1e-4)     # maha in the rotated/scaled frame
    T     = exp( sum_n log(1 - a_n) )                              # order-free transmittance, fp32 sum
    fg[c] = ( sum_n a_n * color_n[c] ) / max(sum_n a_n, 1e-8)
    C[c]  = (1 - T) * fg[c] + T * bg[c]

scale = softplus(log_scale)+1e-4 is applied in torch (the wrapper) so its gradient chains automatically;
the kernel takes the activated scale. bg's gradient (sum_p gC*T) is a pure pixel reduction -> done in
torch, kept out of the kernels. Everything internal is fp32 (more accurate than the bf16 reference path).

Two validated paths (both fp32-internal, [B,C,H,W] out, drop-in for render_alpha_full):
  cull=False (exact): every pixel sums every gaussian. Matches the reference to fp32 round-off
    (fwd 6e-5, grads rel 1e-4). O(P) memory: measured 22GB->25MB @256, 49GB->32MB @384 (900-1539x; the
    reference OOMs the 3070 at these sizes -- that is what forced n_px subsampling). 1.47s @512 N=16384.
  cull=True (tile-binned, default TILE=8): each 8x8 pixel tile sums only the gaussians whose
    cutoff*max-axis bbox overlaps it. Params are PRE-PERMUTED into tile-sorted contiguous buffers
    (coalesced loads); backward writes PER-PAIR grads to contiguous buffers (no atomics) then torch
    index_add scatters them to gaussians. Truncates the >cutoff-sigma tail (approximation: cull-fwd
    converges to exact <1.4e-4 @ 3-sigma; cull-bwd grads match exact to rel 1e-6). MEASURED 53x faster
    than exact on realistic trained splats: 0.03s vs 1.47s @512 N=16384. THIS is the training path.

Lessons that cost time (kept here so they aren't relearned): (1) TILE=16 spills registers on the 256-px
tile (3D [256,64,4] temporaries) -> 50x slower than TILE=8's 64-px tile; TILE=8 is the sweet spot.
(2) Culling only pays when splats are LOCAL; with huge splats (~40% of image) every gaussian hits ~270
tiles and culling degenerates to exact-plus-overhead. Trained splats (~5px / 0.01 normalized) hit a few.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from .params import activate_scale
from .render import _pixels, render_alpha_full


class PairBudgetExceeded(RuntimeError):
    """Raised by _build_tiles when a render would allocate more (tile,gaussian) pairs than max_pairs.
    A SPECIFIC type so the trainer can catch exactly this (skip the offending image) without swallowing
    unrelated RuntimeErrors -- the alternative (a bare except) would hide real failures."""


# ----------------------------------------------------------------------------- forward kernel
@triton.jit
def _render_fwd_kernel(
    muy_ptr, mux_ptr, sy_ptr, sx_ptr, th_ptr, op_ptr, col_ptr, bg_ptr,
    py_ptr, px_ptr,
    out_ptr, T_ptr, asum_ptr, fg_ptr,
    N, P, C,
    C_POW: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, HAS_THETA: tl.constexpr,
):
    b = tl.program_id(0)
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)      # [BLOCK_P]
    mask_p = offs_p < P
    py = tl.load(py_ptr + offs_p, mask=mask_p, other=0.0)            # pixels shared across batch
    px = tl.load(px_ptr + offs_p, mask=mask_p, other=0.0)

    offs_c = tl.arange(0, C_POW)                                    # [C_POW], padded to pow2
    mask_c = offs_c < C
    S_log = tl.zeros([BLOCK_P], tl.float32)
    S_a = tl.zeros([BLOCK_P], tl.float32)
    S_ac = tl.zeros([BLOCK_P, C_POW], tl.float32)                   # opacity-weighted color sum

    nbase = b * N
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)                         # [BLOCK_N]
        mask_n = offs_n < N
        gy = tl.load(muy_ptr + nbase + offs_n, mask=mask_n, other=0.0)
        gx = tl.load(mux_ptr + nbase + offs_n, mask=mask_n, other=0.0)
        gsy = tl.load(sy_ptr + nbase + offs_n, mask=mask_n, other=1.0)
        gsx = tl.load(sx_ptr + nbase + offs_n, mask=mask_n, other=1.0)
        go = tl.load(op_ptr + nbase + offs_n, mask=mask_n, other=0.0)
        # color block [BLOCK_N, C_POW] (padded channels masked to 0)
        col = tl.load(col_ptr + (nbase + offs_n)[:, None] * C + offs_c[None, :],
                      mask=mask_n[:, None] & mask_c[None, :], other=0.0)

        dy = py[:, None] - gy[None, :]                              # [BLOCK_P, BLOCK_N]
        dx = px[:, None] - gx[None, :]
        if HAS_THETA:
            gth = tl.load(th_ptr + nbase + offs_n, mask=mask_n, other=0.0)
            cth = tl.cos(gth)[None, :]
            sth = tl.sin(gth)[None, :]
            ry = cth * dy + sth * dx
            rx = -sth * dy + cth * dx
        else:
            ry = dy
            rx = dx
        inv_sy = 1.0 / gsy[None, :]
        inv_sx = 1.0 / gsx[None, :]
        maha = (ry * ry) * (inv_sy * inv_sy) + (rx * rx) * (inv_sx * inv_sx)
        a = tl.exp(-0.5 * maha) * go[None, :]
        a = tl.minimum(a, 1.0 - 1e-4)
        a = tl.where(mask_n[None, :], a, 0.0)                       # off gaussians contribute nothing

        S_log += tl.sum(tl.log(1.0 - a), axis=1)                    # log1p(-a); 1-a >= 1e-4 > 0
        S_a += tl.sum(a, axis=1)
        # contrib[p,c] = sum_n a[p,n] * col[n,c]  via 3D broadcast-reduce over the gaussian axis
        S_ac += tl.sum(a[:, :, None] * col[None, :, :], axis=1)     # [BLOCK_P, C_POW]

    T = tl.exp(S_log)
    asum = tl.maximum(S_a, 1e-8)
    pbase = b * P
    tl.store(T_ptr + pbase + offs_p, T, mask=mask_p)
    tl.store(asum_ptr + pbase + offs_p, asum, mask=mask_p)
    fg = S_ac / asum[:, None]                                       # [BLOCK_P, C_POW]
    bgv = tl.load(bg_ptr + b * C + offs_c, mask=mask_c, other=0.0)  # [C_POW]
    Cout = (1.0 - T)[:, None] * fg + T[:, None] * bgv[None, :]      # [BLOCK_P, C_POW]
    pc = (pbase + offs_p)[:, None] * C + offs_c[None, :]
    store_mask = mask_p[:, None] & mask_c[None, :]
    tl.store(out_ptr + pc, Cout, mask=store_mask)
    tl.store(fg_ptr + pc, fg, mask=store_mask)


def _forward(mu, scale, color, opacity, bg, py, px, theta, BLOCK_P=64, BLOCK_N=64):
    """Run the fused forward. Inputs already activated (scale = softplus(.)+eps). Returns
    out [B,P,C] and the saved-for-backward (T [B,P], asum [B,P], fg [B,P,C])."""
    B, N, C = color.shape
    P = py.shape[0]
    dev = mu.device
    muy = mu[..., 0].contiguous().float()
    mux = mu[..., 1].contiguous().float()
    sy = scale[..., 0].contiguous().float()
    sx = scale[..., 1].contiguous().float()
    op = opacity.contiguous().float()
    col = color.contiguous().float()
    bgf = bg.contiguous().float()
    has_th = theta is not None
    th = theta.contiguous().float() if has_th else torch.empty(1, device=dev)

    out = torch.empty((B, P, C), device=dev, dtype=torch.float32)
    T = torch.empty((B, P), device=dev, dtype=torch.float32)
    asum = torch.empty((B, P), device=dev, dtype=torch.float32)
    fg = torch.empty((B, P, C), device=dev, dtype=torch.float32)

    grid = (B, triton.cdiv(P, BLOCK_P))
    _render_fwd_kernel[grid](
        muy, mux, sy, sx, th, op, col, bgf,
        py.contiguous().float(), px.contiguous().float(),
        out, T, asum, fg,
        N, P, C, C_POW=triton.next_power_of_2(C),
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, HAS_THETA=has_th,
    )
    return out, T, asum, fg


# ----------------------------------------------------------------------------- backward kernel
@triton.jit
def _render_bwd_kernel(
    muy_ptr, mux_ptr, sy_ptr, sx_ptr, th_ptr, op_ptr, col_ptr, bg_ptr,
    py_ptr, px_ptr,
    T_ptr, asum_ptr, fg_ptr, gout_ptr,
    gmuy_ptr, gmux_ptr, gsy_ptr, gsx_ptr, gth_ptr, gop_ptr, gcol_ptr,
    N, P, C,
    C_POW: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, HAS_THETA: tl.constexpr,
):
    """One program owns a block of gaussians; it loops over ALL pixels accumulating that block's grads.
    Because each gaussian is owned by exactly one program, the per-gaussian grads are plain stores (no
    atomics). bg's grad is a pure pixel reduction and is computed in torch, not here."""
    b = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)      # [BLOCK_N]
    mask_n = offs_n < N
    offs_c = tl.arange(0, C_POW)
    mask_c = offs_c < C
    nbase = b * N

    gy = tl.load(muy_ptr + nbase + offs_n, mask=mask_n, other=0.0)
    gx = tl.load(mux_ptr + nbase + offs_n, mask=mask_n, other=0.0)
    gsy = tl.load(sy_ptr + nbase + offs_n, mask=mask_n, other=1.0)
    gsx = tl.load(sx_ptr + nbase + offs_n, mask=mask_n, other=1.0)
    go = tl.load(op_ptr + nbase + offs_n, mask=mask_n, other=0.0)
    col = tl.load(col_ptr + (nbase + offs_n)[:, None] * C + offs_c[None, :],
                  mask=mask_n[:, None] & mask_c[None, :], other=0.0)       # [BLOCK_N, C_POW]
    if HAS_THETA:
        gth = tl.load(th_ptr + nbase + offs_n, mask=mask_n, other=0.0)
        cth = gth.cos()[None, :]
        sth = gth.sin()[None, :]

    acc_muy = tl.zeros([BLOCK_N], tl.float32)
    acc_mux = tl.zeros([BLOCK_N], tl.float32)
    acc_sy = tl.zeros([BLOCK_N], tl.float32)
    acc_sx = tl.zeros([BLOCK_N], tl.float32)
    acc_op = tl.zeros([BLOCK_N], tl.float32)
    acc_th = tl.zeros([BLOCK_N], tl.float32)
    acc_col = tl.zeros([BLOCK_N, C_POW], tl.float32)

    pbase = b * P
    bgv = tl.load(bg_ptr + b * C + offs_c, mask=mask_c, other=0.0)         # [C_POW]
    inv_sy = 1.0 / gsy[None, :]
    inv_sx = 1.0 / gsx[None, :]
    for p0 in range(0, P, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        mask_p = offs_p < P
        py = tl.load(py_ptr + offs_p, mask=mask_p, other=0.0)
        px = tl.load(px_ptr + offs_p, mask=mask_p, other=0.0)
        T = tl.load(T_ptr + pbase + offs_p, mask=mask_p, other=1.0)        # [BLOCK_P]
        asum = tl.load(asum_ptr + pbase + offs_p, mask=mask_p, other=1.0)
        pc = (pbase + offs_p)[:, None] * C + offs_c[None, :]
        mpc = mask_p[:, None] & mask_c[None, :]
        fg = tl.load(fg_ptr + pc, mask=mpc, other=0.0)                     # [BLOCK_P, C_POW]
        gout = tl.load(gout_ptr + pc, mask=mpc, other=0.0)                 # upstream dL/dC

        # per-pixel reductions onto the running sums (S_log, S_a, S_ac)
        dfg = gout * (1.0 - T)[:, None]                                    # [BLOCK_P, C_POW]
        dTp = tl.sum(gout * (bgv[None, :] - fg), axis=1)                   # [BLOCK_P]
        dS_log = dTp * T
        dS_a = -tl.sum(dfg * fg, axis=1) / asum
        dS_ac = dfg / asum[:, None]                                        # [BLOCK_P, C_POW]

        # recompute geometry [BLOCK_P, BLOCK_N]
        dy = py[:, None] - gy[None, :]
        dx = px[:, None] - gx[None, :]
        if HAS_THETA:
            ry = cth * dy + sth * dx
            rx = -sth * dy + cth * dx
        else:
            ry = dy
            rx = dx
        maha = (ry * ry) * (inv_sy * inv_sy) + (rx * rx) * (inv_sx * inv_sx)
        f = tl.exp(-0.5 * maha)
        fo = f * go[None, :]
        a = tl.minimum(fo, 1.0 - 1e-4)
        a = tl.where(mask_n[None, :], a, 0.0)
        unclamped = fo < (1.0 - 1e-4)

        col_term = tl.sum(dS_ac[:, None, :] * col[None, :, :], axis=2)     # [BLOCK_P, BLOCK_N]
        da = dS_log[:, None] * (-1.0 / (1.0 - a)) + dS_a[:, None] + col_term
        acc_col += tl.sum(a[:, :, None] * dS_ac[:, None, :], axis=0)       # [BLOCK_N, C_POW]

        da_fo = tl.where(unclamped, da, 0.0)                              # through the alpha clamp
        acc_op += tl.sum(da_fo * f, axis=0)
        gm = (da_fo * go[None, :]) * (-0.5 * f)                            # dL/dmaha
        acc_sy += tl.sum(gm * (-2.0 * ry * ry) * (inv_sy * inv_sy * inv_sy), axis=0)
        acc_sx += tl.sum(gm * (-2.0 * rx * rx) * (inv_sx * inv_sx * inv_sx), axis=0)
        gry = gm * (2.0 * ry) * (inv_sy * inv_sy)                          # dL/drot_y
        grx = gm * (2.0 * rx) * (inv_sx * inv_sx)
        if HAS_THETA:
            acc_th += tl.sum(gry * rx - grx * ry, axis=0)
            acc_muy += tl.sum(-(gry * cth - grx * sth), axis=0)
            acc_mux += tl.sum(-(gry * sth + grx * cth), axis=0)
        else:
            acc_muy += tl.sum(-gry, axis=0)
            acc_mux += tl.sum(-grx, axis=0)

    tl.store(gmuy_ptr + nbase + offs_n, acc_muy, mask=mask_n)
    tl.store(gmux_ptr + nbase + offs_n, acc_mux, mask=mask_n)
    tl.store(gsy_ptr + nbase + offs_n, acc_sy, mask=mask_n)
    tl.store(gsx_ptr + nbase + offs_n, acc_sx, mask=mask_n)
    tl.store(gop_ptr + nbase + offs_n, acc_op, mask=mask_n)
    if HAS_THETA:
        tl.store(gth_ptr + nbase + offs_n, acc_th, mask=mask_n)
    tl.store(gcol_ptr + (nbase + offs_n)[:, None] * C + offs_c[None, :], acc_col,
             mask=mask_n[:, None] & mask_c[None, :])


def _backward(grad_out, mu, scale, color, opacity, bg, py, px, theta, T, asum, fg,
              BLOCK_P=64, BLOCK_N=64):
    """grad_out [B,P,C]. Returns grads for mu, scale, color, opacity, bg, theta (theta=None if unused)."""
    B, N, C = color.shape
    P = py.shape[0]
    dev = mu.device
    has_th = theta is not None
    muy = mu[..., 0].contiguous().float(); mux = mu[..., 1].contiguous().float()
    sy = scale[..., 0].contiguous().float(); sx = scale[..., 1].contiguous().float()
    op = opacity.contiguous().float(); col = color.contiguous().float(); bgf = bg.contiguous().float()
    th = theta.contiguous().float() if has_th else torch.empty(1, device=dev)
    go = grad_out.contiguous().float()

    g_muy = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_mux = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_sy = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_sx = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_op = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_th = torch.zeros((B, N), device=dev, dtype=torch.float32)
    g_col = torch.zeros((B, N, C), device=dev, dtype=torch.float32)

    grid = (B, triton.cdiv(N, BLOCK_N))
    _render_bwd_kernel[grid](
        muy, mux, sy, sx, th, op, col, bgf,
        py.contiguous().float(), px.contiguous().float(),
        T.contiguous(), asum.contiguous(), fg.contiguous(), go,
        g_muy, g_mux, g_sy, g_sx, g_th, g_op, g_col,
        N, P, C, C_POW=triton.next_power_of_2(C),
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, HAS_THETA=has_th,
    )
    g_mu = torch.stack([g_muy, g_mux], dim=-1)                             # [B,N,2]
    g_scale = torch.stack([g_sy, g_sx], dim=-1)                            # [B,N,2]
    g_bg = (go * T.unsqueeze(-1)).sum(dim=1)                               # [B,C] pure pixel reduction
    g_theta = g_th if has_th else None
    return g_mu, g_scale, g_col, g_op, g_bg, g_theta


# ----------------------------------------------------------------------------- autograd Function + API
class _AlphaRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, color, opacity, bg, theta, py, px):
        out, T, asum, fg = _forward(mu, scale, color, opacity, bg, py, px, theta)
        ctx.has_theta = theta is not None
        ctx.save_for_backward(mu, scale, color, opacity, bg,
                              theta if theta is not None else mu.new_zeros(1),
                              py, px, T, asum, fg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, scale, color, opacity, bg, theta, py, px, T, asum, fg = ctx.saved_tensors
        th = theta if ctx.has_theta else None
        g_mu, g_scale, g_col, g_op, g_bg, g_th = _backward(
            grad_out, mu, scale, color, opacity, bg, py, px, th, T, asum, fg)
        return g_mu, g_scale, g_col, g_op, g_bg, g_th, None, None


def render_alpha_full_triton(mu, log_scale, color, opacity, bg, H, W, theta=None,
                             cull=False, TILE=8, cutoff=3.0, odds=False):
    """Drop-in for render.render_alpha_full using the fused Triton kernels (O(P) memory). Returns
    [B,C,H,W]. softplus(log_scale)+1e-4 is applied here so the kernel sees activated scale and the
    log_scale gradient chains through autograd.

    cull=False : exact (every pixel sums every gaussian) — matches the reference to fp32 round-off.
                 O(P) memory (25MB), 1.47s fwd+bwd @ 512x512 N=16384.
    cull=True  : tile-binned approximation (truncates >cutoff-sigma tail). 53x FASTER than exact on
                 realistic splats (0.03s vs 1.47s), grads match exact to rel 1e-6. The training path."""
    B, N, C = color.shape
    scale = activate_scale(log_scale)
    if cull:
        out = _AlphaRenderCull.apply(mu, scale, color, opacity, bg, theta, H, W, TILE, cutoff, odds)
    else:
        if odds:
            raise NotImplementedError("odds compositing is only wired through the cull/octave path; "
                                      "use cull=True or render_alpha_octave")
        pyx = _pixels(H, W, mu.device, torch.float32)
        out = _AlphaRender.apply(mu, scale, color, opacity, bg, theta,
                                 pyx[:, 0].contiguous(), pyx[:, 1].contiguous())   # [B,P,C]
    return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------- tile culling
def _build_tiles(mu, scale, H, W, TILE, cutoff, max_pairs=None):
    """Per batch, bin gaussians into the TILE x TILE pixel tiles their cutoff*max-axis bbox overlaps,
    then sort by tile -> CSR (gid_sorted, tile_starts[nt+1]). Conservative circle bound (>= the ellipse),
    so no in-cutoff contribution is dropped; only the >cutoff-sigma tail is truncated.

    All pair-length temporaries are int32 (gaussian/tile indices are < N, nt << 2^31 under any sane budget;
    halves the binning footprint vs the old int64). `max_pairs` is a HARD safety guard: if a batch's pair
    count exceeds it we RAISE rather than allocate -- on the GB10 Spark an O(N*nt) allocation hard-crashes
    the whole OS (large/global splats bin into every tile), so an unconditional bound is a safety
    REQUIREMENT, not an optimization. The multires path keeps coarse (large) splats at low res so their
    pair count stays small; this guard backstops the fine/native pass against any pathological scale drift."""
    B, N, _ = mu.shape
    dev = mu.device
    ntx = (W + TILE - 1) // TILE
    nty = (H + TILE - 1) // TILE
    nt = ntx * nty
    cx = mu[..., 1] * W
    cy = mu[..., 0] * H
    r = cutoff * scale.max(-1).values                                  # [B,N] normalized radius
    rx = r * W
    ry = r * H
    tx0 = ((cx - rx) / TILE).floor().clamp(0, ntx - 1).to(torch.int32)
    tx1 = ((cx + rx) / TILE).floor().clamp(0, ntx - 1).to(torch.int32)
    ty0 = ((cy - ry) / TILE).floor().clamp(0, nty - 1).to(torch.int32)
    ty1 = ((cy + ry) / TILE).floor().clamp(0, nty - 1).to(torch.int32)
    wts = tx1 - tx0 + 1
    counts = (wts * (ty1 - ty0 + 1)).to(torch.int32)                  # [B,N] tiles per gaussian
    if max_pairs is not None:
        worst = int(counts.sum(dim=1).max().item())
        if worst > max_pairs:
            raise PairBudgetExceeded(
                f"_build_tiles: {worst} (tile,gaussian) pairs exceeds the {max_pairs} budget @ {H}x{W} "
                f"TILE={TILE} cutoff={cutoff} N={N}. Large/global splats are binning into ~every tile -- "
                f"route them through the low-res (coarse) pass instead of culling at full res. "
                f"REFUSING to allocate (would risk crashing the box).")
    gid_b, starts_b = [], []
    arangeN = torch.arange(N, device=dev, dtype=torch.int32)
    nt_range = torch.arange(nt + 1, device=dev, dtype=torch.int32)
    for b in range(B):
        c = counts[b]
        tot = int(c.sum().item())                                     # one unavoidable sync (sizes the pair buffers)
        if tot == 0:
            gid_b.append(torch.zeros(1, dtype=torch.int32, device=dev))
            starts_b.append(torch.zeros(nt + 1, dtype=torch.int32, device=dev))
            continue
        gid = torch.repeat_interleave(arangeN, c)                     # [tot] int32
        gl = gid.long()                                               # compute ONCE (was 4x: 4 int32->int64 copies)
        # k = within-gaussian tile rank (0..count-1): global pos minus this gaussian's run start.
        start = (c.to(torch.int64).cumsum(0) - c).to(torch.int32)     # cumsum in int64 (overflow-safe), back to i32
        k = torch.arange(tot, device=dev, dtype=torch.int32) - start[gl]
        w = wts[b][gl]
        tile_x = tx0[b][gl] + (k % w)
        tile_y = ty0[b][gl] + (k // w)
        tid = tile_y * ntx + tile_x                                   # [tot] int32, < nt
        order = tid.argsort()                                         # within-tile order is irrelevant (commutative sum)
        tid_s = tid[order]
        gid_b.append(gid[order].contiguous())
        starts_b.append(torch.searchsorted(tid_s, nt_range).to(torch.int32))
    return gid_b, starts_b, ntx, nt


# --- Launch-config tuning for the cull kernels (sm_86 / RTX 3070, measured @768^2 N=50k, TILE=8) ---
# Each program handles ONE TILE*TILE=64-pixel tile. The forward is light per-pair (3 sums) so it wants
# MANY tiny blocks (num_warps=1 -> 32 lanes for 64 px = 2/lane) to maximize block-level occupancy across
# the 9216 tiles; BLOCK_N=32 keeps the [px,N] register tile small enough to avoid spill. The backward does
# ~10x the per-pair math (all param grads) so it's register/ILP-bound -> num_warps=4 wins, still BLOCK_N=32.
# Measured: fwd 3.25->1.18ms (2.75x), bwd 3.32->1.96ms (1.7x) vs the old num_warps=4/BLOCK_N=64 default.
# NOTE: num_warps=1 with BLOCK_N=64 SPILLS catastrophically (27ms fwd / 34ms bwd) -- keep BLOCK_N<=32 @ nw=1.
_FWD_WARPS, _FWD_STAGES, _FWD_BN = 1, 2, 32   # ns=2 pipelines the inner param loads (~6% over ns=1)
_BWD_WARPS, _BWD_STAGES, _BWD_BN = 4, 1, 32


@triton.jit
def _render_fwd_cull_kernel(
    muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, bg_ptr,
    starts_ptr, out_ptr, T_ptr, asum_ptr, fg_ptr,
    slog_ptr, sa_ptr, sac_ptr,
    b, P, C, ntx, H, W,
    C_POW: tl.constexpr, TILE: tl.constexpr, BLOCK_N: tl.constexpr, HAS_THETA: tl.constexpr,
    FIELDS: tl.constexpr, ODDS: tl.constexpr,
):
    """Coalesced cull forward: gaussian params are PRE-PERMUTED into tile-sorted contiguous buffers
    (*_s), so each tile reads its slice with sequential (coalesced) loads -- no per-gaussian gather.

    FIELDS=False (default): store the composited RGB + (T, asum, fg) -- the original single-pass path.
    FIELDS=True: store the RAW partial sums (S_log, S_a, S_ac) and skip the composite. Alpha compositing
    is a commutative monoid, so a caller can render disjoint gaussian groups at different resolutions, sum
    their partial sums, and composite ONCE -- the basis of the multires (coarse@H/8 + fine@native) render
    that bounds memory regardless of splat scale. S_a is stored UN-clamped so the clamp happens after the
    cross-group sum, not per group.

    ODDS=False: classic convex-blend foreground  fg = Sum(a*c)/Sum(a)  -- a faint splat dilutes a committed
    one (averaging -> the washed-out text blur). ODDS=True (F2): foreground weight is the ODDS w = a/(1-a)
    instead of a, so fg = Sum(w*c)/Sum(w). As a->1, w->inf, so a committed (near-opaque) splat DOMINATES /
    occludes the faint coarse layer under it WITHOUT a depth sort -- still a commutative monoid (three
    additive sums), so the octave combine is unchanged. The composite FORMULA is identical (fg = third/second,
    T=exp(first)); only WHICH weight is summed changes. S_log (coverage/T) uses a in both modes."""
    t = tl.program_id(0)
    tx = t % ntx
    ty = t // ntx
    local = tl.arange(0, TILE * TILE)
    yp = ty * TILE + local // TILE
    xp = tx * TILE + local % TILE
    mask_p = (yp < H) & (xp < W)
    py = (yp.to(tl.float32) + 0.5) / H
    px = (xp.to(tl.float32) + 0.5) / W
    offs_c = tl.arange(0, C_POW)
    mask_c = offs_c < C

    S_log = tl.zeros([TILE * TILE], tl.float32)
    S_a = tl.zeros([TILE * TILE], tl.float32)
    S_ac = tl.zeros([TILE * TILE, C_POW], tl.float32)

    g0 = tl.load(starts_ptr + t).to(tl.int32)
    g1 = tl.load(starts_ptr + t + 1).to(tl.int32)
    for gi in range(g0, g1, BLOCK_N):
        idx = gi + tl.arange(0, BLOCK_N)
        mask_g = idx < g1
        gy = tl.load(muy_s + idx, mask=mask_g, other=0.0)
        gx = tl.load(mux_s + idx, mask=mask_g, other=0.0)
        gsy = tl.load(sy_s + idx, mask=mask_g, other=1.0)
        gsx = tl.load(sx_s + idx, mask=mask_g, other=1.0)
        go = tl.load(op_s + idx, mask=mask_g, other=0.0)
        col = tl.load(col_s + idx[:, None] * C + offs_c[None, :],
                      mask=mask_g[:, None] & mask_c[None, :], other=0.0)
        dy = py[:, None] - gy[None, :]
        dx = px[:, None] - gx[None, :]
        if HAS_THETA:
            gth = tl.load(th_s + idx, mask=mask_g, other=0.0)
            cth = gth.cos()[None, :]
            sth = gth.sin()[None, :]
            ry = cth * dy + sth * dx
            rx = -sth * dy + cth * dx
        else:
            ry = dy
            rx = dx
        inv_sy = 1.0 / gsy[None, :]
        inv_sx = 1.0 / gsx[None, :]
        maha = (ry * ry) * (inv_sy * inv_sy) + (rx * rx) * (inv_sx * inv_sx)
        a = tl.exp(-0.5 * maha) * go[None, :]
        a = tl.minimum(a, 1.0 - 1e-4)
        a = tl.where(mask_g[None, :], a, 0.0)
        S_log += tl.sum(tl.log(1.0 - a), axis=1)
        if ODDS:
            wgt = a / (1.0 - a)                                        # F2 odds weight; a clamped <=1-1e-4
        else:
            wgt = a
        S_a += tl.sum(wgt, axis=1)                                    # S_a := Sum w (odds) or Sum a (alpha)
        S_ac += tl.sum(wgt[:, :, None] * col[None, :, :], axis=1)

    pflat = b * P + yp * W + xp
    pc = pflat[:, None] * C + offs_c[None, :]
    sm = mask_p[:, None] & mask_c[None, :]
    if FIELDS:
        tl.store(slog_ptr + pflat, S_log, mask=mask_p)
        tl.store(sa_ptr + pflat, S_a, mask=mask_p)
        tl.store(sac_ptr + pc, S_ac, mask=sm)
    else:
        T = tl.exp(S_log)
        asum = tl.maximum(S_a, 1e-8)
        tl.store(T_ptr + pflat, T, mask=mask_p)
        tl.store(asum_ptr + pflat, asum, mask=mask_p)
        fg = S_ac / asum[:, None]
        bgv = tl.load(bg_ptr + b * C + offs_c, mask=mask_c, other=0.0)
        Cout = (1.0 - T)[:, None] * fg + T[:, None] * bgv[None, :]
        tl.store(out_ptr + pc, Cout, mask=sm)
        tl.store(fg_ptr + pc, fg, mask=sm)


def _gather_pairs(muy, mux, sy, sx, op, col, th, gid, has_th, dev):
    """Permute per-gaussian params (one batch) into tile-sorted pair order [TOT] / [TOT,C]."""
    g = gid.long()
    sel = lambda a: a.index_select(0, g).contiguous()
    th_s = sel(th) if has_th else torch.empty(1, device=dev)
    return sel(muy), sel(mux), sel(sy), sel(sx), sel(op), sel(col), th_s


def _cull_param_buffers(mu, scale, color, opacity, theta):
    """Shared param prep for the cull fwd/bwd (contiguous fp32 per-component views)."""
    has_th = theta is not None
    return dict(
        muy=mu[..., 0].contiguous().float(), mux=mu[..., 1].contiguous().float(),
        sy=scale[..., 0].contiguous().float(), sx=scale[..., 1].contiguous().float(),
        op=opacity.contiguous().float(), col=color.contiguous().float(),
        th=theta.contiguous().float() if has_th else None, has_th=has_th)


def _forward_cull(mu, scale, color, opacity, bg, theta, H, W, TILE=8, cutoff=3.0, BLOCK_N=64,
                  max_pairs=None, odds=False):
    """Culled fused forward (coalesced). Returns out [B,P,C], (T, asum, fg), tile CSR (for backward)."""
    B, N, C = color.shape
    P = H * W
    dev = mu.device
    pb = _cull_param_buffers(mu, scale, color, opacity, theta)
    muy, mux, sy, sx, op, col = pb["muy"], pb["mux"], pb["sy"], pb["sx"], pb["op"], pb["col"]
    th, has_th = pb["th"], pb["has_th"]
    bgf = bg.contiguous().float()
    gid_b, starts_b, ntx, nt = _build_tiles(mu.detach(), scale.detach(), H, W, TILE, cutoff, max_pairs)

    out = torch.empty((B, P, C), device=dev, dtype=torch.float32)
    T = torch.empty((B, P), device=dev, dtype=torch.float32)
    asum = torch.empty((B, P), device=dev, dtype=torch.float32)
    fg = torch.empty((B, P, C), device=dev, dtype=torch.float32)
    dummy = torch.empty(1, device=dev, dtype=torch.float32)            # unused field ptrs (FIELDS=False)
    C_POW = triton.next_power_of_2(C)
    for b in range(B):
        muy_s, mux_s, sy_s, sx_s, op_s, col_s, th_s = _gather_pairs(
            muy[b], mux[b], sy[b], sx[b], op[b], col[b], th[b] if has_th else None, gid_b[b], has_th, dev)
        _render_fwd_cull_kernel[(nt,)](
            muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, bgf,
            starts_b[b], out, T, asum, fg,
            dummy, dummy, dummy,
            b, P, C, ntx, H, W,
            C_POW=C_POW, TILE=TILE, BLOCK_N=_FWD_BN, HAS_THETA=has_th, FIELDS=False, ODDS=odds,
            num_warps=_FWD_WARPS, num_stages=_FWD_STAGES,
        )
    return out, (T, asum, fg), (gid_b, starts_b, ntx, nt)


def _forward_cull_fields(mu, scale, color, opacity, theta, H, W, TILE=8, cutoff=3.0, BLOCK_N=64,
                         max_pairs=None, odds=False):
    """Culled forward that returns the RAW partial-sum FIELDS (no bg, no composite):
        S_log [B,P], S_a [B,P], S_ac [B,P,C].
    The monoid combine (sum groups, then composite with bg) happens in torch in the multires wrapper."""
    B, N, C = color.shape
    P = H * W
    dev = mu.device
    pb = _cull_param_buffers(mu, scale, color, opacity, theta)
    muy, mux, sy, sx, op, col = pb["muy"], pb["mux"], pb["sy"], pb["sx"], pb["op"], pb["col"]
    th, has_th = pb["th"], pb["has_th"]
    gid_b, starts_b, ntx, nt = _build_tiles(mu.detach(), scale.detach(), H, W, TILE, cutoff, max_pairs)

    slog = torch.zeros((B, P), device=dev, dtype=torch.float32)        # zero-init: empty tiles stay 0
    sa = torch.zeros((B, P), device=dev, dtype=torch.float32)
    sac = torch.zeros((B, P, C), device=dev, dtype=torch.float32)
    dummy = torch.empty(1, device=dev, dtype=torch.float32)            # unused composite ptrs (bg/out/T/fg)
    C_POW = triton.next_power_of_2(C)
    for b in range(B):
        muy_s, mux_s, sy_s, sx_s, op_s, col_s, th_s = _gather_pairs(
            muy[b], mux[b], sy[b], sx[b], op[b], col[b], th[b] if has_th else None, gid_b[b], has_th, dev)
        _render_fwd_cull_kernel[(nt,)](
            muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, dummy,
            starts_b[b], dummy, dummy, dummy, dummy,
            slog, sa, sac,
            b, P, C, ntx, H, W,
            C_POW=C_POW, TILE=TILE, BLOCK_N=_FWD_BN, HAS_THETA=has_th, FIELDS=True, ODDS=odds,
            num_warps=_FWD_WARPS, num_stages=_FWD_STAGES,
        )
    return slog, sa, sac, (gid_b, starts_b, ntx, nt)


@triton.jit
def _render_bwd_cull_kernel(
    muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, bg_ptr,
    starts_ptr, T_ptr, asum_ptr, fg_ptr, gout_ptr,
    dslog_ptr, dsa_ptr, dsac_ptr,
    gmuy_p, gmux_p, gsy_p, gsx_p, gth_p, gop_p, gcol_p,
    b, P, C, ntx, H, W,
    C_POW: tl.constexpr, TILE: tl.constexpr, BLOCK_N: tl.constexpr, HAS_THETA: tl.constexpr,
    FIELDS: tl.constexpr, ODDS: tl.constexpr,
):
    """Coalesced cull backward: params read from tile-sorted buffers; PER-PAIR grads written to
    contiguous [TOT] buffers (each pair is a unique slot -> no atomics). The torch caller then
    index_add's pairs back to gaussians. bg grad is torch-side.

    FIELDS=False: upstream grad is d(image); derive (dS_log, dS_a, dS_ac) from it + (T, asum, fg, bg).
    FIELDS=True: upstream grad is ALREADY (dS_log, dS_a, dS_ac) -- the partial-sum fields' grads handed
    down by the torch monoid-combine. Load them directly; the per-pair grad math below is IDENTICAL.

    ODDS: the composite-grads (dS_log, dS_a, dS_ac) have IDENTICAL form in both modes (S_a just holds Sum w
    instead of Sum a), so the FIELDS=False derivation above is unchanged. ONLY the per-pair assembly differs,
    because the second/third sums are w=a/(1-a) weighted: with u=1/(1-a),
        da   = dS_log*(-u) + (dS_a + col_term) * u^2     [vs alpha's  dS_log*(-u) + dS_a + col_term]
        dcol = dS_ac * (a*u)                              [vs alpha's  dS_ac * a]
    (dw/da = u^2, dlog(1-a)/da = -u). Everything downstream of da -> {op,scale,mu,theta} is identical."""
    t = tl.program_id(0)
    tx = t % ntx
    ty = t // ntx
    local = tl.arange(0, TILE * TILE)
    yp = ty * TILE + local // TILE
    xp = tx * TILE + local % TILE
    mask_p = (yp < H) & (xp < W)
    py = (yp.to(tl.float32) + 0.5) / H
    px = (xp.to(tl.float32) + 0.5) / W
    offs_c = tl.arange(0, C_POW)
    mask_c = offs_c < C
    pflat = b * P + yp * W + xp
    pc = pflat[:, None] * C + offs_c[None, :]
    mpc = mask_p[:, None] & mask_c[None, :]
    if FIELDS:
        dS_log = tl.load(dslog_ptr + pflat, mask=mask_p, other=0.0)
        dS_a = tl.load(dsa_ptr + pflat, mask=mask_p, other=0.0)
        dS_ac = tl.load(dsac_ptr + pc, mask=mpc, other=0.0)
    else:
        T = tl.load(T_ptr + pflat, mask=mask_p, other=1.0)
        asum = tl.load(asum_ptr + pflat, mask=mask_p, other=1.0)
        fg = tl.load(fg_ptr + pc, mask=mpc, other=0.0)
        gout = tl.load(gout_ptr + pc, mask=mpc, other=0.0)
        bgv = tl.load(bg_ptr + b * C + offs_c, mask=mask_c, other=0.0)
        dfg = gout * (1.0 - T)[:, None]
        dTp = tl.sum(gout * (bgv[None, :] - fg), axis=1)
        dS_log = dTp * T
        dS_a = -tl.sum(dfg * fg, axis=1) / asum
        dS_ac = dfg / asum[:, None]

    g0 = tl.load(starts_ptr + t).to(tl.int32)
    g1 = tl.load(starts_ptr + t + 1).to(tl.int32)
    for gi in range(g0, g1, BLOCK_N):
        idx = gi + tl.arange(0, BLOCK_N)
        mask_g = idx < g1
        gy = tl.load(muy_s + idx, mask=mask_g, other=0.0)
        gx = tl.load(mux_s + idx, mask=mask_g, other=0.0)
        gsy = tl.load(sy_s + idx, mask=mask_g, other=1.0)
        gsx = tl.load(sx_s + idx, mask=mask_g, other=1.0)
        go = tl.load(op_s + idx, mask=mask_g, other=0.0)
        col = tl.load(col_s + idx[:, None] * C + offs_c[None, :],
                      mask=mask_g[:, None] & mask_c[None, :], other=0.0)
        dy = py[:, None] - gy[None, :]
        dx = px[:, None] - gx[None, :]
        if HAS_THETA:
            gth = tl.load(th_s + idx, mask=mask_g, other=0.0)
            cth = gth.cos()[None, :]
            sth = gth.sin()[None, :]
            ry = cth * dy + sth * dx
            rx = -sth * dy + cth * dx
        else:
            ry = dy
            rx = dx
        inv_sy = 1.0 / gsy[None, :]
        inv_sx = 1.0 / gsx[None, :]
        maha = (ry * ry) * (inv_sy * inv_sy) + (rx * rx) * (inv_sx * inv_sx)
        f = tl.exp(-0.5 * maha)
        fo = f * go[None, :]
        a = tl.minimum(fo, 1.0 - 1e-4)
        a = tl.where(mask_g[None, :], a, 0.0)
        unclamped = fo < (1.0 - 1e-4)

        inv1ma = 1.0 / (1.0 - a)
        col_term = tl.sum(dS_ac[:, None, :] * col[None, :, :], axis=2)
        if ODDS:
            da = dS_log[:, None] * (-inv1ma) + (dS_a[:, None] + col_term) * (inv1ma * inv1ma)
            cw = a * inv1ma                                               # odds weight a/(1-a) for dcol
        else:
            da = dS_log[:, None] * (-inv1ma) + dS_a[:, None] + col_term
            cw = a
        g_col = tl.sum(cw[:, :, None] * dS_ac[:, None, :], axis=0)         # [BLOCK_N, C_POW]
        da_fo = tl.where(unclamped, da, 0.0)
        g_op = tl.sum(da_fo * f, axis=0)
        gm = (da_fo * go[None, :]) * (-0.5 * f)
        g_sy = tl.sum(gm * (-2.0 * ry * ry) * (inv_sy * inv_sy * inv_sy), axis=0)
        g_sx = tl.sum(gm * (-2.0 * rx * rx) * (inv_sx * inv_sx * inv_sx), axis=0)
        gry = gm * (2.0 * ry) * (inv_sy * inv_sy)
        grx = gm * (2.0 * rx) * (inv_sx * inv_sx)
        if HAS_THETA:
            g_th = tl.sum(gry * rx - grx * ry, axis=0)
            g_muy = tl.sum(-(gry * cth - grx * sth), axis=0)
            g_mux = tl.sum(-(gry * sth + grx * cth), axis=0)
        else:
            g_muy = tl.sum(-gry, axis=0)
            g_mux = tl.sum(-grx, axis=0)

        tl.store(gmuy_p + idx, g_muy, mask=mask_g)
        tl.store(gmux_p + idx, g_mux, mask=mask_g)
        tl.store(gsy_p + idx, g_sy, mask=mask_g)
        tl.store(gsx_p + idx, g_sx, mask=mask_g)
        tl.store(gop_p + idx, g_op, mask=mask_g)
        if HAS_THETA:
            tl.store(gth_p + idx, g_th, mask=mask_g)
        tl.store(gcol_p + idx[:, None] * C + offs_c[None, :], g_col,
                 mask=mask_g[:, None] & mask_c[None, :])


def _backward_cull(grad_out, mu, scale, color, opacity, bg, theta, T, asum, fg, tiles,
                   H, W, TILE=8, BLOCK_N=64, odds=False):
    B, N, C = color.shape
    P = H * W
    dev = mu.device
    gid_b, starts_b, ntx, nt = tiles
    has_th = theta is not None
    muy = mu[..., 0].contiguous().float(); mux = mu[..., 1].contiguous().float()
    sy = scale[..., 0].contiguous().float(); sx = scale[..., 1].contiguous().float()
    op = opacity.contiguous().float(); col = color.contiguous().float(); bgf = bg.contiguous().float()
    th = theta.contiguous().float() if has_th else None
    go = grad_out.contiguous().float()

    z = lambda *s: torch.zeros(s, device=dev, dtype=torch.float32)
    g_muy, g_mux, g_sy, g_sx, g_op, g_th = z(B, N), z(B, N), z(B, N), z(B, N), z(B, N), z(B, N)
    g_col = z(B, N, C)
    dummy = torch.empty(1, device=dev, dtype=torch.float32)            # unused dS ptrs (FIELDS=False)
    C_POW = triton.next_power_of_2(C)
    for b in range(B):
        gid = gid_b[b].long()
        TOT = gid.shape[0]
        muy_s, mux_s, sy_s, sx_s, op_s, col_s, th_s = _gather_pairs(
            muy[b], mux[b], sy[b], sx[b], op[b], col[b], th[b] if has_th else None, gid, has_th, dev)
        p_muy, p_mux, p_sy, p_sx, p_op, p_th = z(TOT), z(TOT), z(TOT), z(TOT), z(TOT), z(TOT)
        p_col = z(TOT, C)
        _render_bwd_cull_kernel[(nt,)](
            muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, bgf,
            starts_b[b], T.contiguous(), asum.contiguous(), fg.contiguous(), go,
            dummy, dummy, dummy,
            p_muy, p_mux, p_sy, p_sx, p_th, p_op, p_col,
            b, P, C, ntx, H, W,
            C_POW=C_POW, TILE=TILE, BLOCK_N=_BWD_BN, HAS_THETA=has_th, FIELDS=False, ODDS=odds,
            num_warps=_BWD_WARPS, num_stages=_BWD_STAGES,
        )
        g_muy[b].index_add_(0, gid, p_muy); g_mux[b].index_add_(0, gid, p_mux)
        g_sy[b].index_add_(0, gid, p_sy); g_sx[b].index_add_(0, gid, p_sx)
        g_op[b].index_add_(0, gid, p_op); g_col[b].index_add_(0, gid, p_col)
        if has_th:
            g_th[b].index_add_(0, gid, p_th)
    g_mu = torch.stack([g_muy, g_mux], dim=-1)
    g_scale = torch.stack([g_sy, g_sx], dim=-1)
    g_bg = (go * T.unsqueeze(-1)).sum(dim=1)
    return g_mu, g_scale, g_col, g_op, g_bg, (g_th if has_th else None)


def _backward_cull_fields(dS_log, dS_a, dS_ac, mu, scale, color, opacity, theta, tiles,
                          H, W, TILE=8, BLOCK_N=64, odds=False):
    """Backward of the FIELDS forward: upstream grads are the partial-sum fields' grads (dS_log [B,P],
    dS_a [B,P], dS_ac [B,P,C]) -- NOT image grads. Returns per-param grads (no bg grad; bg is applied in
    the torch combine, so its grad is produced there). Re-uses the per-pair kernel via FIELDS=True."""
    B, N, C = color.shape
    P = H * W
    dev = mu.device
    gid_b, starts_b, ntx, nt = tiles
    pb = _cull_param_buffers(mu, scale, color, opacity, theta)
    muy, mux, sy, sx, op, col = pb["muy"], pb["mux"], pb["sy"], pb["sx"], pb["op"], pb["col"]
    th, has_th = pb["th"], pb["has_th"]
    dslog = dS_log.contiguous().float(); dsa = dS_a.contiguous().float(); dsac = dS_ac.contiguous().float()

    z = lambda *s: torch.zeros(s, device=dev, dtype=torch.float32)
    g_muy, g_mux, g_sy, g_sx, g_op, g_th = z(B, N), z(B, N), z(B, N), z(B, N), z(B, N), z(B, N)
    g_col = z(B, N, C)
    dummy = torch.empty(1, device=dev, dtype=torch.float32)            # unused composite ptrs
    C_POW = triton.next_power_of_2(C)
    for b in range(B):
        gid = gid_b[b].long()
        TOT = gid.shape[0]
        muy_s, mux_s, sy_s, sx_s, op_s, col_s, th_s = _gather_pairs(
            muy[b], mux[b], sy[b], sx[b], op[b], col[b], th[b] if has_th else None, gid, has_th, dev)
        p_muy, p_mux, p_sy, p_sx, p_op, p_th = z(TOT), z(TOT), z(TOT), z(TOT), z(TOT), z(TOT)
        p_col = z(TOT, C)
        _render_bwd_cull_kernel[(nt,)](
            muy_s, mux_s, sy_s, sx_s, th_s, op_s, col_s, dummy,
            starts_b[b], dummy, dummy, dummy, dummy,
            dslog, dsa, dsac,
            p_muy, p_mux, p_sy, p_sx, p_th, p_op, p_col,
            b, P, C, ntx, H, W,
            C_POW=C_POW, TILE=TILE, BLOCK_N=_BWD_BN, HAS_THETA=has_th, FIELDS=True, ODDS=odds,
            num_warps=_BWD_WARPS, num_stages=_BWD_STAGES,
        )
        g_muy[b].index_add_(0, gid, p_muy); g_mux[b].index_add_(0, gid, p_mux)
        g_sy[b].index_add_(0, gid, p_sy); g_sx[b].index_add_(0, gid, p_sx)
        g_op[b].index_add_(0, gid, p_op); g_col[b].index_add_(0, gid, p_col)
        if has_th:
            g_th[b].index_add_(0, gid, p_th)
    g_mu = torch.stack([g_muy, g_mux], dim=-1)
    g_scale = torch.stack([g_sy, g_sx], dim=-1)
    return g_mu, g_scale, g_col, g_op, (g_th if has_th else None)


class _AlphaRenderCull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, color, opacity, bg, theta, H, W, TILE, cutoff, odds=False):
        out, (T, asum, fg), tiles = _forward_cull(mu, scale, color, opacity, bg, theta, H, W, TILE, cutoff,
                                                  odds=odds)
        ctx.has_theta = theta is not None
        ctx.HW = (H, W, TILE)
        ctx.odds = odds
        ctx.tiles = tiles
        ctx.save_for_backward(mu, scale, color, opacity, bg,
                              theta if theta is not None else mu.new_zeros(1), T, asum, fg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, scale, color, opacity, bg, theta, T, asum, fg = ctx.saved_tensors
        H, W, TILE = ctx.HW
        th = theta if ctx.has_theta else None
        g_mu, g_scale, g_col, g_op, g_bg, g_th = _backward_cull(
            grad_out, mu, scale, color, opacity, bg, th, T, asum, fg, ctx.tiles, H, W, TILE, odds=ctx.odds)
        return g_mu, g_scale, g_col, g_op, g_bg, g_th, None, None, None, None, None


class _AlphaRenderCullFields(torch.autograd.Function):
    """Render the RAW partial-sum fields (S_log, S_a, S_ac) for one gaussian GROUP at (H,W).
    Differentiable w.r.t. mu/scale/color/opacity/theta. No bg, no composite -- the monoid combine
    (sum groups, clamp, composite with bg) lives in torch (render_alpha_multires), so its grads flow
    back as the (dS_log, dS_a, dS_ac) this Function's backward consumes."""
    @staticmethod
    def forward(ctx, mu, scale, color, opacity, theta, H, W, TILE, cutoff, max_pairs, odds=False):
        slog, sa, sac, tiles = _forward_cull_fields(mu, scale, color, opacity, theta, H, W,
                                                    TILE=TILE, cutoff=cutoff, max_pairs=max_pairs, odds=odds)
        ctx.has_theta = theta is not None
        ctx.HW = (H, W, TILE)
        ctx.odds = odds
        ctx.tiles = tiles
        ctx.save_for_backward(mu, scale, color, opacity,
                              theta if theta is not None else mu.new_zeros(1))
        return slog, sa, sac

    @staticmethod
    def backward(ctx, dS_log, dS_a, dS_ac):
        mu, scale, color, opacity, theta = ctx.saved_tensors
        H, W, TILE = ctx.HW
        th = theta if ctx.has_theta else None
        g_mu, g_scale, g_col, g_op, g_th = _backward_cull_fields(
            dS_log, dS_a, dS_ac, mu, scale, color, opacity, th, ctx.tiles, H, W, TILE, odds=ctx.odds)
        # inputs: mu, scale, color, opacity, theta, H, W, TILE, cutoff, max_pairs, odds
        return g_mu, g_scale, g_col, g_op, g_th, None, None, None, None, None, None


def _upsample_field(x, Hc, Wc, H, W):
    """Bilinear-upsample a partial-sum field from coarse (Hc,Wc) to (H,W). x is [B,Pc] or [B,Pc,C], with
    P in row-major (y*W+x). align_corners=False matches the kernel's (idx+0.5)/size pixel-center sampling,
    so coarse and fine fields share one coordinate convention. Differentiable (autograd handles the grad)."""
    B = x.shape[0]
    if x.dim() == 2:
        g = x.view(B, 1, Hc, Wc)
        y = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        return y.reshape(B, H * W)
    C = x.shape[-1]
    g = x.view(B, Hc, Wc, C).permute(0, 3, 1, 2)
    y = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
    return y.permute(0, 2, 3, 1).reshape(B, H * W, C)


def render_alpha_multires(mu, log_scale, color, opacity, bg, H, W, theta=None, n_coarse=None,
                          coarse_factor=8, cutoff=3.0, TILE=8, max_pairs=48_000_000, odds=False):
    """Memory-BOUNDED multi-resolution differentiable render. Returns [B,C,H,W].

    Splits gaussians into a COARSE group (first `n_coarse` -- the large base-layer splats) rendered at
    (H/coarse_factor, W/coarse_factor) and a FINE group rendered at native (H,W). Alpha compositing is a
    commutative MONOID, so we render each group's partial sums (S_log, S_a, S_ac) independently, bilinearly
    upsample the coarse fields, SUM them, and composite ONCE with bg -- exact across groups (the only
    approximation is band-limited upsampling of the smooth coarse field, ~1% at the coarse-scale floor).

    Why it can't OOM: the coarse (large/global) splats are the ones that bin into ~every tile, but at
    H/coarse_factor that's coarse_factor^2 fewer tiles; the fine splats are local so their full-res pair
    count is small. `max_pairs` hard-guards both passes. This replaces single-res culling, which materializes
    O(N_coarse * n_tiles) pairs at full res and crashed the box at 1024px.

    n_coarse=None -> all splats through the full-res pass (single-res fields path; for callers w/o a split)."""
    B, N, C = color.shape
    scale = activate_scale(log_scale)
    has_th = theta is not None

    def fields(lo, hi, H_, W_):
        th = theta[:, lo:hi] if has_th else None
        return _AlphaRenderCullFields.apply(mu[:, lo:hi], scale[:, lo:hi], color[:, lo:hi],
                                            opacity[:, lo:hi], th, H_, W_, TILE, cutoff, max_pairs, odds)

    if n_coarse is None or n_coarse <= 0 or n_coarse >= N:
        slog, sa, sac = fields(0, N, H, W)
    else:
        # keep the coarse grid >= 8px/side: on small images a /coarse_factor grid would be too few texels
        # to band-limit-sample the coarse splats (upsampling error grows). Small images have no pair-blowup
        # anyway, so a gentler downsample costs nothing.
        cf = min(coarse_factor, max(1, min(H, W) // 8))
        Hc = max(1, H // cf); Wc = max(1, W // cf)
        slc, sac_a, sac_c = fields(0, n_coarse, Hc, Wc)               # coarse @ low res
        slf, saf, sacf = fields(n_coarse, N, H, W)                    # fine @ native
        slog = slf + _upsample_field(slc, Hc, Wc, H, W)              # monoid: sum partial sums
        sa = saf + _upsample_field(sac_a, Hc, Wc, H, W)
        sac = sacf + _upsample_field(sac_c, Hc, Wc, H, W)

    asum = sa.clamp_min(1e-8)
    T = slog.exp()
    fg = sac / asum.unsqueeze(-1)
    img = (1.0 - T).unsqueeze(-1) * fg + T.unsqueeze(-1) * bg.float().unsqueeze(1)   # [B,P,C]
    return img.reshape(B, H, W, C).permute(0, 3, 1, 2)


def _octave_of(scale_px, base_texels=8.0, max_octaves=6):
    """Route each splat to an octave o (rendered at H/2^o) by its full-res pixel scale. We want the
    splat's sigma AT ITS OCTAVE in (base_texels/2, base_texels], so its cutoff*sigma bbox spans a BOUNDED
    number of tiles regardless of absolute scale: o = clamp(ceil(log2(sigma_px / base_texels)), 0, max).
    With base_texels=8 (=TILE) and cutoff 3, bbox <= 6*8 = 48px = 6 tiles/side -> <=36 tiles/splat, so the
    total (tile,gaussian) pair count is <= 36*N at ANY resolution. scale_px is [B,N] (detached)."""
    o = torch.ceil(torch.log2((scale_px / base_texels).clamp_min(1e-6)))
    return o.clamp(0, max_octaves).to(torch.int64)


def render_alpha_octave(mu, log_scale, color, opacity, bg, H, W, theta=None, cutoff=3.0, TILE=8,
                        base_texels=8.0, max_octaves=6, max_pairs=48_000_000, odds=False):
    """Memory-bounded differentiable render via SCALE-routed octave LOD. Returns [B,C,H,W].

    Generalizes render_alpha_multires (which routed by the coarse/fine HEAD) to routing by each splat's
    actual scale: every splat is rendered at the octave resolution where its footprint is a bounded number
    of tiles, the octaves' partial-sum fields are bilinearly upsampled to native and summed (alpha
    compositing monoid -> exact across octaves), and composited once with bg. Because tiles/splat is bounded
    by construction (~<=36), total pairs <= 36*N at ANY resolution -- this is what unblocks TRUE native res
    (>1024) where the single coarse/fine split's FINE pass still exploded (~H^4 pairs at init).

    Robust to training drift (a fine splat that grows, or a coarse one that shrinks, just re-routes octave)."""
    B, N, C = color.shape
    scale = activate_scale(log_scale)
    has_th = theta is not None
    rep = float(min(H, W))
    scale_px = scale.detach().amax(-1) * rep                          # [B,N] max-axis sigma in px @full res
    oct = _octave_of(scale_px, base_texels, max_octaves)              # [B,N] int (routing; non-diff)

    P = H * W
    slog_l, sa_l, sac_l = [], [], []
    for b in range(B):
        slog_b = mu.new_zeros(P); sa_b = mu.new_zeros(P); sac_b = mu.new_zeros(P, C)
        ob = oct[b]
        for o in range(max_octaves + 1):
            idx = (ob == o).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            Ho = max(8, H >> o); Wo = max(8, W >> o)
            m = mu[b:b + 1, idx]; s = scale[b:b + 1, idx]; cl = color[b:b + 1, idx]
            op_ = opacity[b:b + 1, idx]; th = theta[b:b + 1, idx] if has_th else None
            sl, saa, sc = _AlphaRenderCullFields.apply(m, s, cl, op_, th, Ho, Wo, TILE, cutoff, max_pairs, odds)
            if o == 0:
                slog_b = slog_b + sl[0]; sa_b = sa_b + saa[0]; sac_b = sac_b + sc[0]
            else:
                slog_b = slog_b + _upsample_field(sl, Ho, Wo, H, W)[0]
                sa_b = sa_b + _upsample_field(saa, Ho, Wo, H, W)[0]
                sac_b = sac_b + _upsample_field(sc, Ho, Wo, H, W)[0]
        slog_l.append(slog_b); sa_l.append(sa_b); sac_l.append(sac_b)
    slog = torch.stack(slog_l); sa = torch.stack(sa_l); sac = torch.stack(sac_l)

    asum = sa.clamp_min(1e-8)
    T = slog.exp()
    fg = sac / asum.unsqueeze(-1)
    img = (1.0 - T).unsqueeze(-1) * fg + T.unsqueeze(-1) * bg.float().unsqueeze(1)
    return img.reshape(B, H, W, C).permute(0, 3, 1, 2)


def _selftest_cull_backward(device="cuda"):
    """Culled grads should match the EXACT triton grads (already validated vs autograd) as cutoff grows."""
    torch.manual_seed(3)
    B, N, C, H, W = 2, 1024, 3, 64, 64

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        ls = (torch.rand(B, N, 2, device=device) - 2.0).requires_grad_(True)
        col = torch.rand(B, N, C, device=device, requires_grad=True)
        op = torch.rand(B, N, device=device, requires_grad=True)
        bg = torch.rand(B, C, device=device, requires_grad=True)
        th = (torch.rand(B, N, device=device) * 3.14).requires_grad_(True)
        return mu, ls, col, op, bg, th

    gC = torch.randn(B, C, H, W, device=device)
    L = leaves()
    out_ex = render_alpha_full_triton(*L[:5], H, W, theta=L[5])
    (out_ex * gC).sum().backward()
    ex = {n: t.grad.clone() for n, t in zip("mu ls col op bg th".split(), L)}

    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    out_cu = render_alpha_full_triton(*L2[:5], H, W, theta=L2[5], cull=True, cutoff=5.0)
    fdiff = (out_cu - out_ex.detach()).abs().max().item()
    (out_cu * gC).sum().backward()
    cu = {n: t.grad for n, t in zip("mu ls col op bg th".split(), L2)}
    print(f"--- cull backward (vs exact triton, cutoff=5)  forward max|diff|={fdiff:.2e}")
    all_ok = fdiff < 5e-3
    for n in "mu ls col op bg th".split():
        d = (cu[n] - ex[n]).abs()
        rel = d.max().item() / (ex[n].abs().max().item() + 1e-12)
        ok = rel < 1e-2
        all_ok &= ok
        print(f"    {n:4s} max|diff|={d.max().item():.3e}  rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[cull-bwd] PASS" if all_ok else "[cull-bwd] FAIL")
    return all_ok


def _selftest_cull_forward(device="cuda"):
    """Culled forward should converge to the exact render as cutoff grows."""
    torch.manual_seed(2)
    B, N, C, H, W = 2, 1024, 3, 64, 64
    mu = torch.rand(B, N, 2, device=device)
    log_scale = (torch.rand(B, N, 2, device=device) - 2.5)            # smaller splats -> culling matters
    color = torch.rand(B, N, C, device=device)
    opacity = torch.rand(B, N, device=device)
    bg = torch.rand(B, C, device=device)
    theta = torch.rand(B, N, device=device) * 3.14159
    scale = activate_scale(log_scale)

    ref = render_alpha_full(mu, log_scale, color, opacity, bg, H, W, theta=theta, chunk_pixels=0)
    ref = ref.permute(0, 2, 3, 1).reshape(B, H * W, C)
    print("--- cull forward (converge to exact as cutoff grows)")
    ok = False
    for cutoff in (3.0, 4.0, 5.0):
        out, _, _ = _forward_cull(mu, scale, color, opacity, bg, theta, H, W, TILE=16, cutoff=cutoff)
        d = (out - ref).abs().max().item()
        print(f"    cutoff={cutoff}: max|diff vs exact|={d:.3e}")
        ok = ok or d < 5e-3
    print("[cull-fwd] PASS" if ok else "[cull-fwd] FAIL")
    return ok


# ----------------------------------------------------------------------------- self-test (forward)
def _selftest_forward(device="cuda"):
    torch.manual_seed(0)
    B, N, C, H, W = 2, 512, 3, 48, 48
    mu = torch.rand(B, N, 2, device=device)
    log_scale = (torch.rand(B, N, 2, device=device) - 2.0)           # small-ish scales
    color = torch.rand(B, N, C, device=device)
    opacity = torch.rand(B, N, device=device)
    bg = torch.rand(B, C, device=device)
    theta = torch.rand(B, N, device=device) * 3.14159

    scale = activate_scale(log_scale)
    py_px = _pixels(H, W, device, torch.float32)                     # [H*W,2]
    out_t, T, asum, fg = _forward(mu, scale, color, opacity, bg, py_px[:, 0], py_px[:, 1], theta)
    ref = render_alpha_full(mu, log_scale, color, opacity, bg, H, W, theta=theta, chunk_pixels=0)
    ref = ref.permute(0, 2, 3, 1).reshape(B, H * W, C)               # [B,P,C]

    diff = (out_t - ref).abs()
    print(f"[fwd] out max|diff|={diff.max().item():.3e} mean|diff|={diff.mean().item():.3e} "
          f"(ref range [{ref.min().item():.3f},{ref.max().item():.3f}])")
    # also test the no-theta path
    out_n, *_ = _forward(mu, scale, color, opacity, bg, py_px[:, 0], py_px[:, 1], None)
    ref_n = render_alpha_full(mu, log_scale, color, opacity, bg, H, W, theta=None, chunk_pixels=0)
    ref_n = ref_n.permute(0, 2, 3, 1).reshape(B, H * W, C)
    dn = (out_n - ref_n).abs()
    print(f"[fwd] no-theta max|diff|={dn.max().item():.3e} mean|diff|={dn.mean().item():.3e}")
    ok = diff.max().item() < 2e-3 and dn.max().item() < 2e-3
    print("[fwd] PASS" if ok else "[fwd] FAIL")
    return ok


def _selftest_backward(device="cuda"):
    """Compare Triton analytic grads to autograd of the reference renderer (same math)."""
    torch.manual_seed(1)
    B, N, C, H, W = 2, 256, 3, 32, 32

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        log_scale = (torch.rand(B, N, 2, device=device) - 1.5).requires_grad_(True)
        color = torch.rand(B, N, C, device=device, requires_grad=True)
        opacity = torch.rand(B, N, device=device, requires_grad=True)
        bg = torch.rand(B, C, device=device, requires_grad=True)
        theta = (torch.rand(B, N, device=device) * 3.14159).requires_grad_(True)
        return mu, log_scale, color, opacity, bg, theta

    gC = torch.randn(B, C, H, W, device=device)
    all_ok = True
    for has_theta in (True, False):
        names = "mu ls col op bg th".split() if has_theta else "mu ls col op bg".split()
        L = leaves()
        mu, log_scale, color, opacity, bg, theta = L
        th = theta if has_theta else None
        out_ref = render_alpha_full(mu, log_scale, color, opacity, bg, H, W, theta=th, chunk_pixels=0)
        (out_ref * gC).sum().backward()
        tensors = [mu, log_scale, color, opacity, bg] + ([theta] if has_theta else [])
        ref = {n: t.grad.clone() for n, t in zip(names, tensors)}

        L2 = [t.detach().clone().requires_grad_(True) for t in L]
        mu2, ls2, col2, op2, bg2, th2 = L2
        tname = th2 if has_theta else None
        out_tri = render_alpha_full_triton(mu2, ls2, col2, op2, bg2, H, W, theta=tname)
        fdiff = (out_tri - out_ref.detach()).abs().max().item()
        (out_tri * gC).sum().backward()
        tensors2 = [mu2, ls2, col2, op2, bg2] + ([th2] if has_theta else [])
        tri = {n: t.grad for n, t in zip(names, tensors2)}

        tag = "theta" if has_theta else "no-theta"
        print(f"--- backward [{tag}]  forward max|diff|={fdiff:.2e}")
        for n in names:
            d = (tri[n] - ref[n]).abs()
            rel = d.max().item() / (ref[n].abs().max().item() + 1e-12)
            ok = rel < 2e-3 and fdiff < 2e-3
            all_ok &= ok
            print(f"    {n:4s} max|diff|={d.max().item():.3e}  rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[bwd] PASS" if all_ok else "[bwd] FAIL")
    return all_ok


def _bench_memory(device="cuda"):
    """Substantiate the O(P) memory claim: peak alloc for a full-res differentiable render+backward,
    reference (retains the per-chunk falloff graph) vs the recompute-in-backward Triton path."""
    import time
    for H, N in ((256, 8192), (384, 8192)):
        W = H
        torch.manual_seed(0)
        def mk():
            return (torch.rand(1, N, 2, device=device, requires_grad=True),
                    (torch.rand(1, N, 2, device=device) - 1.5).requires_grad_(True),
                    torch.rand(1, N, 3, device=device, requires_grad=True),
                    torch.rand(1, N, device=device, requires_grad=True),
                    torch.rand(1, 3, device=device, requires_grad=True),
                    (torch.rand(1, N, device=device) * 3.14).requires_grad_(True))
        gC = torch.randn(1, 3, H, W, device=device)
        res = {}
        for tag, fn in (("ref", render_alpha_full), ("triton", render_alpha_full_triton)):
            mu, ls, col, op, bg, th = mk()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if tag == "ref":
                out = fn(mu, ls, col, op, bg, H, W, theta=th, chunk_pixels=4096)
            else:
                out = fn(mu, ls, col, op, bg, H, W, theta=th)
            (out * gC).sum().backward()
            torch.cuda.synchronize()
            res[tag] = (torch.cuda.max_memory_allocated() / 1e6, time.time() - t0)
        rm, rt = res["ref"]; tm, tt = res["triton"]
        print(f"[mem] {H}x{W} N={N}: ref {rm:7.0f} MB / {rt:5.2f}s   triton {tm:7.0f} MB / {tt:5.2f}s "
              f"  mem x{rm/max(tm,1e-6):.1f}")


def _probe_cull(device="cuda"):
    """Break down where culled time goes: tile build vs fwd kernel vs bwd kernel vs index_add scatter."""
    import time
    H = W = 512; N = 16384
    torch.manual_seed(0)
    mu = torch.rand(1, N, 2, device=device)
    # realistic trained splat size: N splats tiling 512^2 -> ~5px radius -> scale ~0.01 normalized
    scale = activate_scale(torch.rand(1, N, 2, device=device) - 4.5)
    color = torch.rand(1, N, 3, device=device)
    opacity = torch.rand(1, N, device=device)
    bg = torch.rand(1, 3, device=device)
    theta = torch.rand(1, N, device=device) * 3.14
    gout = torch.randn(1, H * W, 3, device=device)
    sync = torch.cuda.synchronize

    for TILE in (16, 8, 4):
        sync(); t0 = time.time()
        for _ in range(3):
            tiles = _build_tiles(mu, scale, H, W, TILE, 3.0)
        sync(); t_build = (time.time() - t0) / 3
        tot = int(tiles[1][0][-1].item())                              # last tile_start = total pairs
        sync(); t0 = time.time()
        for _ in range(3):
            out, (T, asum, fg), tl_ = _forward_cull(mu, scale, color, opacity, bg, theta, H, W, TILE, 3.0)
        sync(); t_fwd = (time.time() - t0) / 3
        sync(); t0 = time.time()
        for _ in range(3):
            _backward_cull(gout, mu, scale, color, opacity, bg, theta, T, asum, fg, tl_, H, W, TILE)
        sync(); t_bwd = (time.time() - t0) / 3
        print(f"[probe] TILE={TILE} pairs={tot} ({tot/N:.1f}/gauss): build {t_build*1e3:.0f}ms  "
              f"fwd {t_fwd*1e3:.0f}ms  bwd {t_bwd*1e3:.0f}ms")


def _bench_cull(device="cuda"):
    """Cull's purpose is speed at large N with localized splats. Time exact vs culled fwd+bwd."""
    import time
    for H, N in ((512, 16384),):
        W = H
        torch.manual_seed(0)
        def mk():
            return (torch.rand(1, N, 2, device=device, requires_grad=True),
                    (torch.rand(1, N, 2, device=device) - 4.5).requires_grad_(True),   # realistic small splats
                    torch.rand(1, N, 3, device=device, requires_grad=True),
                    torch.rand(1, N, device=device, requires_grad=True),
                    torch.rand(1, 3, device=device, requires_grad=True),
                    (torch.rand(1, N, device=device) * 3.14).requires_grad_(True))
        gC = torch.randn(1, 3, H, W, device=device)
        res = {}
        for tag, kw in (("exact", {}), ("cull", {"cull": True, "cutoff": 3.0})):
            mu, ls, col, op, bg, th = mk()
            torch.cuda.synchronize(); t0 = time.time()
            out = render_alpha_full_triton(mu, ls, col, op, bg, H, W, theta=th, **kw)
            (out * gC).sum().backward()
            torch.cuda.synchronize()
            res[tag] = time.time() - t0
        print(f"[cull-speed] {H}x{W} N={N} fwd+bwd: exact {res['exact']:.2f}s  cull {res['cull']:.2f}s "
              f"  speedup x{res['exact']/max(res['cull'],1e-6):.1f}")


def _selftest_fields_equiv(device="cuda"):
    """EXACT-equivalence: render_alpha_multires(n_coarse=None) must equal render_alpha_full_triton(cull=True)
    in BOTH forward and backward. Same kernel math; the only difference is the composite is deferred to
    torch (sum partial sums -> clamp -> composite) instead of done in-kernel. No approximation here, so this
    isolates whether the FIELDS forward kernel, the torch monoid-combine, and the FIELDS backward kernel are
    correct. This is the load-bearing correctness test for the redesign."""
    torch.manual_seed(7)
    B, N, C, H, W = 2, 1024, 3, 64, 64

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        ls = (torch.rand(B, N, 2, device=device) - 2.5).requires_grad_(True)
        col = torch.rand(B, N, C, device=device, requires_grad=True)
        op = torch.rand(B, N, device=device, requires_grad=True)
        bg = torch.rand(B, C, device=device, requires_grad=True)
        th = (torch.rand(B, N, device=device) * 3.14).requires_grad_(True)
        return mu, ls, col, op, bg, th

    gC = torch.randn(B, C, H, W, device=device)
    names = "mu ls col op bg th".split()
    L = leaves()
    ref = render_alpha_full_triton(*L[:5], H, W, theta=L[5], cull=True, cutoff=3.0)
    (ref * gC).sum().backward()
    rg = {n: t.grad.clone() for n, t in zip(names, L)}

    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    out = render_alpha_multires(*L2[:5], H, W, theta=L2[5], n_coarse=None, cutoff=3.0)
    fdiff = (out - ref.detach()).abs().max().item()
    (out * gC).sum().backward()
    mg = {n: t.grad for n, t in zip(names, L2)}
    print(f"--- fields-equiv (multires n_coarse=None vs cull)  forward max|diff|={fdiff:.2e}")
    all_ok = fdiff < 1e-4
    for n in names:
        d = (mg[n] - rg[n]).abs()
        rel = d.max().item() / (rg[n].abs().max().item() + 1e-12)
        ok = rel < 2e-3
        all_ok &= ok
        print(f"    {n:4s} max|diff|={d.max().item():.3e}  rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[fields-equiv] PASS" if all_ok else "[fields-equiv] FAIL")
    return all_ok


def _selftest_multires_split(device="cuda"):
    """Approximation bound: multires (coarse@H/8 + fine@native) vs the exact full-res render of ALL splats.
    The coarse fields are bilinearly upsampled (band-limited approx). With genuinely large coarse splats
    (well-sampled at H/8) and small fine splats, the error should be small. Also finite-difference-checks
    the split backward on the coarse group (the path autograd can't trivially verify)."""
    torch.manual_seed(11)
    B, C, H, W = 1, 3, 128, 128
    Ncoarse, Nfine = 64, 512
    N = Ncoarse + Nfine
    mu = torch.rand(B, N, 2, device=device)
    ls = torch.empty(B, N, 2, device=device)
    ls[:, :Ncoarse] = math.log(math.exp(0.12) - 1.0)                  # coarse: large (sigma~0.12, ~15px@128)
    ls[:, Ncoarse:] = math.log(math.exp(0.015) - 1.0)                 # fine: small (~2px@128)
    col = torch.rand(B, N, C, device=device)
    op = torch.rand(B, N, device=device) * 0.5
    bg = torch.rand(B, C, device=device)
    th = torch.rand(B, N, device=device) * 3.14159

    exact = render_alpha_full_triton(mu, ls, col, op, bg, H, W, theta=th, cull=False)
    mr = render_alpha_multires(mu, ls, col, op, bg, H, W, theta=th, n_coarse=Ncoarse, coarse_factor=8)
    rel = (mr - exact).abs().mean().item() / (exact.abs().mean().item() + 1e-12)
    mx = (mr - exact).abs().max().item()
    print(f"--- multires-split (coarse@H/8) vs exact full-res:  mean-rel={rel:.3e}  max|diff|={mx:.3e}")
    ok_fwd = rel < 0.03

    # finite-diff the split backward w.r.t. one coarse mu coord
    mu_g = mu.clone().requires_grad_(True)
    g = torch.randn(B, C, H, W, device=device)
    out = render_alpha_multires(mu_g, ls, col, op, bg, H, W, theta=th, n_coarse=Ncoarse, coarse_factor=8)
    (out * g).sum().backward()
    ana = mu_g.grad[0, 5, 0].item()
    eps = 1e-3
    mu_p = mu.clone(); mu_p[0, 5, 0] += eps
    mu_m = mu.clone(); mu_m[0, 5, 0] -= eps
    lp = (render_alpha_multires(mu_p, ls, col, op, bg, H, W, theta=th, n_coarse=Ncoarse) * g).sum().item()
    lm = (render_alpha_multires(mu_m, ls, col, op, bg, H, W, theta=th, n_coarse=Ncoarse) * g).sum().item()
    fd = (lp - lm) / (2 * eps)
    grel = abs(ana - fd) / (abs(fd) + 1e-6)
    ok_bwd = grel < 5e-2
    print(f"    coarse-mu grad: analytic={ana:.4f} finite-diff={fd:.4f} rel={grel:.2e} "
          f"{'ok' if ok_bwd else 'FAIL'}")
    print("[multires-split] PASS" if (ok_fwd and ok_bwd) else "[multires-split] FAIL")
    return ok_fwd and ok_bwd


def _probe_multires(device="cuda"):
    """Repro the crash config (large coarse splats @ high res) and show multires is BOUNDED where single-res
    cull would explode. Reports pair counts + peak memory for both passes."""
    torch.manual_seed(0)
    B, C, H, W = 1, 3, 1024, 1024
    grid = (H // 8) * (W // 8)
    Ncoarse, Nfine = grid * 2, grid * 2                               # ~M=2 coarse + 2 fine @ stride 8
    N = Ncoarse + Nfine
    mu = torch.rand(B, N, 2, device=device)
    ls = torch.empty(B, N, 2, device=device)
    ls[:, :Ncoarse] = math.log(math.exp(0.20) - 1.0)                  # the crash-causing init coarse scale
    ls[:, Ncoarse:] = math.log(math.exp(0.018) - 1.0)
    col = torch.rand(B, N, C, device=device, requires_grad=True)
    op = torch.rand(B, N, device=device, requires_grad=True)
    mu.requires_grad_(True); ls.requires_grad_(True)
    bg = torch.rand(B, C, device=device, requires_grad=True)
    print(f"--- probe multires @ {H}x{W}, N={N} (coarse={Ncoarse} @ scale 0.20, fine={Nfine} @ 0.018)")
    # pair counts: coarse@H/8 vs what coarse would cost at full res (the crash)
    sc = activate_scale(ls)
    gid_c, _, _, _ = _build_tiles(mu[:, :Ncoarse].detach(), sc[:, :Ncoarse].detach(), H // 8, W // 8, 8, 3.0)
    pairs_coarse_lowres = gid_c[0].numel()
    cx_full = sc[:, :Ncoarse].max(-1).values * 3.0 * W * 2 / 8                       # ~tiles/side @ full res
    pairs_coarse_fullres = int((cx_full.clamp(max=W / 8) ** 2).sum().item())
    print(f"    coarse pairs: {pairs_coarse_lowres/1e6:.1f}M @H/8  vs  ~{pairs_coarse_fullres/1e6:.0f}M "
          f"@full-res (the OOM)")
    torch.cuda.reset_peak_memory_stats()
    out = render_alpha_multires(mu, ls, col, op, bg, H, W, n_coarse=Ncoarse, coarse_factor=8)
    (out.float().pow(2).mean()).backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"    multires fwd+bwd OK. peak CUDA mem = {peak:.2f} GB  (single-res cull here ~400M pairs -> OOM)")
    return peak < 20.0


def _selftest_octave_equiv(device="cuda"):
    """max_octaves=0 forces every splat to octave 0 (full res, one group) -> must equal cull EXACTLY (fwd
    +bwd). Validates the octave plumbing (per-octave gather, monoid accumulate, composite) is correct."""
    torch.manual_seed(13)
    B, N, C, H, W = 2, 1024, 3, 64, 64

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        ls = (torch.rand(B, N, 2, device=device) - 2.5).requires_grad_(True)
        col = torch.rand(B, N, C, device=device, requires_grad=True)
        op = torch.rand(B, N, device=device, requires_grad=True)
        bg = torch.rand(B, C, device=device, requires_grad=True)
        th = (torch.rand(B, N, device=device) * 3.14).requires_grad_(True)
        return mu, ls, col, op, bg, th

    gC = torch.randn(B, C, H, W, device=device)
    names = "mu ls col op bg th".split()
    L = leaves()
    ref = render_alpha_full_triton(*L[:5], H, W, theta=L[5], cull=True, cutoff=3.0)
    (ref * gC).sum().backward()
    rg = {n: t.grad.clone() for n, t in zip(names, L)}
    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    out = render_alpha_octave(*L2[:5], H, W, theta=L2[5], cutoff=3.0, max_octaves=0)
    fdiff = (out - ref.detach()).abs().max().item()
    (out * gC).sum().backward()
    mg = {n: t.grad for n, t in zip(names, L2)}
    print(f"--- octave-equiv (max_octaves=0 vs cull)  forward max|diff|={fdiff:.2e}")
    all_ok = fdiff < 1e-4
    for n in names:
        rel = (mg[n] - rg[n]).abs().max().item() / (rg[n].abs().max().item() + 1e-12)
        ok = rel < 2e-3; all_ok &= ok
        print(f"    {n:4s} rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[octave-equiv] PASS" if all_ok else "[octave-equiv] FAIL")
    return all_ok


def _selftest_octave_split(device="cuda"):
    """Multi-octave routing vs exact full-res render of ALL splats, across a WIDE scale range (tiny detail
    -> huge base layer, so several octaves are exercised). Band-limited per octave -> small error. Plus a
    finite-diff backward check on a large splat (routed to a downsampled octave)."""
    torch.manual_seed(17)
    B, C, H, W = 1, 3, 256, 256
    N = 800
    mu = torch.rand(B, N, 2, device=device)
    # log-uniform scales from ~1.5px (0.006) to ~80px (0.31) normalized -> spans ~6 octaves
    sig = torch.exp(torch.rand(B, N, device=device) * (math.log(0.31) - math.log(0.006)) + math.log(0.006))
    ls = torch.log(torch.exp(sig).clamp_min(1 + 1e-6) - 1.0).unsqueeze(-1).expand(B, N, 2).contiguous()
    col = torch.rand(B, N, C, device=device); op = torch.rand(B, N, device=device) * 0.4
    th = torch.rand(B, N, device=device) * 3.14159
    bgv = torch.rand(B, C, device=device)
    exact = render_alpha_full_triton(mu, ls, col, op, bgv, H, W, theta=th, cull=False)
    oc = render_alpha_octave(mu, ls, col, op, bgv, H, W, theta=th, cutoff=3.0)
    rel = (oc - exact).abs().mean().item() / (exact.abs().mean().item() + 1e-12)
    print(f"--- octave-split vs exact full-res (wide scale range):  mean-rel={rel:.3e}")
    ok_fwd = rel < 0.03

    mu_g = mu.clone().requires_grad_(True); g = torch.randn(B, C, H, W, device=device)
    (render_alpha_octave(mu_g, ls, col, op, bgv, H, W, theta=th) * g).sum().backward()
    j = int(sig[0].argmax().item())                                  # the largest splat (deep octave)
    ana = mu_g.grad[0, j, 0].item()
    eps = 1e-3
    mp = mu.clone(); mp[0, j, 0] += eps; mm = mu.clone(); mm[0, j, 0] -= eps
    fd = ((render_alpha_octave(mp, ls, col, op, bgv, H, W, theta=th) * g).sum().item()
          - (render_alpha_octave(mm, ls, col, op, bgv, H, W, theta=th) * g).sum().item()) / (2 * eps)
    grel = abs(ana - fd) / (abs(fd) + 1e-6)
    ok_bwd = grel < 5e-2
    print(f"    large-splat mu grad: analytic={ana:.4f} fd={fd:.4f} rel={grel:.2e} {'ok' if ok_bwd else 'FAIL'}")
    print("[octave-split] PASS" if (ok_fwd and ok_bwd) else "[octave-split] FAIL")
    return ok_fwd and ok_bwd


def _probe_octave(device="cuda"):
    """THE win: at 1536 & 2048 with init scales (where the coarse/fine split's fine pass exploded ->
    PairBudgetExceeded), octave LOD must render BOUNDED (pairs <= ~36*N) and fit. Reports peak mem."""
    torch.manual_seed(0)
    ok = True
    for H in (1024, 1536, 2048):
        W = H; grid = (H // 8) * (W // 8); Nc = grid * 2; Nf = grid * 2; N = Nc + Nf
        mu = torch.rand(1, N, 2, device=device)
        ls = torch.empty(1, N, 2, device=device)
        ls[:, :Nc] = math.log(math.exp(0.20) - 1.0); ls[:, Nc:] = math.log(math.exp(0.018) - 1.0)
        col = torch.rand(1, N, 3, device=device, requires_grad=True)
        op = torch.rand(1, N, device=device, requires_grad=True)
        mu.requires_grad_(True); ls.requires_grad_(True); bg = torch.rand(1, 3, device=device, requires_grad=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            out = render_alpha_octave(mu, ls, col, op, bg, H, W, max_pairs=48_000_000)
            out.float().pow(2).mean().backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"    octave @ {H}x{W} N={N}: fwd+bwd OK  peak={peak:.2f}GB  (<=36*N={36 * N / 1e6:.0f}M pairs)")
        except PairBudgetExceeded:
            print(f"    octave @ {H}x{W}: PairBudgetExceeded  FAIL (should be bounded!)"); ok = False
        del mu, ls, col, op, bg; torch.cuda.empty_cache()
    print("[octave-probe] PASS" if ok else "[octave-probe] FAIL")
    return ok


def _torch_odds_ref(mu, log_scale, color, opacity, bg, H, W, theta=None):
    """Independent pure-torch reference for F2 ODDS compositing (materialized [B,P,N], exact, autograd-
    differentiable). fg = Sum(w*c)/Sum(w), w=a/(1-a); T=exp(Sum log(1-a)); C=(1-T)fg+T bg. Geometry matches
    the kernel (pixel centers, rotate, maha, clamp a<=1-1e-4). Used to validate the odds kernel fwd+grads."""
    B, N, C = color.shape
    scale = activate_scale(log_scale)
    pyx = _pixels(H, W, mu.device, torch.float32)                          # [P,2] centers (py,px)
    py = pyx[:, 0]; px = pyx[:, 1]
    dy = py[None, :, None] - mu[:, None, :, 0]                              # [B,P,N]
    dx = px[None, :, None] - mu[:, None, :, 1]
    if theta is not None:
        cth = theta.cos()[:, None, :]; sth = theta.sin()[:, None, :]       # [B,1,N]
        ry = cth * dy + sth * dx
        rx = -sth * dy + cth * dx
    else:
        ry, rx = dy, dx
    sy = scale[:, None, :, 0]; sx = scale[:, None, :, 1]
    maha = (ry * ry) / (sy * sy) + (rx * rx) / (sx * sx)
    a = (opacity[:, None, :] * torch.exp(-0.5 * maha)).clamp(max=1.0 - 1e-4)   # [B,P,N]
    w = a / (1.0 - a)
    S_log = torch.log(1.0 - a).sum(-1)                                      # [B,P]
    S_w = w.sum(-1)
    S_wc = (w[..., None] * color[:, None, :, :]).sum(2)                     # [B,P,C]
    T = S_log.exp()
    fg = S_wc / S_w.clamp_min(1e-8).unsqueeze(-1)
    img = (1.0 - T).unsqueeze(-1) * fg + T.unsqueeze(-1) * bg[:, None, :]
    return img.reshape(B, H, W, C).permute(0, 3, 1, 2)


def _selftest_odds(device="cuda"):
    """THE odds-math correctness test: octave render with odds=True (max_octaves=0, large cutoff so culling
    ~= exact) vs the independent materialized torch odds reference -- forward AND all grads. Validates the
    odds forward accumulation (w=a/(1-a)) and the odds backward assembly (da, dcol) end to end."""
    torch.manual_seed(23)
    B, N, C, H, W = 2, 256, 3, 32, 32

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        ls = (torch.rand(B, N, 2, device=device) - 1.5).requires_grad_(True)   # moderate scales
        col = torch.rand(B, N, C, device=device, requires_grad=True)
        op = (torch.rand(B, N, device=device) * 0.9).requires_grad_(True)      # keep a off the 1-1e-4 clamp
        bg = torch.rand(B, C, device=device, requires_grad=True)
        th = (torch.rand(B, N, device=device) * 3.14).requires_grad_(True)
        return mu, ls, col, op, bg, th

    gC = torch.randn(B, C, H, W, device=device)
    names = "mu ls col op bg th".split()
    L = leaves()
    ref = _torch_odds_ref(*L[:5], H, W, theta=L[5])
    (ref * gC).sum().backward()
    rg = {n: t.grad.clone() for n, t in zip(names, L)}

    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    out = render_alpha_octave(*L2[:5], H, W, theta=L2[5], cutoff=6.0, max_octaves=0, odds=True)
    fdiff = (out - ref.detach()).abs().max().item()
    (out * gC).sum().backward()
    mg = {n: t.grad for n, t in zip(names, L2)}
    print(f"--- odds (octave odds=True vs materialized torch odds-ref)  forward max|diff|={fdiff:.2e}")
    all_ok = fdiff < 5e-3
    for n in names:
        d = (mg[n] - rg[n]).abs()
        rel = d.max().item() / (rg[n].abs().max().item() + 1e-12)
        ok = rel < 1e-2
        all_ok &= ok
        print(f"    {n:4s} max|diff|={d.max().item():.3e}  rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[odds] PASS" if all_ok else "[odds] FAIL")
    return all_ok


def _selftest_odds_fields_equiv(device="cuda"):
    """odds plumbing: octave(odds, max_octaves=0) must equal cull(odds=True) EXACTLY (fwd+bwd) -- same
    kernels, composite deferred to torch. Isolates the FIELDS odds path (combine carries the odds sums)."""
    torch.manual_seed(29)
    B, N, C, H, W = 2, 1024, 3, 64, 64

    def leaves():
        mu = torch.rand(B, N, 2, device=device, requires_grad=True)
        ls = (torch.rand(B, N, 2, device=device) - 2.5).requires_grad_(True)
        col = torch.rand(B, N, C, device=device, requires_grad=True)
        op = (torch.rand(B, N, device=device) * 0.9).requires_grad_(True)
        bg = torch.rand(B, C, device=device, requires_grad=True)
        th = (torch.rand(B, N, device=device) * 3.14).requires_grad_(True)
        return mu, ls, col, op, bg, th

    gC = torch.randn(B, C, H, W, device=device)
    names = "mu ls col op bg th".split()
    L = leaves()
    ref = render_alpha_full_triton(*L[:5], H, W, theta=L[5], cull=True, cutoff=3.0, odds=True)
    (ref * gC).sum().backward()
    rg = {n: t.grad.clone() for n, t in zip(names, L)}
    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    out = render_alpha_octave(*L2[:5], H, W, theta=L2[5], cutoff=3.0, max_octaves=0, odds=True)
    fdiff = (out - ref.detach()).abs().max().item()
    (out * gC).sum().backward()
    mg = {n: t.grad for n, t in zip(names, L2)}
    print(f"--- odds-fields-equiv (octave odds vs cull odds)  forward max|diff|={fdiff:.2e}")
    all_ok = fdiff < 1e-4
    for n in names:
        rel = (mg[n] - rg[n]).abs().max().item() / (rg[n].abs().max().item() + 1e-12)
        ok = rel < 2e-3; all_ok &= ok
        print(f"    {n:4s} rel={rel:.2e}  {'ok' if ok else 'FAIL'}")
    print("[odds-fields-equiv] PASS" if all_ok else "[odds-fields-equiv] FAIL")
    return all_ok


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("no CUDA; Triton kernels require a GPU")
        sys.exit(1)
    f_ok = _selftest_forward(dev)
    b_ok = _selftest_backward(dev)
    c_ok = _selftest_cull_forward(dev)
    cb_ok = _selftest_cull_backward(dev)
    fe_ok = _selftest_fields_equiv(dev)
    ms_ok = _selftest_multires_split(dev)
    pm_ok = _probe_multires(dev)
    oe_ok = _selftest_octave_equiv(dev)
    os_ok = _selftest_octave_split(dev)
    op_ok = _probe_octave(dev)
    od_ok = _selftest_odds(dev)
    ofe_ok = _selftest_odds_fields_equiv(dev)
    if f_ok and b_ok:
        _bench_memory(dev)
        _bench_cull(dev)
    allok = (f_ok and b_ok and c_ok and cb_ok and fe_ok and ms_ok and pm_ok
             and oe_ok and os_ok and op_ok and od_ok and ofe_ok)
    print("\n==== ALL PASS ====" if allok else "\n==== SOME FAILED ====")
    sys.exit(0 if allok else 1)
