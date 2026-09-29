"""Content-adaptive 2D-gaussian encoder. Single forward pass image -> efficient gaussian set, rendered
with the order-independent alpha-compositing renderer (render.py) so opacity GATES and a few large splats
can own flat regions. Two variants (A/B, shared render + losses):

  variant='free'     — unclamped position offsets (gaussians migrate ±free_offset_cells, not ±1) + a Sobel
                       gradient map fed to the heads so positions/opacity are content-aware. Uniform base
                       count, but free to cluster on detail and (via opacity-L1) switch off on flats.

  variant='adaptive' — coarse->fine densification in ONE pass: a COARSE head predicts few LARGE splats;
                       the residual error E=|I - C_coarse| is computed and fed (with the coarse render) to
                       a FINE head that spends small splats only where the coarse pass under-reconstructs
                       (text/edges). Feed-forward port of 3DGS clone/split + Image-GS error-guided P_add.

Both predict per splat: mu(2), log_scale(2), theta(1), color(3, sigmoid), opacity(1, sigmoid SEPARATE gate)
plus a per-image learned background bg(3). Covariance is scale+theta (Σ=R S² Rᵀ, PSD) — same layout the
JEPA already consumes, so assemble_gaussian_params still works (opacity is the gate, not folded into color).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .render import render_alpha_full
from .render_triton import render_alpha_full_triton, render_alpha_octave


def _norm(channels, norm_type):
    if norm_type == "group":
        return nn.GroupNorm(8, channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


class _Backbone(nn.Module):
    """Strided CNN that EXPOSES its intermediate (high-res) feature maps as skips. stride=8 -> 3 downsample
    convs (fH=H/8); stride=16 -> 4 (fH=H/16, scarcity). The s2/s4 skips carry the high-frequency localization
    the strided pooling aliases away -- the fine decoder fuses them so the fine head can place a splat ON a
    2px stroke instead of averaging it to the cell-mean (the size-graded text blur). Channel widths per
    stride: s2=d/4, s4=d/2, s8=d, s16=d."""

    def __init__(self, in_ch, d, norm_type, stride):
        super().__init__()
        if stride not in (8, 16):
            raise ValueError(f"stride must be 8 or 16, got {stride}")
        self.stride = stride
        self.stage_s2 = nn.Sequential(
            nn.Conv2d(in_ch, d // 4, 7, stride=2, padding=3), nn.GELU(), _norm(d // 4, norm_type))
        self.stage_s4 = nn.Sequential(
            nn.Conv2d(d // 4, d // 2, 3, stride=2, padding=1), nn.GELU(), _norm(d // 2, norm_type))
        self.stage_s8 = nn.Sequential(
            nn.Conv2d(d // 2, d, 3, stride=2, padding=1), nn.GELU(), _norm(d, norm_type))
        if stride == 16:
            self.stage_s16 = nn.Sequential(
                nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU(), _norm(d, norm_type))

    def forward(self, x):
        s2 = self.stage_s2(x)
        s4 = self.stage_s4(s2)
        s8 = self.stage_s8(s4)
        skips = {2: s2, 4: s4, 8: s8}
        if self.stride == 16:
            s16 = self.stage_s16(s8)
            skips[16] = s16
            return s16, skips
        return s8, skips


# channels emitted at each output stride by _Backbone (for sizing the decoder fusions)
def _skip_channels(d):
    return {2: d // 4, 4: d // 2, 8: d, 16: d}


class _UpBlock(nn.Module):
    """Upsample 2x to the skip's spatial size, concat the (high-res) skip, fuse with a 3x3 conv. Using the
    skip's actual shape for the upsample target handles native (non-power-of-2) resolutions exactly."""

    def __init__(self, in_ch, skip_ch, out_ch, norm_type):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, stride=1, padding=1), nn.GELU(), _norm(out_ch, norm_type))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class _FineDecoder(nn.Module):
    """U-Net decoder: upsample the stride-{base} feature down to stride-{fine}, fusing the matching skips at
    each level. Output is a high-res feature map the fine SplatHead runs on -- so fine splats live on an
    H/fine grid (4x more cells than H/8 at fine=4) AND see restored high-frequency detail. out_ch reported
    for the head's in_ch wiring."""

    def __init__(self, d, base_stride, fine_stride, norm_type):
        super().__init__()
        if fine_stride not in (2, 4, 8) or fine_stride > base_stride:
            raise ValueError(f"fine_stride must be in (2,4,8) and <= stride, got {fine_stride}")
        skip_ch = _skip_channels(d)
        self.blocks = nn.ModuleList()
        self.target_strides = []
        s, in_ch = base_stride, d
        while s > fine_stride:
            tgt = s // 2
            out_ch = max(d // 2, 32)
            self.blocks.append(_UpBlock(in_ch, skip_ch[tgt], out_ch, norm_type))
            self.target_strides.append(tgt)
            in_ch, s = out_ch, tgt
        self.out_ch = in_ch                      # == d (identity) when fine_stride == base_stride

    def forward(self, feat, skips):
        x = feat
        for blk, tgt in zip(self.blocks, self.target_strides):
            x = blk(x, skips[tgt])
        return x


def sobel_gradmag(img: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> per-pixel gradient magnitude [B,1,H,W] (detail/edge signal)."""
    gray = img.mean(1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device) / 4.0
    ky = kx.t().contiguous()
    gx = F.conv2d(gray, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_pixel_indices(images: torch.Tensor, n_px: int, lam: float) -> torch.Tensor:
    """Sample n_px pixel indices per image ∝ (1-lam)*‖∇I‖ + lam*uniform (Image-GS P_opt). Concentrates
    the loss/refine on edges/text. images [B,3,H,W] -> [B,n_px] long. Shared by the amortized trainer,
    the POD refine, and the per-image fitter so all three optimize the SAME edge-weighted objective."""
    B, _, H, W = images.shape
    g = sobel_gradmag(images).float().reshape(B, -1)                # [B, H*W] fp32 PMF (multinomial)
    g = g / g.sum(1, keepdim=True).clamp_min(1e-8)
    pmf = (1 - lam) * g + lam / (H * W)
    return torch.multinomial(pmf, min(n_px, H * W), replacement=True)


class _SplatHead(nn.Module):
    """1x1 conv heads predicting M splats per feature cell from a (possibly fused) feature map.
    Produces mu (base grid + bounded free offset), log_scale, theta, color(sigmoid), opacity(sigmoid)."""

    def __init__(self, in_ch, M, color_dim=3, scale_bias=-3.0, opacity_bias=2.0,
                 offset_cells=1.0, use_rotation=True, min_opacity=0.0):
        super().__init__()
        self.M, self.color_dim, self.use_rotation = M, color_dim, use_rotation
        self.offset_cells = offset_cells
        # SCREEN-DOOR FIX: opacity FLOOR for the coarse base layer. Without it the coarse layer dies from the
        # RECON loss (not sparsity) in the first ~250 steps -- once the dense fine H/4 lattice can tile
        # everything, big blurry coarse splats only add haze under the sharp fine layer, so the optimizer
        # zeroes them -> the fine lattice owns the whole image -> screen door. Reparameterizing
        # opacity in [min_opacity, 1] forces coarse to PARTICIPATE; odds compositing lets dense fine OCCLUDE it
        # on edges while it fills the flats smoothly (no lattice). See SCREEN-DOOR-FIX.md.
        self.min_opacity = min_opacity
        self.mu_head = nn.Conv2d(in_ch, 2 * M, 1)
        self.scale_head = nn.Conv2d(in_ch, 2 * M, 1)
        self.color_head = nn.Conv2d(in_ch, color_dim * M, 1)
        self.opacity_head = nn.Conv2d(in_ch, M, 1)
        if use_rotation:
            self.theta_head = nn.Conv2d(in_ch, M, 1)
        nn.init.zeros_(self.mu_head.weight); nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.scale_head.bias, scale_bias)
        nn.init.constant_(self.opacity_head.bias, opacity_bias)

    def forward(self, feat):
        B, _, fH, fW = feat.shape
        M = self.M
        dev, dt = feat.device, feat.dtype
        ys = torch.linspace(0.5 / fH, 1 - 0.5 / fH, fH, device=dev, dtype=dt)
        xs = torch.linspace(0.5 / fW, 1 - 0.5 / fW, fW, device=dev, dtype=dt)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gy, gx], 0).unsqueeze(0).unsqueeze(0)            # [1,1,2,fH,fW]

        # FREE position offset: ± offset_cells cells (baseline was clamped to ±1 cell -> no migration)
        off_scale = self.offset_cells / max(fH, fW)
        mu_raw = self.mu_head(feat).reshape(B, M, 2, fH, fW)
        mu = base + torch.tanh(mu_raw) * off_scale
        mu = mu.permute(0, 3, 4, 1, 2).reshape(B, fH * fW * M, 2).clamp(0.005, 0.995)

        ls = self.scale_head(feat).reshape(B, M, 2, fH, fW).permute(0, 3, 4, 1, 2).reshape(B, fH * fW * M, 2)
        col = torch.sigmoid(self.color_head(feat).reshape(B, M, self.color_dim, fH, fW)
                            .permute(0, 3, 4, 1, 2).reshape(B, fH * fW * M, self.color_dim))
        op = torch.sigmoid(self.opacity_head(feat).reshape(B, M, fH, fW)
                           .permute(0, 2, 3, 1).reshape(B, fH * fW * M))
        if self.min_opacity > 0.0:                            # floor: op in [min_opacity, 1] (coarse-survival)
            op = self.min_opacity + (1.0 - self.min_opacity) * op
        if self.use_rotation:
            th = torch.tanh(self.theta_head(feat).reshape(B, M, fH, fW)
                            .permute(0, 2, 3, 1).reshape(B, fH * fW * M)) * (math.pi / 2)
        else:
            th = None
        return {"mu": mu, "log_scale": ls, "color": col, "opacity": op, "theta": th}


class EfficientGaussianEncoder(nn.Module):
    def __init__(self, d_feature=128, stride=8, variant="free", norm_type="group",
                 use_rotation=True, use_grad_input=True,
                 gaussians_per_cell=2,            # 'free': splats/cell ; 'adaptive': COARSE splats/cell
                 fine_per_cell=2,                 # 'adaptive': FINE splats/cell
                 free_offset_cells=4.0,           # 'free' position migration range (cells)
                 coarse_scale_bias=-1.5,          # 'adaptive' coarse: LARGE splats
                 coarse_min_scale=0.04,           # 'adaptive' FLOOR on coarse scale (force big base-layer
                                                  # splats that cover flats; else L1 shrinks them to dots)
                 fine_scale_bias=-4.0,            # 'adaptive' fine: small splats
                 free_scale_bias=-3.0,            # 'free' splats
                 opacity_bias=2.0,                # init opacity = sigmoid(bias); 2.0 -> 0.88 (saturated mush
                                                  # -> blur-basin catchment); 0.0 -> 0.5 (the fitter's value)
                 residual_grid=64,                # 'adaptive' coarse-render grid for the residual map
                 fine_stride=None,                # 'adaptive' fine head's grid (None -> = stride; 4 -> H/4
                                                  # via a U-Net skip decoder so it can SEE 2px strokes)
                 coarse_min_opacity=0.0,          # 'adaptive' COARSE opacity floor (op in [floor,1]) so the
                                                  # base layer can't die from recon -> screen-door fix
                 coarse_offset_cells=2.0,         # ALLOCATION-A0: how far a splat may migrate from its
                 fine_offset_cells=2.0,           # cell, in cells. Was HARDCODED at 2.0 for both adaptive
                                                  # heads. mu = base_grid + tanh(raw)*offset/max(fH,fW), so
                                                  # this is the ONLY route by which a splat can leave the
                                                  # patch that predicted it -- and the measured lattice
                                                  # (per-cell count CV 0.0000) has two candidate causes,
                                                  # patch-local mu and fixed slots per cell, which nobody
                                                  # has separated. Raising this isolates the first: if
                                                  # splats can migrate far and the count CV STILL reads
                                                  # 0.0000, migration is not the binding constraint and
                                                  # the fixed slot count is, which is the expensive fix.
                 odds=False):                     # F2: odds-weighted compositing (a/(1-a)) so committed
                                                  # splats OCCLUDE not average -- here for the internal coarse
                                                  # residual render so it matches the trainer's final render
        super().__init__()
        assert variant in ("free", "adaptive")
        assert stride in (8, 16), "efficient encoder supports stride 8 or 16 (16 = scarce)"
        self.variant = variant
        self.stride = stride
        self.fine_stride = fine_stride if fine_stride is not None else stride
        self.odds = odds
        self.use_grad_input = use_grad_input
        self.residual_grid = residual_grid
        # softplus_inv(coarse_min_scale): clamp coarse log_scale >= this so softplus(ls) >= coarse_min_scale
        self.coarse_ls_floor = math.log(math.exp(coarse_min_scale) - 1.0) if coarse_min_scale > 0 else None
        self.M = gaussians_per_cell
        self.Mf = fine_per_cell

        in_ch = 3 + (1 if use_grad_input else 0)              # input gets the gradient map too
        self.backbone = _Backbone(in_ch, d_feature, norm_type, stride)
        self.bg_head = nn.Linear(d_feature, 3)                # per-image learned background

        # fuse a downsampled gradient map into the head input (content signal for allocation)
        g_ch = 1 if use_grad_input else 0
        head_in = d_feature + g_ch
        if variant == "free":
            self.head = _SplatHead(head_in, self.M, scale_bias=free_scale_bias, opacity_bias=opacity_bias,
                                   offset_cells=free_offset_cells, use_rotation=use_rotation)
        else:
            self.coarse_head = _SplatHead(head_in, self.M, scale_bias=coarse_scale_bias,
                                          opacity_bias=opacity_bias, offset_cells=coarse_offset_cells,
                                          use_rotation=use_rotation, min_opacity=coarse_min_opacity)
            # fine path: U-Net decoder lifts features to fine_stride, then the head reads
            # (decoded feat + grad@fine + residual/coarse cond@fine). 4 extra channels = resid(3)+coarse(1).
            self.fine_decoder = _FineDecoder(d_feature, stride, self.fine_stride, norm_type)
            self.fine_head = _SplatHead(self.fine_decoder.out_ch + g_ch + 4, self.Mf,
                                        scale_bias=fine_scale_bias, opacity_bias=opacity_bias,
                                        offset_cells=fine_offset_cells, use_rotation=use_rotation)

    def _bg(self, feat):
        return torch.sigmoid(self.bg_head(feat.mean(dim=(2, 3))))           # [B,3]

    def gather_splat_latents(self, output):
        """Per-splat latent [B, N, C] aligned 1:1 with the splat axis of output['mu'].

        _SplatHead flattens as `permute(0,3,4,1,2).reshape(B, fH*fW*M, ...)`, i.e. index =
        (y*fW + x)*M + m -- CELL-major, then slot. Getting that backwards would silently pair every
        splat with the wrong latent and still produce plausible-looking tensors, so the convention is
        pinned by a test rather than trusted to this comment.

        In 'adaptive' the coarse splats have no entry in the fine map, so they are returned as ZEROS
        to keep the splat axis aligned; a caller indexing the full set cannot silently misalign. (They
        are dead in the golden checkpoint anyway -- mean opacity 0.0000.) Requires forward(...,
        return_latents=True).
        """
        if "splat_features" not in output:
            raise KeyError("no 'splat_features' -- call forward(image, return_latents=True)")
        feat = output["splat_features"]                                     # [B,C,fH,fW]
        B, C, fH, fW = feat.shape
        per_cell = self.Mf if self.variant == "adaptive" else self.M
        latents = feat.permute(0, 2, 3, 1).reshape(B, fH * fW, 1, C)
        latents = latents.expand(B, fH * fW, per_cell, C).reshape(B, fH * fW * per_cell, C)
        n_coarse = int(output.get("n_coarse", 0))
        if n_coarse:
            pad = latents.new_zeros(B, n_coarse, C)
            latents = torch.cat([pad, latents], dim=1)
        if latents.shape[1] != output["mu"].shape[1]:
            raise RuntimeError(f"latent/splat axis mismatch: {latents.shape[1]} vs {output['mu'].shape[1]}")
        return latents

    def _grid_to_full(self, feat, image):
        """Backbone features + (optional) downsampled gradient map concatenated for the heads."""
        if not self.use_grad_input:
            return feat
        g = sobel_gradmag(image)                                            # [B,1,H,W]
        g = F.adaptive_avg_pool2d(g, feat.shape[2:])                        # [B,1,fH,fW]
        return torch.cat([feat, g], dim=1)

    def forward(self, image, return_latents: bool = False):
        """return_latents=True additionally emits `splat_features` -- the feature map each splat was
        predicted FROM. This is the encoder's per-splat LATENT, and it matters because the head is a 1x1
        conv: it compresses its input channels down to 9 numbers per splat, discarding most of what the
        backbone knows (a linear probe on these features reads face attributes at ~90%). Downstream
        consumers have only ever seen the 9-dim side of that bottleneck.

        'adaptive' -> the FINE decoder map [B, max(d/2,32), H/fine_stride, W/fine_stride]; the coarse
        splats have no entry because in the golden checkpoint that layer is 100% dead. 'free' -> the
        fused backbone map the single head reads. Off by default: holding this through a training step
        is real memory for no benefit to the render path, so inference consumers opt in.

        `backbone_features` ([B, d, H/8, W/8]) is returned unconditionally -- pre-existing contract,
        consumed by gaussian_jepa/venc/model.py."""
        B = image.shape[0]
        if self.use_grad_input:
            grad = sobel_gradmag(image)
            x = torch.cat([image, grad], dim=1)
        else:
            x = image
        feat, skips = self.backbone(x)                                     # feat [B,D,fH,fW] + s2/s4 skips
        fused = self._grid_to_full(feat, image)
        bg = self._bg(feat)

        if self.variant == "free":
            g = self.head(fused)
            g["bg"] = bg
            g["n_gaussians"] = g["mu"].shape[1]
            g["backbone_features"] = feat
            if return_latents:
                g["splat_features"] = fused      # 'free': the single head reads `fused` directly
            return g

        # --- adaptive: coarse -> residual -> fine ---
        coarse = self.coarse_head(fused)
        if self.coarse_ls_floor is not None:          # force coarse to be the LARGE base layer (covers
            coarse["log_scale"] = coarse["log_scale"].clamp(min=self.coarse_ls_floor)  # flats; no dots)
        rg = self.residual_grid
        # Octave-LOD render (memory- AND compute-bounded: pairs <=36*n_coarse) so residual_grid can go high
        # without the exact path's O(rg^2 * n_coarse) cost.
        c_render = render_alpha_octave(coarse["mu"], coarse["log_scale"], coarse["color"],
                                       coarse["opacity"], bg, rg, rg, coarse["theta"],
                                       odds=self.odds)                                   # [B,3,rg,rg]
        target_small = F.interpolate(image, size=(rg, rg), mode="bilinear", align_corners=False)
        resid = (target_small - c_render).abs()                            # [B,3,rg,rg]
        cond = torch.cat([resid, c_render.mean(1, keepdim=True)], dim=1)   # [B,4,rg,rg] residual+coarse

        # Fine path at fine_stride: U-Net decoder restores high-freq, residual is routed at FINE resolution
        # (interpolated to the fine grid, NOT pooled down to s8 -- so the fine head sees text-scale error at
        # text scale, the whole point of v7). grad map also at fine resolution.
        fine_feat = self.fine_decoder(feat, skips)                         # [B, out_ch, fHf, fWf]
        ffH, ffW = fine_feat.shape[2:]
        cond_fine = F.interpolate(cond, size=(ffH, ffW), mode="bilinear", align_corners=False)
        if self.use_grad_input:
            g_fine = F.adaptive_avg_pool2d(sobel_gradmag(image), (ffH, ffW))
            fine_in = torch.cat([fine_feat, g_fine, cond_fine], dim=1)
        else:
            fine_in = torch.cat([fine_feat, cond_fine], dim=1)
        fine = self.fine_head(fine_in)

        out = {k: torch.cat([coarse[k], fine[k]], dim=1)
               for k in ("mu", "log_scale", "color", "opacity")}
        out["theta"] = (torch.cat([coarse["theta"], fine["theta"]], dim=1)
                        if coarse["theta"] is not None else None)
        out["bg"] = bg
        out["n_gaussians"] = out["mu"].shape[1]
        out["n_coarse"] = coarse["mu"].shape[1]
        out["backbone_features"] = feat
        if return_latents:
            # Splats are ordered [coarse | fine] along dim=1, so fine splat j = i - n_coarse reads cell
            # (j // Mf) of this map, row-major over (fHf, fWf). gather_splat_latents() does that.
            out["splat_features"] = fine_feat
        return out
