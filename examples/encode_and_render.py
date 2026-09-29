"""Encode images with the released checkpoint and render the predicted splats."""

import argparse
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_splat_encoder.efficient_encoder import EfficientGaussianEncoder
from gaussian_splat_encoder.render_triton import render_alpha_octave


def load_encoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["cfg"]["encoder"])
    accepted = set(inspect.signature(EfficientGaussianEncoder.__init__).parameters) - {"self"}
    unknown = set(config) - accepted - {"image_size"}
    if unknown:
        raise ValueError(f"Unknown checkpoint architecture fields: {sorted(unknown)}")
    encoder = EfficientGaussianEncoder(**{k: v for k, v in config.items() if k in accepted})
    encoder.load_state_dict(checkpoint["model"], strict=True)
    return encoder.to(device).eval()


def render(encoder, x):
    out = encoder(x)
    reconstruction = render_alpha_octave(
        out["mu"].float(), out["log_scale"].float(), out["color"].float(),
        out["opacity"].float(), out["bg"].float(), x.shape[-2], x.shape[-1],
        theta=out["theta"].float(), odds=encoder.odds,
    ).clamp(0, 1)
    return out, reconstruction


def as_image(x):
    return Image.fromarray((x.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/encoder_golden_d256.pt"))
    parser.add_argument("--out", type=Path, default=Path("samples/reconstructions"))
    parser.add_argument("--benchmark", action="store_true", help="CUDA events, model forward only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The released encoder's coarse render uses Triton and requires a CUDA GPU")
    device = torch.device("cuda")
    encoder = load_encoder(args.checkpoint, device)
    records = []
    with torch.inference_mode():
        for path in args.inputs:
            src = Image.open(path).convert("RGB")
            arr = np.asarray(src).copy()
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
            out, reconstruction = render(encoder, x)
            mse = float((x - reconstruction).square().mean().item())
            record = {"input": str(path), "width": src.width, "height": src.height,
                      "splats": int(out["mu"].shape[1]), "live_splats_opacity_gt_0.1":
                      int((out["opacity"] > 0.1).sum().item()),
                      "psnr_db": -10 * math.log10(max(mse, 1e-12))}
            recon_image = as_image(reconstruction[0])
            recon_image.save(args.out / f"{path.stem}_reconstruction.png")
            panel = Image.new("RGB", (src.width * 2, src.height))
            panel.paste(src, (0, 0))
            panel.paste(recon_image, (src.width, 0))
            panel.save(args.out / f"{path.stem}_panel.png")
            if args.benchmark:
                for _ in range(5):
                    encoder(x)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                timings = []
                for _ in range(30):
                    start.record()
                    encoder(x)
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
                record["model_ms_median"] = float(np.median(timings))
                record["model_ms_p10"] = float(np.quantile(timings, 0.1))
                record["model_ms_p90"] = float(np.quantile(timings, 0.9))
            print(json.dumps(record, sort_keys=True), flush=True)
            records.append(record)
    (args.out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
