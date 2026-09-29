# Gaussian splat encoder

A frozen, feed-forward RGB encoder that predicts explicit 2D anisotropic Gaussian splats. A small CNN predicts position, scale, rotation, RGB, and opacity for each splat. The included Triton renderer reconstructs an image from that set. The checkpoint has 833,107 parameters.

At 768 × 768, the encoder emits 92,160 candidates. About 73,000 have opacity above 0.1 in the examples below. The checkpoint is [`encoder_golden_d256.pt`](weights/encoder_golden_d256.pt), SHA-256 `ff7ebf0167173bb17c9d84317cf8eb33e2fdd050bca90dcac366b9e98992da2f`.

## Results

The archived evaluation on 64 held-out golden-pair images at 768 × 768 reported **33.41 dB PSNR**, **0.938 SSIM**, and **0.708 edge IoU**. The three new test cards below are original synthetic images. They measure this exact packaged checkpoint on an RTX 3070. They are not a photo benchmark.

| Input and reconstruction | PSNR | Model forward, median |
|---|---:|---:|
| [Geometry](samples/reconstructions/geometry_panel.png) | 31.02 dB | 11.66 ms |
| [Gradient](samples/reconstructions/gradient_panel.png) | 33.68 dB | 11.68 ms |
| [Poster and small text](samples/reconstructions/poster_panel.png) | 28.44 dB | 11.77 ms |

Each panel shows input on the left and reconstruction on the right. Timing uses CUDA events, 5 warmups and 30 runs per image. It includes the encoder's coarse render. It excludes image loading and the final full-resolution render. Small text is visibly softer in the reconstruction. See [measurements.json](samples/reconstructions/measurements.json) and [sample generator](examples/make_samples.py).

Published methods such as [AIR](https://arxiv.org/abs/2605.20820) and [Fast 2DGS](https://arxiv.org/abs/2512.12774) use different images, splat counts, and timing boundaries. These results do not establish a speed ranking.

## Run

Use a CUDA GPU. The tested environment used Python 3.14, PyTorch 2.12.0+cu130, Triton 3.7.0, NumPy 2.4.6, and Pillow 12.2.0.

```sh
uv venv --python 3.14
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0
uv pip install triton==3.7.0 numpy==2.4.6 pillow==12.2.0
.venv/bin/python examples/encode_and_render.py my_image.png --out outputs
```

The command saves a reconstruction PNG, a side-by-side panel, and PSNR in `outputs/measurements.json`. Input stays at its native resolution. Use only checkpoints you trust, since PyTorch loads this checkpoint with pickle.

Source code is Apache-2.0. Original sample images are CC BY 4.0. The checkpoint's redistribution and downstream license status are not established by this review. See [provenance](PROVENANCE.md) for the training and sample record.
