# DeepVision Crowd Monitor (Milestone 3)

Production-grade crowd density estimation and overcrowding detection system built on CSRNet with a modular PyTorch architecture.

## Highlights

- Adaptive kNN Gaussian density maps from ShanghaiTech .mat annotations.
- Strict count preservation in density generation and downsampling.
- Density map caching for faster training and validation.
- CSRNet (primary) and MCNN (optional swap-in) model support.
- Stable training: AMP, gradient clipping, ReduceLROnPlateau, checkpointing.
- Metrics: MAE (primary), RMSE, and R2.
- Real-time pipeline: webcam/video, heatmap overlay, alerts, and FPS display.

## Project Structure

```
project/
├── data/
├── models/
│   ├── csrnet.py
│   └── mcnn.py
├── dataset/
│   ├── build_cache.py
│   ├── density_map.py
│   └── loader.py
├── training/
│   ├── evaluate.py
│   ├── train.py
│   └── validate.py
├── inference/
│   ├── benchmark.py
│   ├── realtime.py
│   └── utils.py
├── utils/
│   ├── metrics.py
│   └── visualization.py
├── config.py
└── requirements.txt
```

## Dataset

Expected layout in workspace root:

```
part_A_final/
part_B_final/
```

Each part must contain:

- train_data/images
- train_data/ground_truth
- test_data/images
- test_data/ground_truth

Ground-truth files must be .mat files in ShanghaiTech format.

## Setup

From workspace root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r project/requirements.txt
```

## 1) Build Density Cache (Recommended)

```bash
python -m project.dataset.build_cache \
	--dataset-root . \
	--parts A B \
	--splits train test \
	--cache-dir project/data/cache \
	--max-dim 1536 \
	--output-stride 8
```

## 2) Train

Part A:

```bash
python -m project.training.train \
	--dataset-root . \
	--part A \
	--model csrnet \
	--epochs 300 \
	--batch-size 4 \
	--workers 8 \
	--lr 1e-5 \
	--amp \
	--cache-dir project/data/cache \
	--work-dir project/data/runs \
	--exp-name csrnet_partA
```

Part B:

```bash
python -m project.training.train \
	--dataset-root . \
	--part B \
	--model csrnet \
	--epochs 300 \
	--batch-size 8 \
	--workers 8 \
	--lr 1e-5 \
	--amp \
	--cache-dir project/data/cache \
	--work-dir project/data/runs \
	--exp-name csrnet_partB
```

Training outputs:

- checkpoints: `project/data/runs/<exp-name>/checkpoints/{last.pt,best.pt}`
- logs: `project/data/runs/<exp-name>/metrics.csv`
- curves: `project/data/runs/<exp-name>/{loss_curve.png,metrics_curve.png,r2_curve.png}`

## 3) Evaluate

```bash
python -m project.training.evaluate \
	--dataset-root . \
	--part A \
	--split test \
	--model csrnet \
	--checkpoint project/data/runs/csrnet_partA/checkpoints/best.pt \
	--batch-size 2 \
	--workers 4 \
	--cache-dir project/data/cache \
	--vis-dir project/data/eval_vis/partA \
	--report-dir project/data/eval_reports
```

Evaluation artifacts are automatically exported to:

- `project/data/eval_reports/<run_name>/summary.json`
- `project/data/eval_reports/<run_name>/per_image_metrics.csv`
- `project/data/eval_reports/<run_name>/counts_scatter_gt_vs_pred.png`
- `project/data/eval_reports/<run_name>/counts_residual_hist.png`
- `project/data/eval_reports/<run_name>/counts_evaluation_matrix.png`
- `project/data/eval_reports/<run_name>/metrics_panel.png`
- qualitative detailed comparison cards in `--vis-dir`:
	- Original image
	- Ground-truth density map
	- Predicted density map
	- Embedded stats: GT count, predicted count, error, MAE, MSE

## 4) Real-Time Crowd Monitoring (Milestone 3)

Webcam:

```bash
python -m project.inference.realtime \
	--model csrnet \
	--checkpoint project/data/runs/csrnet_partA/checkpoints/best.pt \
	--source 0 \
	--max-dim 1280 \
	--use-fp16 \
	--global-count-threshold 120 \
	--region-grid 4 \
	--region-density-threshold 0.0025
```

Video file:

```bash
python -m project.inference.realtime \
	--model csrnet \
	--checkpoint project/data/runs/csrnet_partA/checkpoints/best.pt \
	--source input_video.mp4 \
	--output project/data/output/annotated.mp4 \
	--max-dim 1280 \
	--use-fp16
```

Press `q` to stop.

## 5) FPS Measurement

```bash
python -m project.inference.benchmark \
	--model csrnet \
	--checkpoint project/data/runs/csrnet_partA/checkpoints/best.pt \
	--image part_A_final/test_data/images/IMG_1.jpg \
	--max-dim 1280 \
	--iters 200 \
	--warmup 30 \
	--fp16
```

## Overcrowding Logic

Per frame workflow:

1. Video/Webcam frame input
2. Resize with max-dimension constraint
3. Normalize (ImageNet stats)
4. CSRNet inference
5. Density map to count (`sum(density)`)
6. Overcrowding check:
	 - global count threshold
	 - region-based density threshold
7. Alert output:
	 - console alert
	 - visual status (`SAFE` / `OVERCROWDED`) and red border when triggered

## Expected Performance Targets

- ShanghaiTech Part A MAE: ~65 to 85 (well-trained CSRNet)
- ShanghaiTech Part B MAE: ~8 to 15 (well-trained CSRNet)
- Real-time inference: >= 10 FPS on CUDA GPU (input-size and hardware dependent)

## Notes

- Use `--max-dim` to balance speed and accuracy.
- Caching adaptive density maps is critical for stable training throughput.
- If GPU memory is tight, reduce `--batch-size` and `--crop-size`.
- Repository upload checklist: see `GITHUB_PREP.md`.
