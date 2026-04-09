# DeepVision Crowd Monitor

Milestone 3 implementation for crowd counting and overcrowding detection using CSRNet.

## Features

- ShanghaiTech-style dataset loading for `part_A_final` and `part_B_final`
- Adaptive Gaussian density map generation with count-preserving kernels
- CSRNet crowd counting model in PyTorch
- Training and validation pipeline with MAE and RMSE reporting
- Image, video, and webcam inference with heatmap overlays
- Overcrowding alerts with a configurable threshold

## Dependencies

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train on Part A:

```bash
python -m training.train \
  --data-roots part_A_final \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --resize-height 384 \
  --resize-width 384 \
  --train-random-crop
```

Train on both Part A and Part B:

```bash
python -m training.train \
  --data-roots part_A_final part_B_final \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --resize-height 384 \
  --resize-width 384 \
  --train-random-crop
```

The best checkpoint is saved to `checkpoints/csrnet_best.pth`.

## Validation

```bash
python -m training.validate \
  --data-roots part_A_final \
  --checkpoint checkpoints/csrnet_best.pth \
  --save-examples-dir outputs/validation_examples
```

## Image Inference

Single image:

```bash
python -m inference.image_inference \
  --input test.jpg \
  --checkpoint checkpoints/csrnet_best.pth \
  --output-dir outputs/image_inference
```

Folder of images:

```bash
python -m inference.image_inference \
  --input path/to/images \
  --checkpoint checkpoints/csrnet_best.pth \
  --output-dir outputs/image_inference
```

## Video Inference

```bash
python -m inference.video_inference \
  --input input.mp4 \
  --output outputs/annotated.mp4 \
  --checkpoint checkpoints/csrnet_best.pth \
  --sample-fps 2 \
  --threshold 120
```

## Webcam Inference

```bash
python -m inference.realtime \
  --source 0 \
  --checkpoint checkpoints/csrnet_best.pth \
  --threshold 120
```

Press `q` to quit the webcam window.

## React Dashboard

The project now includes a modern React/Vite dashboard with a FastAPI inference backend.

Backend API:

```bash
python -m api.server
```

Frontend UI:

```bash
cd frontend
npm install
npm run dev
```

The React dashboard supports:

- Model selection from all trained checkpoints
- Image upload with overlay + density-only outputs
- Video upload with overlay + density-only videos
- Browser webcam sampling for live analysis
- Polished glass-style UI with embedded crowd count HUD

Streamlit remains available as a fallback dashboard in `dashboard/app.py`, but the React frontend is the new primary UI path.

## Modern Dashboard (Live + Upload)

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard capabilities:

- Live webcam monitoring with side-by-side outputs:
  - Overlay frame (video frame + density map)
  - Density-only frame
- Upload image and get both outputs saved and previewed.
- Upload video and generate two videos:
  - Overlay output video
  - Density-only output video

Default dashboard model paths are preconfigured to the strongest checkpoint and calibration:

- `checkpoints/high_accuracy_best_partA.pth`
- `outputs/count_calibration_partA_push_v1_mae.json`

## Notes

- The density map cache is stored in `cache/density_maps`.
- The model outputs a single-channel density map and the crowd count is the sum of that map.
- Video and webcam processing support frame sampling for faster inference.
