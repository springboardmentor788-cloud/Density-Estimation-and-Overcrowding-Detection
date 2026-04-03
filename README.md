# DeepVision Crowd Monitor

Real-time deep learning system for crowd density estimation and overcrowding detection from surveillance video, using CSRNet on PyTorch.

---

## Milestone 1: Setup and Data Preparation (Weeks 1–2)

### Things
- Development environment (Python, PyTorch, OpenCV, etc.)
- ShanghaiTech dataset downloaded and preprocessed (resize, normalization)
- Data loading and visualization scripts

### Evaluation
- All libraries install and import successfully
- Load, preprocess, and display samples without errors
- Setup and data preparation documented below

### Run (Milestone 1)
1. **Environment:**  
   `python -m venv venv` → `venv\Scripts\activate` → `pip install -r requirements.txt`
2. **Dataset:** Place ShanghaiTech under `data/ShanghaiTech` (see structure in repo).
3. **Preprocess:**  
   `cd src` → `python preprocess.py`
4. **Visualize samples:**  
   `python visualize_samples.py`

---

## Milestone 2: Model Development and Training (Weeks 3–4)

### Things
- CSRNet implemented in PyTorch (`src/csrnet.py`)
- Training script and training on (a subset of) the dataset
- Density map generation and visualization from the trained model

### Evaluation
- Correct CSRNet architecture
- Training runs with loss convergence
- Qualitative density map visualization
- MAE on validation set

### Run (Milestone 2)

1. **Train (from project root, with venv active):**
   ```bash
   cd src
   python train.py --part A --epochs 50 --batch_size 4
   ```
   - Optional: `--subset 200` to use only 200 train images; `--lr 1e-5` (default); `--save_dir ../checkpoints`.
   - Best model is saved as `checkpoints/csrnet_best.pt` (lowest validation MAE).

2. **Visualize density maps (after training):**
   ```bash
   python visualize_density.py --checkpoint ../checkpoints/csrnet_best.pt --part A --split test --num 3
   ```
   - Saves `density_vis.png` in the project root (image | GT density | predicted density).

### Project layout (Milestone 2)

- `src/csrnet.py` – CSRNet model and `density_to_count`
- `src/density_utils.py` – GT density map generation from point annotations
- `src/dataset.py` – `ShanghaiTechDensityDataset` (image + density + count) and `ShanghaiTechImageDataset`
- `src/train.py` – training loop, MSE loss, validation MAE, checkpointing
- `src/visualize_density.py` – qualitative density map comparison
- `checkpoints/` – saved models (created on first run)

---

## Requirements

- `requirements.txt`: torch, torchvision, torchaudio, opencv-python, numpy, pillow, matplotlib, tqdm, scipy
