# GitHub Upload Checklist

This repo is prepared for GitHub publication with reproducible training and evaluation outputs.

## Included Engineering Assets

- Training metrics CSV and curves (loss, MAE/RMSE, R2)
- Evaluation report artifacts:
  - per-image metrics CSV
  - summary JSON
  - GT vs prediction scatter
  - residual histogram
  - evaluation matrix (2D count histogram)
  - density map qualitative comparisons

## Pre-Upload Steps

1. Keep dataset local only (already ignored by `.gitignore`).
2. Run training and evaluation commands from `project/README.md`.
3. Commit source code + docs only.
4. Optionally publish report images by removing media ignore lines from `.gitignore`.

## Suggested Repository Structure

- `project/` source code
- `project/README.md` usage and benchmarks
- `GITHUB_PREP.md` publish notes
- `.gitignore` dataset/runtime exclusions

## Notes

- Current evaluation script writes all artifacts under `project/data/eval_reports/<run_name>/`.
- For strict reproducibility in CI, pin package versions in `project/requirements.txt` if needed.
