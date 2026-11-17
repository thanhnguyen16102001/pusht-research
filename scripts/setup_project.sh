
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Push-T Research - Project Setup
# ============================================================

BASE="$HOME/AI-application"
PROJECT="$BASE/pusht-research"

SOURCE="$BASE/pusht-work/data/pusht_demo.zarr"
TARGET="$PROJECT/data/demonstrations/pusht_demo.zarr"

echo "=== Push-T Research Setup ==="

# 1. Create project directories
mkdir -p \
    "$PROJECT/data/demonstrations" \
    "$PROJECT/data/processed" \
    "$PROJECT/src" \
    "$PROJECT/scripts" \
    "$PROJECT/configs" \
    "$PROJECT/checkpoints" \
    "$PROJECT/experiments" \
    "$PROJECT/results" \
    "$PROJECT/docs"

echo "[OK] Project directories created."

# 2. Create README.md without overwriting existing content
if [[ ! -e "$PROJECT/README.md" ]]; then

    cat > "$PROJECT/README.md" <<'README'
# Push-T Research

Personal research project for robot learning, Diffusion Policy, and World Models using the Push-T environment.

## Objectives

- Train and evaluate Diffusion Policy models.
- Study model architectures and learning algorithms.
- Develop and evaluate new World Model approaches.

## Project Structure

| Directory | Purpose |
|---|---|
| `data/demonstrations/` | Human demonstration datasets |
| `data/processed/` | Preprocessed training data |
| `src/` | Custom model and training source code |
| `scripts/` | Setup, visualization and utility scripts |
| `configs/` | Model and experiment configurations |
| `checkpoints/` | Trained model weights |
| `experiments/` | Experiment runs and logs |
| `results/` | Evaluation metrics and visualizations |
| `docs/` | Research notes and documentation |

## Dataset

- File: `data/demonstrations/pusht_demo.zarr`
- Collected manually using Pygame.
- Initial dataset: 146 successful episodes, 30,486 steps.
- Contents: `img`, `state`, `action`, `keypoint`, `n_contacts`.
- Not yet used to train a model.

## Environment

- Ubuntu 22.04 / WSL2
- Conda environment: `pusht-sim`
- Python 3.9.18

## Reference

Stanford Diffusion Policy:
https://github.com/real-stanford/diffusion_policy

The original repository is maintained separately at
`../diffusion_policy` and must not be modified.

All custom development, datasets, checkpoints, and results
belong to this project.
README

    echo "[OK] README.md created."
else
    echo "[SKIP] README.md already exists."
fi

# 3. Copy the demonstration dataset
if [[ ! -d "$SOURCE" ]]; then
    echo "[ERROR] Source dataset not found: $SOURCE" >&2
    exit 1
fi

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
    echo "[SKIP] Destination dataset already exists."
else
    echo "[INFO] Copying demonstration dataset..."
    cp -a "$SOURCE" "$TARGET"
    echo "[OK] Dataset copied."
fi

# 4. Verify copied dataset
echo "[INFO] Verifying dataset files..."

if diff -qr "$SOURCE" "$TARGET"; then
    echo "[OK] Source and destination are identical."
else
    echo "[ERROR] Dataset verification failed." >&2
    echo "[INFO] Original dataset remains unchanged."
    exit 1
fi

# 5. Show summary
echo ""
echo "=== Setup Complete ==="
echo "Project: $PROJECT"
echo "Dataset: $TARGET"

du -sh "$TARGET"

echo ""
echo "Original source remains unchanged."
