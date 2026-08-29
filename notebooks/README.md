# Notebook workflow

The project uses two focused Colab notebooks. Repository modules contain the
reusable logic; notebooks orchestrate that logic and display results.

## 01_prepare_a2d2_data.ipynb

Purpose:

1. Clone or pull this repository.
2. Install the package with `pip install -e .`.
3. Download selected A2D2 sequences.
4. Convert source RGB labels with
   `configs/a2d2_semantic_mapping_v2.yaml`.
5. Create sequence-level train and validation manifests.
6. Verify sample pairs and class distributions visually.

Expected persistent artifacts:

- `data/processed/a2d2_v2/images/`
- `data/processed/a2d2_v2/masks/`
- `data/splits/a2d2_v2_train.jsonl`
- `data/splits/a2d2_v2_val.jsonl`
- a short dataset summary JSON

Split entire driving sequences, not neighboring frames, between training and
validation. This reduces temporal leakage.

## 02_train_semantic_model.ipynb

Purpose:

1. Clone or pull this repository.
2. Install the package with `pip install -e .`.
3. Restore the prepared dataset and manifests.
4. Construct `SemanticSegmentationUNet` with six output classes.
5. Train with class-aware loss and checkpoint the best validation mIoU.
6. Report all-class mIoU, navigation-only mIoU, per-class IoU, and latency.
7. Export predictions, confidence maps, history, and the best checkpoint.

Expected persistent artifacts:

- `outputs/checkpoints/best_semantic_model.pt`
- `outputs/metrics/training_history.json`
- `outputs/metrics/validation_metrics.json`
- `outputs/annotations/validation_predictions.png`

## Colab persistence rule

Colab storage is temporary. Before disconnecting, copy datasets and outputs to
Google Drive or download them. Checkpoints are intentionally ignored by Git;
source code, configuration, manifests, and small metrics files belong in Git.

After model training, return to WSL for LiDAR projection, costmap construction,
ROS 2 integration, Nav2 compatibility, tests, benchmarking, and Docker packaging.
