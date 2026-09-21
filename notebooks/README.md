# Notebooks

The A2D2 training and evaluation notebook produced the epoch-29 U-Net
checkpoint described in `docs/model_card.md`. It is useful for retraining or
reproducing model metrics, but it is not required for the offline mapping demo.

Stable inference, fusion, costmap, and mapping logic lives in normal source
files under `src/semantic_costmap/` so it can be tested and reused without a
notebook runtime. Notebook outputs, datasets, and model checkpoints are not
committed to Git.
