# Notebooks

`01_train_segmentation.ipynb` contains the A2D2 training and evaluation
workflow used to produce the epoch-29 U-Net checkpoint.

The notebook is useful for retraining or reproducing model metrics, but it is
not required for normal inference. Stable inference and fusion logic belongs in
`src/semantic_costmap/` so local tools and ROS 2 nodes can share it.

Notebook outputs and model checkpoints are not committed to Git.
