# Segmentation model card

## Model

- Architecture: U-Net trained from random initialization
- Parameters: 7,762,693
- Input: A2D2 front-center RGB image, resized to the training resolution
- Output: five per-pixel class logits
- Selected checkpoint: epoch 29

The five output classes are `drivable`, `non_drivable`, `static_obstacle`,
`dynamic_obstacle`, and `background`. Background includes sky and other visual
context that should not be projected into a physical costmap.

## Evaluation

| Metric | Value |
|---|---:|
| Validation navigation mIoU | 0.8456 |
| Test navigation mIoU | 0.7966 |
| Test all-class mIoU | 0.8314 |
| Test loss | 0.2609 |

| Test class | IoU |
|---|---:|
| drivable | 0.9439 |
| non_drivable | 0.8707 |
| static_obstacle | 0.6452 |
| dynamic_obstacle | 0.7265 |
| background | 0.9709 |

Navigation mIoU averages the four classes that enter the costmap and excludes
background. The held-out test result is the final reported estimate; it was not
used to select the checkpoint.

## Local artifact provenance

The checkpoint is intentionally excluded from Git. The verified local artifact
used by the demos is:

```text
outputs/checkpoints/epoch29_restore/best_semantic_unet.pt
size: 93,271,255 bytes
sha256: 6c0d9d807979008f06ed75681ba87105adebde9583fc66cc462374e1bbd20a9b
```

The checkpoint stores the epoch, model state, and validation score. Loading is
performed with an explicit CPU/GPU map location; inference uses `eval()` and
`torch.inference_mode()`.

## Appropriate use and limitations

The model was trained for an A2D2-style road domain and a navigation-oriented
reduced taxonomy. It should not be assumed to generalize to arbitrary cameras,
weather, terrain, or countries without evaluation. Small or rare obstacles can
be missed, and semantic confidence is not a formal safety guarantee. Raw LiDAR
obstacle precedence and normal Nav2 safety layers must remain enabled.

The training notebook is useful for future retraining but is not required to
run the saved checkpoint. Inference, fusion, and ROS integration are maintained
as normal source modules rather than notebook-only code.
