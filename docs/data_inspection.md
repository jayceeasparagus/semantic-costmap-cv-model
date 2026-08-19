# Dataset Inspection

## Dataset

Initial inspection dataset: RELLIS-3D image annotation examples.

This sample is being used to learn the RGB image and semantic-label
pipeline. The final project dataset may use KITTI-360.

## Image and label structure

- RGB images: JPEG
- Semantic labels: single-channel PNG
- Image resolution: 1920 x 1200
- RGB and label filenames match by filename stem
- Labels contain numeric semantic class IDs

## Observed class IDs

The sample contains the following IDs:

- 3
- 4
- 7
- 8
- 9
- 18
- 19
- 31
- 33

Not every image contains every class.

## Initial observations

Classes 3, 4, 7, and 19 occupy most pixels in the sample.
Class 33 is relatively rare.
Classes 8, 9, 18, and 31 are very rare in this small sample.

This indicates class imbalance may affect model training.

## Important caution

The numeric IDs should not be assigned semantic meanings manually.
The official RELLIS-3D ontology and label mapping should be used.