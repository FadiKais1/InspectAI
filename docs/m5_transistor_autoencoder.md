# Milestone 5 — Convolutional Autoencoder Baseline

## Hypothesis

A convolutional autoencoder trained only on normal transistor images
should reconstruct normal structures better than anomalous structures,
allowing reconstruction error to serve as an anomaly score.

It was hypothesized before evaluation that this reconstruction baseline
could perform worse than pretrained patch-feature anomaly detection
because an autoencoder may also reconstruct anomalous structures.

## Protocol

Dataset: MVTec AD transistor

Training:
170 train/good images

Validation:
43 train/good images

Official test:
100 images

Input resolution: 256x256
Loss: MSE
Optimizer: Adam
Learning rate: 1e-3
Batch size: 8
Maximum epochs: 30
Seed: 42
Image anomaly score: spatial maximum reconstruction error

Best epoch: 30
Best validation loss: 0.000867

The official test set was not used for model training, checkpoint
selection, hyperparameter selection, or threshold tuning.

## Results

Overall image AUROC: 0.6158
Pixel AUROC: 0.5898

Per-defect image AUROC:

- bent_lead: 0.675
- cut_lead: 0.615
- damaged_case: 0.5133
- misplaced: 0.660

## Comparison with Milestone 4

Frozen ResNet patch-kNN:
Image AUROC: 0.980
Pixel AUROC: 0.7866

Convolutional Autoencoder:
Image AUROC: 0.6158
Pixel AUROC: 0.5898

The frozen pretrained patch representation substantially outperformed
the reconstruction-based Autoencoder at both image-level detection and
pixel-level localization.

A low normal-validation reconstruction loss therefore did not imply
strong anomaly discrimination.

One likely explanation is that the Autoencoder learned to reconstruct
both normal appearance and some anomalous structures sufficiently well,
causing normal and defective reconstruction-error distributions to
overlap.

The 256x256 input resolution may also suppress fine defect detail, but
this was part of the pre-registered baseline and was not modified after
observing the test results.

These observations are post-hoc analysis only. The baseline will not
be retuned using official test performance.
