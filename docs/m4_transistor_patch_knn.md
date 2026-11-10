# Milestone 4 — Frozen Patch-kNN Baseline

## Setup

Dataset: MVTec AD transistor
Training data: 213 train/good images only
Test data: official 100-image test split

Backbone: ImageNet-pretrained ResNet18 (frozen)
Feature layer: layer3
Input resolution: 512x512
k: 1
Memory cap: 50,000 patches
Memory sampling seed: 42
Image score: spatial maximum

All configuration choices were fixed before observing official test-set
results.

## Results

Overall image AUROC: 0.980
Pixel AUROC: 0.7866

Per-defect image AUROC:
- bent_lead: 0.990
- cut_lead: 0.970
- damaged_case: 0.9933
- misplaced: 0.9667

## Qualitative Findings

The frozen patch-feature representation provides very strong image-level
anomaly discrimination, even though the model was fitted only on normal
training images.

Bent-lead and damaged-case examples generally produce high anomaly
responses around the physically abnormal regions. Cut-lead defects are
also detected, although anomaly maps tend to be spatially broader than
the corresponding ground-truth masks.

Pixel localization is substantially weaker than image-level detection.
The layer3 anomaly maps often spread beyond the true defect boundary,
suggesting that the deeper CNN representation provides strong semantic
features but relatively coarse spatial localization.

High-scoring normal samples reveal another failure mode: background
texture, lighting variation, component edges, and normal appearance
variation can receive high anomaly responses.

A particularly interesting limitation occurs for misplaced objects.
The current nearest-neighbor memory is appearance-based and does not
explicitly encode spatial position. Therefore, background appearing in
a location where a transistor would normally be expected may still
match normal background patches observed elsewhere in the training
set. This allows structurally abnormal images to contain locally
normal-looking patches.

The baseline therefore models local visual normality well but does not
fully represent spatial or global structural normality.

These observations are recorded as post-hoc failure analysis only.
The official test results will not be used to tune the baseline.
