# Milestone 6 — PatchCore-style ResNet18

## Hypothesis

A multi-scale PatchCore-style representation using intermediate frozen CNN
features and a representative coreset memory would preserve local detail
better than the single-layer random-memory patch-kNN baseline (M4),
potentially improving pixel-level localization while maintaining strong
image-level detection.

**This hypothesis was not supported** on MVTec transistor under the locked
protocol below.

## Protocol

Dataset: MVTec AD transistor
Fitting: 213 train/good images only
Official test: 100-image split, used only for final evaluation

Backbone: ImageNet-pretrained ResNet18 (frozen)
Feature layers: `layer2` + `layer3` (aligned, concatenated)
Input resolution: 512×512
k: 1
Image score: spatial maximum

Memory:

- raw candidates: all fused patches (~872k)
- candidate pool: 50,000 (deterministic random, seed 42)
- final memory: 10,000 via greedy farthest-point sampling in a 128-D
  Gaussian random projection
- coreset note: full greedy FPS on all raw patches was infeasible on local
  CPU. The 50k pre-pool was a **pre-evaluation computational amendment**,
  recorded before any official test metrics.

The official test split was not used for architecture changes, memory-size
tuning, or threshold selection.

## Results

Overall image AUROC: 0.9479
Pixel AUROC: 0.7689

Per-defect image AUROC (official test; small n=10 defectives each):

- bent_lead: 0.9883
- cut_lead: 0.9417
- damaged_case: 1.000
- misplaced: 0.8617

## Comparison with locked M4

Frozen ResNet18 patch-kNN (M4): image AUROC 0.9800, pixel AUROC 0.7866.

PatchCore-style (M6): image AUROC 0.9479, pixel AUROC 0.7689.

M6 did **not** improve image-level detection or pixel localization relative
to the simpler layer3 random-memory baseline on this category.

These observations are post-hoc. M6 was not retuned after seeing test
metrics.
