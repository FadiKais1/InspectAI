# Milestone 7 — Frozen DINOv3 ViT-S/16 Patch Memory

## Hypothesis

Frozen DINOv3 patch embeddings may provide richer local and structural
representations than supervised ImageNet ResNet18 features, potentially
improving pixel-level localization while maintaining strong image-level
anomaly detection.

## Protocol

Dataset: MVTec AD transistor
Training / fitting: 213 train/good images only
Test data: official 100-image test split (used once, after all choices
were locked)

Backbone: official frozen DINOv3 ViT-S/16
(`facebook/dinov3-vits16-pretrain-lvd1689m`)
No fine-tuning. Classification token is not used for scoring.

Input resolution: 512×512 (divisible by patch size 16)
Patch grid: 32×32
Embedding dimension: 384
Register tokens: 4 (dropped before scoring)

Memory:
- candidate pool size: 50,000
- final memory size: 10,000
- seed: 42
- selection: deterministic random sampling (not a PatchCore coreset)

Scoring:
- k = 1 nearest-neighbor L2 distance in patch embedding space
- image score: spatial maximum of the patch anomaly map
- localization: bilinear upsample of the patch map to native resolution

All configuration choices were fixed before observing official test-set
results. The official test split was not used for model selection,
hyperparameter selection, threshold tuning, or architecture changes.

## Results

Overall image AUROC: 0.9646
Pixel AUROC: 0.9663

Per-defect image AUROC:
- bent_lead: 0.975
- cut_lead: 0.8967
- damaged_case: 1.000
- misplaced: 0.9867

## Comparison with locked baselines

These numbers are recorded for comparison only. M4, M5, and M6 were not
modified or retuned.

| Experiment | Image AUROC | Pixel AUROC |
| --- | --- | --- |
| M5 Convolutional Autoencoder | 0.6158 | 0.5898 |
| M4 Frozen ResNet18 patch-kNN | 0.9800 | 0.7866 |
| M6 PatchCore-style ResNet18 | 0.9479 | 0.7689 |
| M7 Frozen DINOv3 ViT-S/16 | 0.9646 | 0.9663 |

DINOv3 image AUROC remained slightly below M4 (0.9646 vs 0.9800).
Pixel AUROC improved dramatically relative to the ResNet patch-kNN
baseline (0.9663 vs 0.7866).

## Qualitative findings

DINOv3 produced substantially sharper localization than the ResNet
baselines. Anomaly maps for bent-lead, damaged-case, and misplaced
samples closely followed the actual defect regions rather than spreading
broadly around them.

The main remaining weakness was subtle cut-lead image ranking
(per-defect image AUROC 0.8967). Some cut-lead images were spatially
localized correctly but received lower spatial-max image scores than
isolated high-response patches in some normal images.

This suggests that DINOv3 representation quality significantly improved
localization, while the simple spatial-max image aggregation rule can
still be sensitive to isolated unusual patches.

These observations are post-hoc interpretation only. Spatial-max scoring
will not be changed after seeing the official test data.

## Future work

Alternative image aggregation such as top-k or percentile scoring could
be studied in a later, separately registered experiment if the goal is
to reduce sensitivity to isolated high-response patches. That change
must not be applied retroactively to this locked M7 result.
