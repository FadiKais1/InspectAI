# InspectAI v1.0 — final comparison

Evaluated on **MVTec AD transistor** as a portfolio / research prototype.
Fitting used train/good only. Official test metrics are locked.

## Master results

| Method | Image AUROC | Pixel AUROC |
| --- | ---: | ---: |
| M5 Convolutional Autoencoder | 0.6158 | 0.5898 |
| M4 Frozen ResNet18 patch-kNN | 0.9800 | 0.7866 |
| M6 PatchCore-style ResNet18 | 0.9479 | 0.7689 |
| M7 Frozen DINOv3 ViT-S/16 | 0.9646 | 0.9663 |

DINOv3 per-defect image AUROC (10 defectives each; do not over-interpret):

| Defect | Image AUROC |
| --- | ---: |
| bent_lead | 0.975 |
| cut_lead | 0.8967 |
| damaged_case | 1.000 |
| misplaced | 0.9867 |

## Methods (short)

- **Autoencoder:** reconstruct normal images; score reconstruction error.
- **ResNet18 patch-kNN:** frozen ImageNet `layer3` patches vs a random
  50k normal memory; k=1 L2; spatial max.
- **PatchCore-style:** frozen `layer2`+`layer3`, 50k pool then 10k greedy
  coreset.
- **DINOv3:** frozen ViT-S/16 dense patches, 50k pool then 10k deterministic
  random memory; k=1 L2; spatial max.

## Main findings

1. Low Autoencoder reconstruction loss did not yield useful AUROC.
2. Frozen pretrained patch features far outperformed reconstruction.
3. PatchCore-style complexity did not beat simple ResNet patch-kNN here.
4. DINOv3 produced much sharper localization (pixel AUROC 0.9663 vs 0.7866).
5. Representation quality drove localization more than coreset machinery.
6. Spatial-max image scores can still be sensitive to isolated unusual
   patches (post-hoc; seen especially with subtle `cut_lead`).

Do **not** claim DINOv3 is best at image AUROC. M4 remains highest
(0.9800 vs 0.9646).

## Qualitative conclusions

DINOv3 maps for bent-lead, damaged-case, and misplaced samples tracked
defect regions more tightly than the ResNet baselines. Some cut-lead
images localized spatially but ranked below high-response normal patches
under spatial max. Overlays are local-only (MVTec images are not
redistributed in this repo).

## Selected model for the Inspector / demo

**DINOv3 ViT-S/16 patch memory.**

Reason: not the highest image AUROC, but excellent image-level performance
**plus** dramatically superior pixel localization — the best overall
detection/localization balance among the locked methods.

The Inspector loads a **saved** 10k-vector memory artifact. It does not
rebuild train/good memory at inference time. It does not emit
NORMAL/DEFECTIVE unless a caller supplies a threshold. No deployment
threshold was estimated from official test labels.

## Limitations

- One MVTec category; small official test set
- No independently calibrated decision threshold
- CPU latency on the order of 1–2 s/image in development
- Exact 1-NN over 10k vectors
- No domain-shift / line-deployment study
