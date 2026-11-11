# InspectAI documentation

Portfolio / research prototype for industrial visual anomaly detection,
evaluated on MVTec AD transistor. Metrics below are **locked**. They were
not retuned after official test evaluation.

## Index

| Document | Contents |
| --- | --- |
| [final_results.md](final_results.md) | Master comparison, selected-model rationale, limitations |
| [m4_transistor_patch_knn.md](m4_transistor_patch_knn.md) | Frozen ResNet18 patch-kNN (highest image AUROC) |
| [m5_transistor_autoencoder.md](m5_transistor_autoencoder.md) | Convolutional Autoencoder baseline |
| [m6_transistor_patchcore.md](m6_transistor_patchcore.md) | PatchCore-style multi-scale coreset (hypothesis not supported) |
| [m7_transistor_dinov3.md](m7_transistor_dinov3.md) | Frozen DINOv3 ViT-S/16 (selected Inspector backbone) |
| [release_checklist.md](release_checklist.md) | v1.0 release checks |
| [assets/README.md](assets/README.md) | Why qualitative photos are not committed |

Locked headline numbers:

| Method | Image AUROC | Pixel AUROC |
| --- | ---: | ---: |
| Autoencoder | 0.6158 | 0.5898 |
| ResNet18 patch-kNN | 0.9800 | 0.7866 |
| PatchCore-style | 0.9479 | 0.7689 |
| DINOv3 | 0.9646 | 0.9663 |

An older PDF summary (`InspectAI_project_summary.pdf`) may predate later
milestones; prefer the markdown experiment notes above.
