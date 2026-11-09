"""Model components. Milestone 4 ships a frozen patch-kNN scorer only."""

from inspectai.models.patch_knn import (
    FrozenResNetPatchEncoder,
    PatchKNNConfig,
    PatchKNNScorer,
)

__all__ = ["FrozenResNetPatchEncoder", "PatchKNNConfig", "PatchKNNScorer"]
