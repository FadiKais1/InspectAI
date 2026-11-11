"""InspectAI Streamlit demo. Uses the saved DINOv3 Inspector artifact only."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import streamlit as st
from PIL import Image

from inspectai.inference.artifact import ArtifactError
from inspectai.inference.demo import (
    EVALUATION_SUMMARY,
    LOCKED_EVALUATION,
    anomaly_heatmap_figure,
    artifact_is_ready,
    format_inspection_metrics,
    missing_artifact_message,
    resolve_artifact_dir,
    user_facing_error,
)
from inspectai.inference.images import to_uint8_rgb
from inspectai.inference.inspector import Inspector
from inspectai.models.dinov3_anomaly import DINOWeightsError

DEFAULT_ARTIFACT = resolve_artifact_dir(repo_root=REPO_ROOT)


@st.cache_resource(show_spinner="Loading DINOv3 and the normal memory bank…")
def load_cached_inspector(artifact_dir: str) -> Inspector:
    """Load frozen DINOv3 + saved memory once per Streamlit process."""
    return Inspector.load(artifact_dir)


def _read_upload(uploaded) -> np.ndarray:
    try:
        with Image.open(uploaded) as image:
            rgb = to_uint8_rgb(image)
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError("Could not read that file as a PNG or JPEG image.") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Could not read that file as a PNG or JPEG image.")
    return rgb


def main() -> None:
    st.set_page_config(page_title="InspectAI", layout="wide")
    st.title("InspectAI")
    st.caption("Industrial Visual Anomaly Detection")
    st.write(
        "InspectAI compares dense DINOv3 visual representations from an uploaded "
        "image against a memory bank learned only from normal transistor images."
    )

    artifact_dir = DEFAULT_ARTIFACT
    artifact_ready = artifact_is_ready(artifact_dir)
    if not artifact_ready:
        st.error(missing_artifact_message(artifact_dir))

    uploaded = st.file_uploader(
        "Upload an industrial image",
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, or JPEG.",
    )

    original: np.ndarray | None = None
    if uploaded is not None:
        try:
            original = _read_upload(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            original = None
        else:
            st.image(original, caption="Original image", use_container_width=True)

    with st.expander("Advanced", expanded=False):
        st.caption(
            "Optional user-supplied score threshold. Disabled by default. "
            "Not a deployment decision rule and not derived from official test labels."
        )
        use_threshold = st.checkbox("Enable user-supplied threshold", value=False)
        threshold_value = st.number_input(
            "Threshold",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.01,
            disabled=not use_threshold,
        )

    inspect_clicked = st.button("Inspect Image", type="primary", disabled=not artifact_ready)
    if inspect_clicked:
        if not artifact_ready:
            st.error(missing_artifact_message(artifact_dir))
        elif original is None:
            st.warning("Upload a PNG or JPEG image first.")
        else:
            try:
                threshold = float(threshold_value) if use_threshold else None
                with st.spinner("Loading inspector if needed, then inspecting…"):
                    inspector = load_cached_inspector(str(artifact_dir))
                    result = inspector.inspect(original, threshold=threshold)
                st.session_state["inspection"] = {
                    "original": original,
                    "result": result,
                }
            except (DINOWeightsError, ArtifactError, OSError, ValueError, RuntimeError) as exc:
                st.error(user_facing_error(exc, artifact_dir=artifact_dir))

    payload = st.session_state.get("inspection")
    if payload is not None:
        result = payload["result"]
        shown = payload["original"]
        metrics = format_inspection_metrics(
            anomaly_score=result.anomaly_score,
            latency_ms=result.latency_ms,
            model_name=result.model_name,
            input_hw=result.input_hw,
            patch_grid=result.patch_grid,
        )
        cols = st.columns(5)
        for column, (label, value) in zip(cols, metrics.items(), strict=True):
            column.metric(label, value)
        st.caption(
            "Higher anomaly scores indicate greater distance from visual patterns "
            "observed in normal training images."
        )
        if result.threshold is not None:
            comparison = "at or above" if result.anomaly_score >= result.threshold else "below"
            st.info(
                f"User-supplied threshold {result.threshold:.4f}: score is {comparison} threshold. "
                "This is not a NORMAL/DEFECTIVE deployment label."
            )

        left, mid, right = st.columns(3)
        with left:
            st.subheader("Original")
            st.image(shown, use_container_width=True)
        with mid:
            st.subheader("Anomaly heatmap")
            figure = anomaly_heatmap_figure(result.anomaly_map)
            st.pyplot(figure, clear_figure=True)
        with right:
            st.subheader("Overlay")
            st.image(result.overlay, use_container_width=True)

    with st.expander("How it works"):
        st.markdown(
            """
1. The system was fitted only on normal MVTec transistor training images.
2. Frozen DINOv3 ViT-S/16 extracts dense 16×16 patch representations.
3. Each test patch is compared to a 10,000-vector normal memory bank.
4. Nearest-neighbor distance produces the anomaly map.
5. Spatial maximum gives the image anomaly score.

No defect labels are required during fitting. This demo is currently
specialized to the transistor category.
            """
        )

    with st.expander("Model Evaluation", expanded=True):
        for row in LOCKED_EVALUATION:
            st.write(
                f"**{row['name']}** — Image AUROC {row['image_auroc']}, "
                f"Pixel AUROC {row['pixel_auroc']}"
            )
        st.write(EVALUATION_SUMMARY)


if __name__ == "__main__":
    main()
