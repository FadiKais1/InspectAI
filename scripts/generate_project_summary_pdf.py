"""Generate docs/InspectAI_project_summary.pdf with a pure-Python PDF writer."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "InspectAI_project_summary.pdf"

PAGE_W = 595  # A4
PAGE_H = 842


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Page:
    def __init__(self) -> None:
        self.ops: list[str] = []

    def fill_rect(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")

    def line(self, x1: float, y1: float, x2: float, y2: float, r: float, g: float, b: float, width: float = 1.2) -> None:
        self.ops.append(
            f"{width:.1f} w {r:.3f} {g:.3f} {b:.3f} RG {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S"
        )

    def text(self, x: float, y: float, content: str, *, size: int = 11, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        self.ops.append(
            f"BT /{font} {size} Tf {x:.1f} {y:.1f} Td ({_esc(content)}) Tj ET"
        )


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else f"{cur} {w}"
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def write_pdf(pages: list[Page], path: Path) -> None:
    objects: list[bytes] = []

    def add(obj: str) -> int:
        objects.append(obj.encode("latin-1", errors="replace"))
        return len(objects)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    content_ids = []

    # fonts
    # we'll assign page objects after knowing count; simpler: build content streams first
    content_objs = []
    for page in pages:
        stream = "\n".join(page.ops)
        content_objs.append(stream)

    # object 1 catalog, 2 pages, 3 F1, 4 F2, then pairs of page+content
    # rebuild with known IDs
    objects.clear()
    n_pages = len(pages)
    # 1 catalog, 2 pages tree, 3 Helvetica, 4 Helvetica-Bold
    # pages start at 5: for i in range n: page obj, content obj
    first_page_id = 5
    page_ids = [first_page_id + 2 * i for i in range(n_pages)]
    content_ids = [first_page_id + 2 * i + 1 for i in range(n_pages)]

    add("<< /Type /Catalog /Pages 2 0 R >>")
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    add(f"<< /Type /Pages /Kids [ {kids_str} ] /Count {n_pages} >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    for page, pid, cid, stream in zip(pages, page_ids, content_ids, content_objs, strict=True):
        add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {cid} 0 R >>"
        )
        payload = stream.encode("latin-1", errors="replace")
        objects.append(f"<< /Length {len(payload)} >>\nstream\n".encode("latin-1") + payload + b"\nendstream")

    # xref
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(f"{i} 0 obj\n".encode("latin-1"))
        buf.extend(obj)
        if not obj.endswith(b"\n"):
            buf.extend(b"\n")
        buf.extend(b"endobj\n")
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def chrome(page: Page, footer: str) -> None:
    page.fill_rect(0, PAGE_H - 22, PAGE_W, 22, 0.06, 0.15, 0.27)
    page.fill_rect(0, 0, PAGE_W, 28, 0.06, 0.15, 0.27)
    page.text(36, 10, footer, size=8)


def add_heading(page: Page, y: float, title: str) -> float:
    page.text(40, y, title, size=16, bold=True)
    page.line(40, y - 8, 555, y - 8, 0.12, 0.44, 0.55, 1.6)
    return y - 28


def add_bullets(page: Page, y: float, items: list[str], *, width: int = 92) -> float:
    for item in items:
        lines = wrap("•  " + item, width)
        for i, line in enumerate(lines):
            page.text(44 if i == 0 else 56, y, line, size=10)
            y -= 14
        y -= 4
    return y


def build() -> list[Page]:
    pages: list[Page] = []

    p = Page()
    chrome(p, "InspectAI  |  portfolio roadmap  |  Python 3.11  |  PyTorch planned")
    p.text(40, 790, "InspectAI", size=26, bold=True)
    p.text(40, 768, "Industrial visual anomaly detection  -  project summary", size=12)
    y = 740
    for line in wrap(
        "A research-oriented Python package for MVTec-style quality inspection. "
        "Long-term goal: a defensible portfolio system that can later support real-time inference. "
        "This PDF describes architecture, milestones, and planned deep-learning parts. "
        "No model has been trained yet. Dataset numbers below are EDA facts, not experiment scores.",
        92,
    ):
        p.text(40, y, line, size=10)
        y -= 14
    y -= 10
    y = add_heading(p, y, "Current status (done)")
    y = add_bullets(
        p,
        y,
        [
            "Python 3.11 src-layout package (inspectai) with pytest and Ruff.",
            "Milestone 1: transistor EDA notebook; data/raw/ is read-only.",
            "Milestone 2: framework-independent MVTecDataset (NumPy/Pillow, no PyTorch).",
            "Train = normals only; test discovers defect folders; masks uint8 {0,1}; native 1024x1024.",
            "Unit tests on fake trees plus transistor integration tests (skip if data missing).",
        ],
    )
    y -= 6
    y = add_heading(p, y, "Immediate next step")
    p.text(40, y, "Milestone 3  -  PyTorch adapter and experiment contract (not modeling yet).", size=11, bold=True)
    y -= 20
    y = add_bullets(
        p,
        y,
        [
            "Install PyTorch as an explicit, approved dependency.",
            "Add a thin torch.utils.data.Dataset wrapper around MVTecDataset (CHW tensors).",
            "Do not introduce aggressive resize that would hide small transistor lead defects.",
            "Write hypothesis, baseline, and metrics before any training run (AGENTS.md).",
            "The first model experiment is Milestone 4, after the adapter lands.",
        ],
    )
    pages.append(p)

    p = Page()
    chrome(p, "InspectAI  |  repository parts")
    y = add_heading(p, 790, "Repository parts")
    parts = [
        ("src/inspectai/data/", "MVTec loading, pairing, validation, paired-transform hook. Implemented."),
        ("src/inspectai/models/", "PyTorch modules. Empty. Planned AE / PatchCore backbone / later detectors."),
        ("src/inspectai/evaluation/", "Image/pixel AUROC, PRO, thresholds, pass/fail. Planned."),
        ("src/inspectai/visualization/", "Overlays and score maps. Helpers still live in the EDA notebook."),
        ("src/inspectai/inference/", "Single-image, batch, later stream scoring. Planned."),
        ("src/inspectai/utils/", "Seed, paths, logging. Minimal later."),
        ("configs/", "data / model / experiment YAML. Placeholders only."),
        ("notebooks/", "EDA and analysis only. 01_mvtec_eda.ipynb exists."),
        ("scripts/", "Thin CLIs (train, eval, infer) calling the package. Planned."),
        ("tests/", "Package smoke plus MVTec unit/integration. Implemented."),
        ("data/raw/", "Immutable MVTec AD. Transistor: 213 train good, 60 test good, 40 anomalous."),
        ("outputs/", "figures, checkpoints, experiment logs (gitignored contents)."),
    ]
    for name, desc in parts:
        p.text(40, y, name, size=9, bold=True)
        y -= 13
        for line in wrap(desc, 92):
            p.text(48, y, line, size=10)
            y -= 13
        y -= 6
    pages.append(p)

    p = Page()
    chrome(p, "InspectAI  |  roadmap")
    y = add_heading(p, 790, "Roadmap to a complete portfolio project")
    stages = [
        ("M1 EDA", "DONE", "Understand transistor: imbalance, tiny leads, 1024px, mask pairing."),
        ("M2 Data API", "DONE", "MVTecDataset + MVTecSample; split-specific validation; tests."),
        ("M3 Torch adapter", "NEXT", "Install torch; Dataset wrapper; DataLoader smoke; no training."),
        ("M4 Baseline AD", "PLANNED", "Hypothesis + metrics first. Frozen encoder + kNN/coreset or similar."),
        ("M5 Strong AD", "PLANNED", "Memory-bank method (PatchCore-style) at high resolution or patches."),
        ("M6 Second family", "PLANNED", "Reconstruction autoencoder for comparison, not as the only model."),
        ("M7 Evaluation", "PLANNED", "Image AUROC, pixel AUROC, PRO; per-defect tables; no invented scores."),
        ("M8 Categories", "PLANNED", "Same pipeline on more MVTec classes; configs per experiment."),
        ("M9 Inference CLI", "PLANNED", "scripts/infer.py: image in, score + overlay out."),
        ("M10 Optional product", "LATER", "Stream/API/dashboard only after models and eval are documented."),
    ]
    p.text(40, y, "Stage", size=9, bold=True)
    p.text(150, y, "State", size=9, bold=True)
    p.text(220, y, "Scope", size=9, bold=True)
    y -= 16
    p.line(40, y + 8, 555, y + 8, 0.8, 0.84, 0.88, 0.6)
    for stage, state, scope in stages:
        p.text(40, y, stage, size=10, bold=True)
        p.text(150, y, state, size=9, bold=True)
        sy = y
        for i, line in enumerate(wrap(scope, 58)):
            p.text(220, sy, line, size=9)
            sy -= 12
        y = sy - 8
    y -= 8
    y = add_heading(p, y, "Definition of done (core project)")
    for line in wrap(
        "Reproducible train/eval on transistor; at least two model families compared with the same "
        "metrics; written limitations (small defects, resize risk); documented inference path. "
        "A real-time UI is extra, not required for the research portfolio core.",
        92,
    ):
        p.text(40, y, line, size=10)
        y -= 14
    pages.append(p)

    p = Page()
    chrome(p, "InspectAI  |  planned models and DL parts")
    y = add_heading(p, 790, "Planned models and deep-learning parts")
    p.text(40, y, "None of the following are implemented. They are the intended stack after Milestone 3.", size=10)
    y -= 24
    blocks = [
        (
            "Embedding / memory-bank (primary AD path)",
            (
                "Wide-ResNet or similar frozen ImageNet backbone, patch-level memory (PatchCore-style). "
                "Fits MVTec one-class training and small defects if patches stay high-resolution. "
                "Typically little or no gradient training; scoring is nearest-neighbor distance in feature space."
            ),
        ),
        (
            "Reconstruction (comparison baseline)",
            (
                "Convolutional autoencoder trained on good images only. Residual maps as anomaly scores. "
                "Useful as a baseline; often weaker on fine lead defects than patch embeddings."
            ),
        ),
        (
            "Self-supervised features (optional, after baseline)",
            (
                "DINO / ViT features instead of a supervised CNN, same memory-bank head. "
                "Only after a working PatchCore-style baseline exists."
            ),
        ),
        (
            "Detection (optional, not the default)",
            (
                "YOLO-style detectors need box labels. MVTec AD is one-class segmentation. "
                "Use YOLO only if the problem shifts to boxed detection, as a separate experiment."
            ),
        ),
    ]
    for title, para in blocks:
        p.text(40, y, title, size=11, bold=True)
        y -= 16
        for line in wrap(para, 92):
            p.text(44, y, line, size=10)
            y -= 13
        y -= 10
    y = add_heading(p, y, "DL system pieces (engineering, not architectures)")
    y = add_bullets(
        p,
        y,
        [
            "Data: MVTecDataset -> later tensor Dataset -> DataLoader. Paired (image, mask) transforms.",
            "Train: good-only; no defect labels in the one-class setting.",
            "Score: distance to memory bank, or reconstruction error (AE path).",
            "Eval: image AUROC; pixel AUROC / PRO vs binary masks; per-defect breakdown.",
            "Inference: anomaly map + scalar; threshold pass/fail; overlay visualization.",
            "Ops: experiment YAML, seeds, checkpoints under outputs/; never invent metrics.",
        ],
        width=90,
    )
    pages.append(p)

    p = Page()
    chrome(p, "InspectAI  |  constraints and experiment contract")
    y = add_heading(p, 790, "Transistor constraints that shape the DL design")
    y = add_bullets(
        p,
        y,
        [
            "One-class: 213 train good, 60 test good, 10 images x 4 defect types (40 anomalous).",
            "All RGB images and masks 1024x1024 PNG; masks pair 1:1 with defects.",
            "bent_lead / cut_lead can be well under 2% of pixels; 224px resize can erase them.",
            "misplaced is pose of the whole part (~40% of pixels), not a local nick.",
            "Good leads already curve into the board; bent_lead is extreme displacement.",
            "Default: native resolution or high-res patches until an experiment justifies resize.",
        ],
    )
    y -= 8
    y = add_heading(p, y, "Experiment contract (before any training)")
    y = add_bullets(
        p,
        y,
        [
            "Hypothesis, e.g. patch embeddings at native scale beat an autoencoder on lead defects.",
            "Baseline: simple frozen-encoder kNN or AE, not a random score.",
            "Metrics: image AUROC overall and per defect; pixel AUROC; qualitative overlays.",
            "Split: official MVTec train/test only; never mix test defects into train.",
            "Report failures, especially tiny leads vs large misplaced regions.",
        ],
    )
    y -= 8
    y = add_heading(p, y, "Out of scope until explicitly approved")
    y = add_bullets(
        p,
        y,
        [
            "Training loops, PatchCore/DINO/YOLO implementations, APIs, and dashboards.",
            "Any modification of files under data/raw/.",
            "Reusable training logic inside notebooks.",
        ],
    )
    y -= 12
    p.text(40, y, "Next action: approve Milestone 3 (PyTorch adapter only), then define the M4 hypothesis.", size=11, bold=True)
    pages.append(p)
    return pages


def main() -> None:
    write_pdf(build(), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
