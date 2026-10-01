"""Do medical VLMs look at the X-ray? Counterfactual chest X-rays to test and fix MedGemma.

One module per stage, each run from the project folder as `python -m cxr.<stage>`.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # datasets (cluster only, not in git)
OUTPUTS = ROOT / "outputs"    # big working files: per-image tables, checkpoints (not in git)
RESULTS = ROOT / "results"    # small tables and figures shown in the README (in git)


def code_hash() -> str:
    """SHA-256 of the pipeline code (every cxr/*.py, name and bytes). pod/sync.sh push writes it to
    CODE_SHA256, so the pod can check that its code is still the code pushed with COMMIT."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "cxr").glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()
