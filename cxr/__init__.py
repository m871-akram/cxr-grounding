"""Do medical VLMs look at the X-ray? Counterfactual chest X-rays to test and fix MedGemma.

One module per stage, each run from the project folder as `python -m cxr.<stage>`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # datasets (cluster only, not in git)
OUTPUTS = ROOT / "outputs"    # big working files: per-image tables, checkpoints (not in git)
RESULTS = ROOT / "results"    # small tables and figures shown in the README (in git)
