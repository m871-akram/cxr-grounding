"""Provenance records written before the network volume is deleted (run on a pod with the volume).

    python -m cxr.provenance images        # decode every saved image fully -> results/provenance_images.csv
    python -m cxr.provenance manifests     # code commits of the H3 manifests -> results/provenance_manifests.csv
    python -m cxr.provenance environment   # pip freeze and model versions -> results/provenance_*.txt / .csv

Images: the generation runs saved straight to their final paths and skip existing files, so a file
truncated by an interrupted session would have been skipped on resume; decoding every pixel (PIL
checks the PNG chunks' checksums and raises on a truncated file) rules that out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from . import OUTPUTS, RESULTS

FOLDERS = ["h3/train", "h3/val", "h3/fresh", "pairs/test", "dose"]


def images(args: argparse.Namespace) -> None:
    """Every PNG under the saved-image folders, fully decoded: one row per folder (count, failures,
    sizes and modes) -> results/provenance_images.csv; failures, if any, listed one per row."""
    rows, failures = [], []
    for folder in FOLDERS:
        n, sizes = 0, {}
        for path in sorted((OUTPUTS / folder).rglob("*.png")):
            n += 1
            try:
                with Image.open(path) as im:
                    im.load()  # every pixel: a truncated or corrupted file raises here
                    sizes[f"{im.size[0]}x{im.size[1]} {im.mode}"] = sizes.get(f"{im.size[0]}x{im.size[1]} {im.mode}", 0) + 1
            except Exception as error:  # noqa: BLE001 - every failure is reported, whatever its kind
                failures.append({"file": str(path.relative_to(OUTPUTS)), "error": repr(error)})
        rows.append({"folder": f"outputs/{folder}", "png_files": n, "failed": sum(f["file"].startswith(folder) for f in failures),
                     "sizes_and_modes": json.dumps(sizes, sort_keys=True)})
        print(f"  outputs/{folder}: {n} PNG files decoded, {rows[-1]['failed']} failed; {sizes}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "provenance_images.csv", index=False)
    pd.DataFrame(failures, columns=["file", "error"]).to_csv(RESULTS / "provenance_image_failures.csv", index=False)


def manifests(args: argparse.Namespace) -> None:
    """The settings hash, code commits and model revisions recorded by every H3 generation folder."""
    rows = []
    for path in sorted((OUTPUTS / "h3").rglob("manifest.json")):
        m = json.loads(path.read_text())
        rows.append({"folder": str(path.parent.relative_to(OUTPUTS)), "settings_sha256": m["settings_hash"],
                     "commits": " ".join(m["commits"]), "revisions": json.dumps(m["revisions"], sort_keys=True),
                     "png_files": sum(1 for _ in path.parent.glob("*.png"))})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "provenance_manifests.csv", index=False)
    commits = sorted({c for row in rows for c in row["commits"].split()})
    print(f"{len(rows)} H3 folders; distinct code commits: {commits}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment(args: argparse.Namespace) -> None:
    """The exact environment: pip freeze of this Python (the pod's venv) -> results/provenance_pip_freeze.txt;
    the snapshot of every model in the Hugging Face cache (refs/main and the snapshot folders), and the
    SHA-256 of the TorchXRayVision weights after loading the classifiers and segmenter the checks use
    -> results/provenance_models.csv."""
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (RESULTS / "provenance_pip_freeze.txt").write_text(f"# {sys.executable}, Python {sys.version.split()[0]}\n" + freeze)
    rows = []
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for repo in sorted(hub.glob("models--*")):
        ref = repo / "refs" / "main"
        rows.append({"source": "huggingface", "model": repo.name[len("models--"):].replace("--", "/"),
                     "ref_main": ref.read_text().strip() if ref.exists() else "",
                     "snapshots": " ".join(sorted(p.name for p in (repo / "snapshots").iterdir())), "file": "", "sha256": ""})
    import torchxrayvision as xrv  # loading the models downloads their weights if they are not cached yet

    from .checks import CLASSIFIERS
    for name in CLASSIFIERS:
        xrv.models.DenseNet(weights=name)
    xrv.baseline_models.chestx_det.PSPNet()
    cache = Path.home() / ".torchxrayvision"
    for path in sorted(p for p in cache.rglob("*") if p.is_file()):
        rows.append({"source": f"torchxrayvision {xrv.__version__}", "model": path.parent.name, "ref_main": "",
                     "snapshots": "", "file": str(path.relative_to(cache)), "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(RESULTS / "provenance_models.csv", index=False)
    print(pd.DataFrame(rows)[["source", "model", "ref_main", "file"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in [("images", images), ("manifests", manifests), ("environment", environment)]:
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
