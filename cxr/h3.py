"""H3 (PLAN.md, section 4): does LoRA fine-tuning on diffusion counterfactuals transfer to real films?
The training data first:

    python -m cxr.h3 generate --test   # 2 source films per set, into outputs/h3_test: tests the code
    python -m cxr.h3 generate          # E's validation (60 val films), fresh edits (100), E (750)
    python -m cxr.h3 measure           # segmenter CTR of E's images and of the real films R draws from

Patients, by the md5 hash h of the patient ID (data.patient_hash, the hash of the splits): the
training pool is the train split with h >= 0.32, validation the val split, and the fresh test set
the train split with h < 0.32. The fresh test set is reserved for H3's final evaluation: its edits
are made here, but nothing of it is measured or scored before that evaluation.
Source films: normal films with a CheXmask CTR < 0.5 and a good CheXmask mask, one per patient, in a
random order (seed 0); film k is edited with seed + k at every growth, and its sham with the same seed.
Every image folder holds a manifest.json: the settings and their hash, the source films, the seed
rule, the code's git commit and the pinned Hugging Face revisions of RadEdit's parts. A run refuses
to add images to a folder whose manifest differs or is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import DATA, OUTPUTS, RESULTS, ROOT
from .data import patient_hash
from .edit import GENERATE, NORMAL_PROMPT, edit_film, grown_heart_mask, landmarks_512, load_films, load_radedit

FRESH_BELOW = 0.32  # train-split patients below this hash are the fresh test set (train is h >= 0.2)
GROWTHS = [0.06, 0.12, 0.18, 0.24, 0.30]
SETS = {  # set: (source films, growths of the additions); every film also gets a sham at the 0.30 mask
    "val": (60, GROWTHS),                  # E's validation images
    "fresh": (100, [0.0] + GROWTHS),       # H3's edited-film test (as in the dose run)
    "train": (750, GROWTHS),               # E
}
RADEDIT_REPOS = ["microsoft/radedit", "stabilityai/sdxl-vae", "microsoft/BiomedVLP-BioViL-T"]
MARGIN = 0.025  # training examples with a CTR within this of 0.5 are dropped (PLAN.md, H3)


def commit() -> str:
    """The git commit of the code, written to COMMIT by pod/sync.sh push (the pod has no .git)."""
    path = ROOT / "COMMIT"
    return path.read_text().strip() if path.exists() else "unknown"


def cached_revision(repo: str) -> str:
    """The commit of the snapshot of repo in this pod's Hugging Face cache: the version used so far."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main").read_text().strip()


def source_films(which: str, n: int, seed: int) -> pd.DataFrame:
    """Normal films with a CheXmask CTR < 0.5 and a good mask, one per patient, in a random order:
    from the training pool ('train'), the val split ('val') or the fresh test set ('fresh')."""
    films = load_films("val" if which == "val" else "train")
    films = films.merge(pd.read_csv(OUTPUTS / "ctr_nih.csv", usecols=["image", "ctr"]), on="image")
    h = films["patient_id"].map(patient_hash)
    if which == "train":
        films = films[h >= FRESH_BELOW]
    elif which == "fresh":
        films = films[h < FRESH_BELOW]
    films = films[(films["no_finding"] == 1) & (films["ctr"] < 0.5)].drop_duplicates("patient_id")
    return films.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def check_folder(folder: Path, manifest: dict) -> None:
    """Refuse to add images to a folder made with other settings or model revisions, or of unknown
    origin; otherwise write (or update) its manifest. A resumed run adds its commit to the list."""
    path = folder / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["settings_hash"] != manifest["settings_hash"] or old["revisions"] != manifest["revisions"]:
            raise SystemExit(f"{folder}: made with other settings or model revisions; refusing to reuse it")
        manifest["commits"] = old["commits"] + [c for c in manifest["commits"] if c not in old["commits"]]
    elif any(folder.glob("*.png")):
        raise SystemExit(f"{folder}: images without a manifest; refusing to reuse them")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))


def generate(args: argparse.Namespace) -> None:
    """Additions at every growth and one sham per source film, for each set (SETS)."""
    root = OUTPUTS / ("h3_test" if args.test else "h3")
    revisions = {repo: cached_revision(repo) for repo in RADEDIT_REPOS}
    print(f"code commit {commit()}; RadEdit revisions {revisions}", flush=True)
    prompt, _, skip = GENERATE["cardiomegaly"]
    pipe = None
    for which, (n, growths) in SETS.items():
        films = source_films(which, 2 if args.test else n, args.seed)
        jobs = [("edit", g, prompt) for g in growths] + [("sham", 0.30, NORMAL_PROMPT)]
        for kind, growth, text in jobs:
            settings = {"prompt": text, "mask": "grown_heart", "growth": growth, "guidance": args.guidance,
                        "skip_ratio": skip, "steps": args.steps, "feather": args.feather, "tf32": True,
                        "films": films["image"].tolist(), "seed_rule": f"film k: seed {args.seed} + k"}
            digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            check_folder(root / which / kind / f"growth{growth:.2f}",
                         {"settings_hash": digest, "settings": settings, "commits": [commit()], "revisions": revisions})
        print(f"{which}: {len(films)} source films, {len(films) * len(jobs)} images", flush=True)
        for k, film in films.iterrows():
            xy, original = landmarks_512(film), Image.open(DATA / "nih512" / film["file"])
            for kind, growth, text in jobs:
                out = root / which / kind / f"growth{growth:.2f}" / film["image"]
                if out.exists():  # finished images are skipped (the manifest guarantees the settings)
                    continue
                if pipe is None:
                    pipe = load_radedit(revisions=revisions)
                    torch.backends.cuda.matmul.allow_tf32 = True
                Image.fromarray(edit_film(pipe, original, grown_heart_mask(xy, growth), text, args.guidance,
                                          skip, args.steps, args.seed + k, args.feather)).save(out)
            if (k + 1) % 25 == 0:
                print(f"  {which}: {k + 1}/{len(films)} films", flush=True)


def real_films(which: str, seed: int) -> pd.DataFrame:
    """The real PA films R draws from: every film of the training pool ('train') or the val split
    ('val'), whatever its label, at most 3 per patient, drawn at random before any CTR is seen."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta = meta[meta["available"] & (meta["split"] == which)]
    if which == "train":
        meta = meta[meta["patient_id"].map(patient_hash) >= FRESH_BELOW]
    return meta.sample(frac=1, random_state=seed).groupby("patient_id").head(3).reset_index(drop=True)


def segment_ctr(paths: list) -> list:
    """Segmenter CTR of each image, 1,000 images at a time (the whole set would not fit in memory)."""
    from .checks import ctr_of, load_segmenter, segment

    seg, ctrs = load_segmenter(), []
    for i in range(0, len(paths), 1000):
        hearts, lungs = segment(seg, [np.asarray(Image.open(p).convert("L")) for p in paths[i:i + 1000]])
        ctrs += [ctr_of(h, l) for h, l in zip(hearts, lungs)]
        print(f"  segmented {min(i + 1000, len(paths))}/{len(paths)}", flush=True)
    return ctrs


def measure(args: argparse.Namespace) -> None:
    """Segmenter CTR of E's images (training and validation sets) and of the real films of the
    training pool and the val split -> outputs/h3*/ctr_edits.csv and ctr_real.csv; the number of
    examples of each class once the margin is dropped -> results/h3_supply.csv (h3_supply_test.csv
    with --test). The fresh test set is not measured."""
    root = OUTPUTS / ("h3_test" if args.test else "h3")
    edits = [{"set": which, "kind": p.parent.parent.name, "growth": float(p.parent.name[6:]), "image": p.name, "path": p}
             for which in ["train", "val"] for p in sorted((root / which).glob("*/growth*/*.png"))]
    real = pd.concat([real_films(which, args.seed).assign(set=which) for which in ["train", "val"]], ignore_index=True)
    if args.test:
        real = real.groupby("set").head(50)
    edits = pd.DataFrame(edits)
    edits["ctr"] = segment_ctr(edits["path"].tolist())
    real["ctr"] = segment_ctr([DATA / "nih512" / f for f in real["file"]])
    edits.drop(columns="path").to_csv(root / "ctr_edits.csv", index=False)
    real.to_csv(root / "ctr_real.csv", index=False)
    supply(args)


def supply_table(edits: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Examples of each class once the margin is dropped: "yes" is CTR > 0.525 and "no" CTR < 0.475, as
    in training; every other image with a CTR is dropped (the CTRs exactly at 0.475 or 0.525 too), so
    each image is counted once."""
    rows = []
    for arm, table in [("E", edits), ("R", real)]:
        for which, g in table.groupby("set"):
            yes, no = g["ctr"] > 0.5 + MARGIN, g["ctr"] < 0.5 - MARGIN
            row = {"arm": arm, "set": which, "images": len(g), "no_ctr": int(g["ctr"].isna().sum()), "yes": int(yes.sum()),
                   "no": int(no.sum()), "dropped_margin": int((g["ctr"].notna() & ~yes & ~no).sum())}
            assert row["no_ctr"] + row["yes"] + row["no"] + row["dropped_margin"] == row["images"], row
            rows.append(row)
    return pd.DataFrame(rows)


def supply(args: argparse.Namespace) -> None:
    """The supply table from the saved CTR tables -> results/h3_supply.csv (h3_supply_test.csv with --test)."""
    root = OUTPUTS / ("h3_test" if args.test else "h3")
    table = supply_table(pd.read_csv(root / "ctr_edits.csv"), pd.read_csv(root / "ctr_real.csv"))
    table.to_csv(RESULTS / ("h3_supply_test.csv" if args.test else "h3_supply.csv"), index=False)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="E's images, its validation images and the fresh edits")
    g.add_argument("--test", action="store_true", help="2 source films per set, into outputs/h3_test")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--guidance", type=float, default=15.0)
    g.add_argument("--feather", type=float, default=3.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=generate)
    m = sub.add_parser("measure", help="Segmenter CTR of E's images and of R's real films (not the fresh set)")
    m.add_argument("--test", action="store_true", help="The test images and 50 real films per set")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=measure)
    s = sub.add_parser("supply", help="The supply table from the saved CTR tables (no segmentation)")
    s.add_argument("--test", action="store_true", help="The test images")
    s.set_defaults(func=supply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
