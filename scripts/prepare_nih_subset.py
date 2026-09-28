"""Build a 512-px NIH ChestX-ray14 subset with a patient-level split.

Reads either the Kaggle zip directly (--nih-zip, no need to unzip 45 GB) or an
extracted folder (--nih-dir). Works on whatever part of the dataset is present;
images already converted are skipped; metadata.csv is rewritten at each run.

Examples (inside a job, see slurm/download_nih.sbatch):
    python scripts/prepare_nih_subset.py --nih-zip "$JOB_TMP/data.zip" --out-dir data/nih512
    python scripts/prepare_nih_subset.py --nih-dir /path/to/extracted --out-dir data/nih512

Selection (defaults): PA views only; every image with Cardiomegaly, Effusion or
Pneumothorax; plus 8,000 "No Finding" images. Split 80/10/10 by patient, so the
same patient never appears in two splits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ALL_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

_ZIP: zipfile.ZipFile | None = None  # one handle per worker process


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--nih-zip", type=Path,
                     help="The Kaggle zip (nih-chest-xrays/data), read without extracting.")
    src.add_argument("--nih-dir", type=Path,
                     help="Folder with Data_Entry_2017*.csv and the PNG images (any sub-folders).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=None, help="Metadata CSV (default: found in the zip/folder).")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--views", default="PA", help="Comma-separated view positions to keep, e.g. PA or PA,AP.")
    p.add_argument("--findings", default="Cardiomegaly,Effusion,Pneumothorax")
    p.add_argument("--n-normal", type=int, default=8000, help="Number of 'No Finding' images to keep.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    return p.parse_args()


def find_column(df: pd.DataFrame, *candidates: str) -> str:
    lowered = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise KeyError(f"None of {candidates} found in CSV columns {list(df.columns)}")


def load_metadata(csv_source) -> pd.DataFrame:
    raw = pd.read_csv(csv_source)
    df = pd.DataFrame({
        "image": raw[find_column(raw, "Image Index")],
        "finding_labels": raw[find_column(raw, "Finding Labels")],
        "patient_id": raw[find_column(raw, "Patient ID")],
        "age": raw[find_column(raw, "Patient Age")],
        "sex": raw[find_column(raw, "Patient Gender", "Patient Sex")],
        "view": raw[find_column(raw, "View Position")],
        "follow_up": raw[find_column(raw, "Follow-up #")],
    })
    labels = df["finding_labels"].str.split("|")
    for finding in ALL_FINDINGS:
        df[finding.lower()] = labels.apply(lambda ls, f=finding: int(f in ls))
    df["no_finding"] = labels.apply(lambda ls: int(ls == ["No Finding"]))
    return df


def patient_split(patient_id, val_frac: float, test_frac: float) -> str:
    """Deterministic split from a hash of the patient ID."""
    h = int(hashlib.md5(str(patient_id).encode()).hexdigest(), 16) / 16**32
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def select(df: pd.DataFrame, views: list[str], findings: list[str], n_normal: int, seed: int) -> pd.DataFrame:
    df = df[df["view"].isin(views)]
    target_cols = [f.lower() for f in findings]
    positives = df[df[target_cols].sum(axis=1) > 0]
    normals = df[df["no_finding"] == 1]
    normals = normals.sample(n=min(n_normal, len(normals)), random_state=seed)
    return pd.concat([positives, normals]).drop_duplicates("image").reset_index(drop=True)


def _open_zip(path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(path)


def to_square_resized(src, dst: Path, size: int) -> None:
    img = Image.open(src).convert("L")  # also handles the few RGBA images in NIH
    w, h = img.size
    if w != h:  # pad to square (black borders) to keep the anatomy's proportions
        side = max(w, h)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    img = img.resize((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    img.save(tmp)
    tmp.replace(dst)  # atomic: no half-written files if the job is killed


def convert_one(args: tuple[str, str, int]) -> str | None:
    src, dst, size = args
    try:
        source = io.BytesIO(_ZIP.read(src)) if _ZIP is not None else Path(src)
        to_square_resized(source, Path(dst), size)
        return None
    except Exception as exc:  # keep going; report at the end
        return f"{src}: {exc}"


def main() -> None:
    args = parse_args()

    zf = zipfile.ZipFile(args.nih_zip) if args.nih_zip else None
    if zf is not None:
        members = zf.namelist()
        csv_member = next((m for m in sorted(members)
                           if Path(m).name.startswith("Data_Entry_2017") and m.endswith(".csv")), None)
        if args.csv is None and csv_member is None:
            raise SystemExit(f"No Data_Entry_2017*.csv inside {args.nih_zip}; pass --csv.")
        csv_source = args.csv or io.BytesIO(zf.read(csv_member))
    else:
        csv_source = args.csv or next(iter(sorted(args.nih_dir.rglob("Data_Entry_2017*.csv"))), None)
        if csv_source is None:
            raise SystemExit(f"No Data_Entry_2017*.csv found under {args.nih_dir}; pass --csv.")

    views = [v.strip() for v in args.views.split(",")]
    findings = [f.strip() for f in args.findings.split(",")]
    unknown = set(findings) - set(ALL_FINDINGS)
    if unknown:
        raise SystemExit(f"Unknown findings {unknown}. Valid: {ALL_FINDINGS}")

    meta = select(load_metadata(csv_source), views, findings, args.n_normal, args.seed)
    meta["split"] = meta["patient_id"].apply(patient_split, args=(args.val_frac, args.test_frac))
    meta["file"] = "images/" + meta["image"]

    wanted = set(meta["image"])
    if zf is not None:
        sources = {Path(m).name: m for m in members if Path(m).name in wanted}
    else:
        sources = {p.name: str(p) for p in args.nih_dir.rglob("*.png") if p.name in wanted}
    image_dir = args.out_dir / "images"

    tasks = [
        (sources[name], str(image_dir / name), args.size)
        for name in meta["image"]
        if name in sources and not (image_dir / name).exists()
    ]
    errors: list[str] = []
    if tasks:
        pool_kwargs = {"initializer": _open_zip, "initargs": (str(args.nih_zip),)} if zf is not None else {}
        with ProcessPoolExecutor(max_workers=args.workers, **pool_kwargs) as pool:
            for i, err in enumerate(pool.map(convert_one, tasks, chunksize=32), start=1):
                if err:
                    errors.append(err)
                if i % 1000 == 0:
                    print(f"  converted {i}/{len(tasks)}", flush=True)

    meta["available"] = meta["image"].apply(lambda n: (image_dir / n).exists())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.out_dir / "metadata.csv", index=False)

    avail = meta[meta["available"]]
    print(f"Selected {len(meta)} images ({', '.join(views)} views); "
          f"found in source: {len(sources)}; converted now: {len(tasks) - len(errors)}; "
          f"available in total: {len(avail)}")
    print("Available per split:", avail["split"].value_counts().to_dict())
    print("Available per finding:", {f: int(avail[f.lower()].sum()) for f in findings},
          "| no finding:", int(avail["no_finding"].sum()))
    if errors:
        print(f"{len(errors)} errors, first ones:", *errors[:5], sep="\n  ")


if __name__ == "__main__":
    main()
