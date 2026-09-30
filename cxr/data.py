"""Stage 1: data. All three commands are run by pod/data.sh.

    python -m cxr.data chexmask --src ChestX-Ray8.csv        # compact CheXmask file (landmarks only)
    python -m cxr.data ctr                                   # CTR per image, summary + first figure
    python -m cxr.data prepare --nih-zip images_001.zip --csv data/Data_Entry_2017_v2020.csv
                                                             # 512-px NIH subset -> data/nih512

chexmask: keeps the columns this project uses from the 2.2 GB CheXmask file for NIH (image,
mask quality, landmarks, image size): ~115 MB. The heart and lung masks are these landmark
contours, filled, so they can be redrawn at any resolution when needed.

ctr: cardiothoracic ratio (CTR = heart width / chest width) of every NIH image, measured on the
CheXmask landmarks (heart and lung contours). Writes one row per image to outputs/ctr_nih.csv,
then results/ctr_summary.csv and results/ctr_by_view.png: CTR of cardiomegaly vs "No Finding"
films, on PA and AP views. This checks the physiology behind the project: the CTR rule should
separate the two groups on PA films, and less so on AP films (the heart looks bigger on AP).

prepare: NIH ChestX-ray14 images from a zip (read without unzipping) or an extracted folder. Works
on whatever part of the dataset it is given, so the 12 image zips can be fed one at a time. Keeps PA views with Cardiomegaly, Effusion or Pneumothorax plus 8,000 "No Finding" images,
padded to square and resized to 512 px, split 80/10/10 by patient (a patient never appears in
two splits). Images already converted are skipped, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import DATA, OUTPUTS, RESULTS

ALL_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

# CheXmask landmarks: 120 (x, y) points in pixels of the original image, in this order:
# 44 on the right lung, 50 on the left lung, 26 on the heart.
LUNGS, HEART = slice(0, 94), slice(94, 120)
N_LANDMARK_VALUES = 240

# Figure colors: light chart surface, ink, and one accent (the rest stays gray).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"


# ---------------------------------------------------------------- labels

def find_column(df: pd.DataFrame, *candidates: str) -> str:
    lowered = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise KeyError(f"None of {candidates} found in CSV columns {list(df.columns)}")


def load_labels(csv_source) -> pd.DataFrame:
    """NIH Data_Entry_2017*.csv -> one row per image, one 0/1 column per finding."""
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


# ---------------------------------------------------------------- ctr

def heart_chest_widths(landmarks: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Widths in pixels from CheXmask landmark strings ("x1,y1,x2,y2,..."); NaN if malformed.

    Heart width: horizontal extent of the heart contour. Chest width: horizontal extent of both
    lungs, from the outer edge of one lung to the outer edge of the other.
    """
    ok = landmarks.str.count(",").eq(N_LANDMARK_VALUES - 1).fillna(False).to_numpy(dtype=bool)
    heart = np.full(len(landmarks), np.nan)
    chest = np.full(len(landmarks), np.nan)
    if ok.any():
        xy = landmarks[ok].str.split(",", expand=True).astype(float).to_numpy().reshape(-1, 120, 2)
        x = xy[:, :, 0]
        heart[ok] = x[:, HEART].max(axis=1) - x[:, HEART].min(axis=1)
        chest[ok] = x[:, LUNGS].max(axis=1) - x[:, LUNGS].min(axis=1)
    return heart, chest


def auroc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties counted as half)."""
    ranks = pd.Series(np.concatenate([positives, negatives])).rank().to_numpy()
    n_pos, n_neg = len(positives), len(negatives)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in ["PA", "AP"]:
        d = df[df["view"] == view]
        cardio, normal = d[d["cardiomegaly"] == 1], d[d["no_finding"] == 1]
        for group, g in [("Cardiomegaly", cardio), ("No finding", normal)]:
            rows.append({
                "view": view, "group": group,
                "images": len(g), "patients": g["patient_id"].nunique(),
                "ctr_median": g["ctr"].median(),
                "ctr_q25": g["ctr"].quantile(0.25), "ctr_q75": g["ctr"].quantile(0.75),
                "pct_ctr_above_0.5": 100 * (g["ctr"] > 0.5).mean(),
                "auroc_vs_no_finding": (auroc(cardio["ctr"].to_numpy(), normal["ctr"].to_numpy())
                                        if group == "Cardiomegaly" else np.nan),
            })
    return pd.DataFrame(rows).round(3)


def plot_ctr(df: pd.DataFrame, summary: pd.DataFrame, path: Path, min_rca: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(0.24, 0.8601, 0.02)  # 0.5 is a bin edge
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor=SURFACE)
    top = 0.0
    for ax, view in zip(axes, ["PA", "AP"]):
        d = df[df["view"] == view]
        for name, values, color in [("No finding", d.loc[d["no_finding"] == 1, "ctr"], MUTED),
                                    ("Cardiomegaly", d.loc[d["cardiomegaly"] == 1, "ctr"], ACCENT)]:
            density, _ = np.histogram(values, bins=bins, density=True)
            top = max(top, density.max())
            ax.stairs(density, bins, fill=True, color=color, alpha=0.1, linewidth=0)
            ax.stairs(density, bins, color=color, linewidth=2, label=name)
        ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))

        s = summary[summary["view"] == view].set_index("group")
        ax.set_title(f"{view} films", loc="left", fontsize=12, color=INK, fontweight="medium")
        ax.text(0.98, 0.96,
                f"films with CTR > 0.5\n"
                f"cardiomegaly  {s.loc['Cardiomegaly', 'pct_ctr_above_0.5']:.0f}%  "
                f"(n = {s.loc['Cardiomegaly', 'images']:,})\n"
                f"no finding  {s.loc['No finding', 'pct_ctr_above_0.5']:.0f}%  "
                f"(n = {s.loc['No finding', 'images']:,})\n"
                f"AUROC  {s.loc['Cardiomegaly', 'auroc_vs_no_finding']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK_2,
                linespacing=1.5)

        ax.set_facecolor(SURFACE)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("CTR (heart width / chest width)", fontsize=10, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(AXIS)
    axes[0].set_ylim(0, 1.45 * top)  # headroom so the text never sits on a curve
    axes[0].set_ylabel("Density", fontsize=10, color=INK_2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)

    fig.text(0.01, 0.97, "Cardiothoracic ratio from CheXmask masks, NIH ChestX-ray14",
             fontsize=13, color=INK, fontweight="medium", va="top")
    fig.text(0.01, 0.905, f"One value per film · masks with RCA Dice ≥ {min_rca} · "
             "labels mined from radiology reports (noisy) · dashed line: the 0.5 rule",
             fontsize=9.5, color=INK_2, va="top")
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.99, wspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


CHEXMASK_COLUMNS = ["Image Index", "Dice RCA (Mean)", "Landmarks", "Height", "Width"]


def chexmask(args: argparse.Namespace) -> None:
    tmp = args.out.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, chunk in enumerate(pd.read_csv(args.src, usecols=CHEXMASK_COLUMNS, chunksize=20_000)):
        chunk[CHEXMASK_COLUMNS].to_csv(tmp, mode="w" if i == 0 else "a", header=i == 0, index=False)
        n += len(chunk)
    tmp.replace(args.out)  # atomic: a killed job never leaves a half-written file
    print(f"{n:,} rows -> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


def ctr(args: argparse.Namespace) -> None:
    parts = []
    for chunk in pd.read_csv(args.chexmask, usecols=["Image Index", "Dice RCA (Mean)", "Landmarks"],
                             chunksize=20_000):
        heart, chest = heart_chest_widths(chunk["Landmarks"])
        parts.append(pd.DataFrame({"image": chunk["Image Index"].to_numpy(),
                                   "rca_dice": chunk["Dice RCA (Mean)"].to_numpy(),
                                   "heart_width": heart, "chest_width": chest}))
    masks = pd.concat(parts, ignore_index=True)
    masks["ctr"] = masks["heart_width"] / masks["chest_width"]

    labels = load_labels(args.labels)
    keep = ["image", "patient_id", "view", "age", "sex",
            "cardiomegaly", "effusion", "pneumothorax", "no_finding"]
    df = labels[keep].merge(masks, on="image", how="inner")
    df["good_mask"] = (df["rca_dice"] >= args.min_rca) & df["ctr"].notna()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out, index=False)
    print(f"{len(labels):,} labelled images, {len(masks):,} with CheXmask masks, "
          f"{len(df):,} matched, {int(df['good_mask'].sum()):,} with RCA Dice >= {args.min_rca}. "
          f"Per-image table: {args.out}")

    good = df[df["good_mask"]]
    summary = summarize(good)
    args.results.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results / "ctr_summary.csv", index=False)
    plot_ctr(good, summary, args.results / "ctr_by_view.png", args.min_rca)
    print(summary.to_string(index=False))
    print(f"Figure: {args.results / 'ctr_by_view.png'}")


# ---------------------------------------------------------------- prepare

_ZIP: zipfile.ZipFile | None = None  # one handle per worker process


def patient_hash(patient_id) -> float:
    """md5 of the patient ID as a number in [0, 1): the splits (and H3's fresh test set) are ranges of it."""
    return int(hashlib.md5(str(patient_id).encode()).hexdigest(), 16) / 16**32


def patient_split(patient_id, val_frac: float, test_frac: float) -> str:
    """Deterministic split from a hash of the patient ID."""
    h = patient_hash(patient_id)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def select(df: pd.DataFrame, views: list[str], findings: list[str], n_normal: int, seed: int) -> pd.DataFrame:
    df = df[df["view"].isin(views)]
    positives = df[df[[f.lower() for f in findings]].sum(axis=1) > 0]
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


def convert_one(task: tuple[str, str, int]) -> str | None:
    src, dst, size = task
    try:
        source = io.BytesIO(_ZIP.read(src)) if _ZIP is not None else Path(src)
        to_square_resized(source, Path(dst), size)
        return None
    except Exception as exc:  # keep going; report at the end
        return f"{src}: {exc}"


def prepare(args: argparse.Namespace) -> None:
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

    meta = select(load_labels(csv_source), views, findings, args.n_normal, args.seed)
    meta["split"] = meta["patient_id"].apply(patient_split, args=(args.val_frac, args.test_frac))
    meta["file"] = "images/" + meta["image"]

    wanted = set(meta["image"])
    if zf is not None:
        sources = {Path(m).name: m for m in members if Path(m).name in wanted}
    else:
        sources = {p.name: str(p) for p in args.nih_dir.rglob("*.png") if p.name in wanted}
    image_dir = args.out_dir / "images"

    tasks = [(sources[name], str(image_dir / name), args.size)
             for name in meta["image"] if name in sources and not (image_dir / name).exists()]
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
    print(f"Selected {len(meta)} images ({', '.join(views)} views); found in source: {len(sources)}; "
          f"converted now: {len(tasks) - len(errors)}; available in total: {len(avail)}")
    print("Available per split:", avail["split"].value_counts().to_dict())
    print("Available per finding:", {f: int(avail[f.lower()].sum()) for f in findings},
          "| no finding:", int(avail["no_finding"].sum()))
    if errors:
        print(f"{len(errors)} errors, first ones:", *errors[:5], sep="\n  ")
        # Fail loudly (e.g. disk quota): the job then stops before marking this zip as done.
        raise SystemExit(f"{len(errors)} images failed; run again to retry them.")


# ---------------------------------------------------------------- command line

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("chexmask", help="Compact CheXmask file: landmarks and mask quality only")
    m.add_argument("--src", type=Path, required=True, help="CheXmask OriginalResolution/ChestX-Ray8.csv")
    m.add_argument("--out", type=Path, default=DATA / "chexmask_nih_landmarks.csv")
    m.set_defaults(func=chexmask)

    c = sub.add_parser("ctr", help="CTR per image from CheXmask, summary table and figure")
    c.add_argument("--chexmask", type=Path, default=DATA / "chexmask_nih_landmarks.csv",
                   help="Output of the chexmask command (or the original CheXmask CSV)")
    c.add_argument("--labels", type=Path, default=DATA / "Data_Entry_2017_v2020.csv", help="NIH labels CSV")
    c.add_argument("--min-rca", type=float, default=0.7,
                   help="Keep masks with Dice RCA (Mean) >= this (CheXmask authors' advice: 0.7)")
    c.add_argument("--out", type=Path, default=OUTPUTS / "ctr_nih.csv")
    c.add_argument("--results", type=Path, default=RESULTS)
    c.set_defaults(func=ctr)

    p = sub.add_parser("prepare", help="512-px NIH subset with a patient-level split")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--nih-zip", type=Path, help="A zip with NIH PNGs (any part of the dataset), read without extracting")
    src.add_argument("--nih-dir", type=Path, help="Folder with Data_Entry_2017*.csv and the PNG images")
    p.add_argument("--out-dir", type=Path, default=DATA / "nih512")
    p.add_argument("--csv", type=Path, default=None, help="Labels CSV (default: found in the zip/folder)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--views", default="PA", help="Comma-separated views to keep, e.g. PA or PA,AP")
    p.add_argument("--findings", default="Cardiomegaly,Effusion,Pneumothorax")
    p.add_argument("--n-normal", type=int, default=8000, help="Number of 'No Finding' images to keep")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    # Inside a container os.cpu_count() sees all the host's CPUs, not the pod's share: cap it.
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.set_defaults(func=prepare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
