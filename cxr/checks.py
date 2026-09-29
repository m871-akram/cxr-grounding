"""Stage 2 checks: did an edit really change the finding? Tools independent of RadEdit and MedGemma.

    python -m cxr.checks validate   # both tools on real NIH PA val films, before any use on edits
                                    # -> results/edit_classifiers.csv, results/segmenter_ctr.{csv,png}

Classifiers: TorchXRayVision DenseNets trained on PadChest (-pc) and CheXpert (-chex), not on NIH
or MIMIC (RadEdit saw NIH; MIMIC is in RadEdit's and MedGemma's training data). A classifier is
used for a finding only if its AUROC on real films (finding vs no finding) is at least 0.80; its
threshold is the one that best separates those films (Youden's J).
Segmenter: TorchXRayVision's ChestX-Det PSPNet finds the heart and lungs of any film, edited or
not: the CTR for cardiomegaly edits, the lung area at the bases for effusion edits. Its CTR is
first compared with the CheXmask CTR of the same real films.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import DATA, OUTPUTS, RESULTS
from .audit import auroc_ci, youden
from .data import ACCENT, AXIS, GRID, INK, INK_2, MUTED, SURFACE, auroc

CLASSIFIERS = ["densenet121-res224-pc", "densenet121-res224-chex"]
LABELS = {"cardiomegaly": "Cardiomegaly", "effusion": "Effusion"}
MIN_AUROC = 0.80
HEART, LEFT_LUNG, RIGHT_LUNG = 8, 4, 5  # channels of the PSPNet output (PSPNet.targets)


def to_xrv(images: list, size: int) -> torch.Tensor:
    """Gray films (0-255) -> TorchXRayVision input: 1 channel, values in [-1024, 1024], size x size."""
    import torchxrayvision as xrv

    resize = xrv.datasets.XRayResizer(size)
    return torch.from_numpy(np.stack([resize(xrv.datasets.normalize(np.asarray(im, np.float32), 255)[None])
                                      for im in images]))


def load_classifiers(device: str = "cuda") -> dict:
    import torchxrayvision as xrv

    return {name: xrv.models.DenseNet(weights=name).to(device).eval() for name in CLASSIFIERS}


@torch.inference_mode()
def classify(models: dict, images: list, device: str = "cuda", batch_size: int = 64) -> pd.DataFrame:
    """One row per film, one column per classifier and finding (0.5 = the model's operating point)."""
    x = to_xrv(images, 224)
    columns = {}
    for name, model in models.items():
        out = torch.cat([model(x[i:i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)])
        for finding, label in LABELS.items():
            columns[f"{name}:{finding}"] = out[:, model.pathologies.index(label)].numpy()
    return pd.DataFrame(columns)


def load_segmenter(device: str = "cuda"):
    import torchxrayvision as xrv

    return xrv.baseline_models.chestx_det.PSPNet().to(device).eval()


@torch.inference_mode()
def segment(seg, images: list, device: str = "cuda", batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Heart and lung masks (N x 512 x 512, bool). The PSPNet returns logits: > 0 means p > 0.5."""
    x = to_xrv(images, 512)
    hearts, lungs = [], []
    for i in range(0, len(x), batch_size):
        logits = seg(x[i:i + batch_size].to(device))
        hearts.append((logits[:, HEART] > 0).cpu().numpy())
        lungs.append(((logits[:, LEFT_LUNG] > 0) | (logits[:, RIGHT_LUNG] > 0)).cpu().numpy())
    return np.concatenate(hearts), np.concatenate(lungs)


def ctr_of(heart: np.ndarray, lungs: np.ndarray) -> float:
    """CTR = horizontal extent of the heart / of both lungs, as in data.heart_chest_widths. Columns
    with fewer than 3 mask pixels are ignored, so a few stray pixels cannot widen an extent."""
    heart_cols, lung_cols = np.flatnonzero(heart.sum(0) >= 3), np.flatnonzero(lungs.sum(0) >= 3)
    if len(heart_cols) == 0 or len(lung_cols) == 0:
        return np.nan
    return (heart_cols[-1] - heart_cols[0]) / (lung_cols[-1] - lung_cols[0])


def val_films() -> pd.DataFrame:
    """Converted val films with their labels and CheXmask CTR."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta = meta[meta["available"] & (meta["split"] == "val")]
    ctr = pd.read_csv(OUTPUTS / "ctr_nih.csv", usecols=["image", "ctr", "good_mask"])
    return meta.merge(ctr, on="image")


def used_classifiers() -> dict[str, dict[str, float]]:
    """Per finding, the classifiers that passed validate (AUROC >= 0.80 on real films) and their
    thresholds. With none for a finding, the edits of that finding cannot be checked: stop."""
    table = pd.read_csv(RESULTS / "edit_classifiers.csv")
    used = {f: dict(zip(g["classifier"], g["threshold"])) for f, g in table[table["used"]].groupby("finding")}
    missing = set(LABELS) - set(used)
    if missing:
        raise SystemExit(f"No classifier reaches AUROC {MIN_AUROC} for {missing}: decide before editing.")
    return used


def validate(args: argparse.Namespace) -> None:
    """Val films with each finding and normal films (one per patient): classifier AUROC and threshold
    per finding; segmenter CTR against the CheXmask CTR on the same films."""
    films = val_films()
    groups = [films[films[col] == 1].drop_duplicates("patient_id") for col in ["no_finding", *LABELS]]
    films = pd.concat(groups).drop_duplicates("image").reset_index(drop=True)
    images = [np.asarray(Image.open(DATA / "nih512" / f).convert("L")) for f in films["file"]]
    print(f"{len(films)} val films: {', '.join(f'{int(films[c].sum())} {c}' for c in ['no_finding', *LABELS])}")

    scores = pd.concat([films[["image", "patient_id", "no_finding", *LABELS, "ctr", "good_mask"]],
                        classify(load_classifiers(), images)], axis=1)
    rows = []
    for name in CLASSIFIERS:
        for finding in LABELS:
            d = scores[(scores[finding] == 1) | (scores["no_finding"] == 1)]
            d = d.assign(label=d[finding], score=d[f"{name}:{finding}"])
            point, low, high = auroc_ci(d, score="score", seed=args.seed)
            threshold = youden(d.loc[d["label"] == 1, "score"].to_numpy(), d.loc[d["label"] == 0, "score"].to_numpy())
            rows.append({"classifier": name, "finding": finding, "n_with": int(d["label"].sum()),
                         "n_normal": int((d["label"] == 0).sum()), "auroc": point, "ci_low": low,
                         "ci_high": high, "threshold": threshold})
    table = pd.DataFrame(rows).round(3)
    table["used"] = table["auroc"] >= MIN_AUROC
    table.to_csv(RESULTS / "edit_classifiers.csv", index=False)
    print(table.to_string(index=False))

    hearts, lungs = segment(load_segmenter(), images)
    scores["seg_ctr"] = [ctr_of(h, l) for h, l in zip(hearts, lungs)]
    scores.round(4).to_csv(OUTPUTS / "checks_val.csv", index=False)
    d = scores[scores["good_mask"] & ((scores["cardiomegaly"] == 1) | (scores["no_finding"] == 1))]
    both = d.dropna(subset=["seg_ctr"])
    summary = pd.DataFrame([{
        "n_films": len(d), "segmenter_failures": int(d["seg_ctr"].isna().sum()),
        "pearson_r": np.corrcoef(both["ctr"], both["seg_ctr"])[0, 1],
        "mean_abs_diff": (both["ctr"] - both["seg_ctr"]).abs().mean(),
        "agree_above_0.5": ((both["ctr"] > 0.5) == (both["seg_ctr"] > 0.5)).mean(),
        "auroc_chexmask": auroc(both.loc[both["cardiomegaly"] == 1, "ctr"].to_numpy(),
                                both.loc[both["no_finding"] == 1, "ctr"].to_numpy()),
        "auroc_segmenter": auroc(both.loc[both["cardiomegaly"] == 1, "seg_ctr"].to_numpy(),
                                 both.loc[both["no_finding"] == 1, "seg_ctr"].to_numpy()),
    }]).round(3)
    summary.to_csv(RESULTS / "segmenter_ctr.csv", index=False)
    print(summary.to_string(index=False))
    plot_ctr_agreement(both, RESULTS / "segmenter_ctr.png")


def plot_ctr_agreement(d: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    for label, color in [("no_finding", MUTED), ("cardiomegaly", ACCENT)]:
        g = d[d[label] == 1]
        ax.scatter(g["ctr"], g["seg_ctr"], s=8, color=color, alpha=0.6, linewidths=0,
                   label="Cardiomegaly" if label == "cardiomegaly" else "No finding")
    ax.plot([0.25, 0.8], [0.25, 0.8], color=INK_2, linewidth=0.8)
    ax.axvline(0.5, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(0.5, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xlim(0.25, 0.8)
    ax.set_ylim(0.25, 0.8)
    ax.set_xlabel("CTR from CheXmask", color=INK_2)
    ax.set_ylabel("CTR from the segmenter", color=INK_2)
    ax.set_title("Segmenter CTR vs CheXmask CTR, val films", loc="left", color=INK, fontsize=11)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="Classifier AUROC and segmenter CTR on real val films")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=validate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
