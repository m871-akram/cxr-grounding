"""Checks after the first audit (PLAN.md, 2026-09-29): does MedGemma follow the anatomy?

    python -m cxr.dose run       # dose-response edits, blur control, measures, MedGemma scores
    python -m cxr.dose gallery   # effusion shams listed in results/effusion_sham_gallery.csv

Cardiomegaly dose-response: the first 100 test source films of the audit (same films, same seeds)
edited with the grown-heart mask at 6 strengths, from growth 0 (the heart's own outline) to the
audit's 0.30 (reused, not regenerated). Every edit is measured (classifier, segmenter CTR) and
scored by both MedGemma models whatever its validity, and so are all real test films: P(yes)
against the measured CTR, for real and edited films.
Effusion evidence-loss control: the lung-base mask of the first 100 effusion source films blurred
(Gaussian, pasted back with the same feathered border, no RadEdit) at two strengths. If P(yes)
rises with blur, lost normal detail is enough to move the answer toward "yes".
Re-scored images (originals, 0.30 edits, effusion shams) are compared with the audit's scores,
which checks that the scores are deterministic.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from . import DATA, OUTPUTS, RESULTS
from .audit import MODELS, load_medgemma, score_images
from .edit import (GENERATE, edit_film, gallery, grown_heart_mask, landmarks_512, load_films, load_radedit,
                   paste_back, tuning_mask)

GROWTHS = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30]  # 0.30 = the audit's additions
BLURS = [3.0, 8.0]  # Gaussian sigma (px at 512) of the evidence-loss control
DOSE = OUTPUTS / "dose"


def audit_sources(finding: str, n: int, seed: int) -> pd.DataFrame:
    """The first n source films of the audit for this finding, in the audit's order: the same
    selection as edit.generate, so film k was edited with seed + k."""
    films = load_films("test").merge(pd.read_csv(OUTPUTS / "ctr_nih.csv", usecols=["image", "ctr"]), on="image")
    normals = films[films["no_finding"] == 1]
    counts = {"cardiomegaly": 380, "effusion": 600}  # the audit's --n-cardiomegaly / --n-effusion
    pool = normals[normals["ctr"] < 0.5] if finding == "cardiomegaly" else normals
    chosen = pool.drop_duplicates("patient_id")
    chosen = chosen.sample(n=min(counts[finding], len(chosen)), random_state=seed)
    return chosen.head(n).reset_index(drop=True)


def blurred(original: Image.Image, mask: np.ndarray, sigma: float, feather: float) -> np.ndarray:
    """The film blurred inside the mask only, with the same feathered paste-back as the edits."""
    before = np.asarray(original.convert("L"), dtype=float)
    blur = np.asarray(original.convert("L").filter(ImageFilter.GaussianBlur(sigma)), dtype=float)
    return np.clip(paste_back(before, blur, mask, feather), 0, 255).round().astype(np.uint8)


def make_images(args, cardio: pd.DataFrame, effusion: pd.DataFrame) -> list[dict]:
    """Dose edits and blurred films on disk (finished ones skipped); one row per image."""
    prompt, _, skip = GENERATE["cardiomegaly"]
    pipe, rows = None, []
    for k, film in cardio.iterrows():
        for growth in GROWTHS:
            out = DOSE / "cardiomegaly" / f"growth{growth:.2f}" / film["image"]
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                if growth == 0.30:  # the audit's edit of this film: same mask, prompt and seed
                    shutil.copy(OUTPUTS / "pairs" / "test" / "cardiomegaly" / "edit" / film["image"], out)
                else:
                    if pipe is None:
                        pipe = load_radedit()
                        torch.backends.cuda.matmul.allow_tf32 = True
                    original = Image.open(DATA / "nih512" / film["file"])
                    mask = grown_heart_mask(landmarks_512(film), growth)
                    Image.fromarray(edit_film(pipe, original, mask, prompt, args.guidance, skip, args.steps,
                                              args.seed + k, args.feather)).save(out)
            rows.append({"group": "dose", "finding": "cardiomegaly", "image": film["image"],
                         "patient_id": film["patient_id"], "level": growth, "path": out})
        if (k + 1) % 20 == 0:
            print(f"  cardiomegaly dose: {k + 1}/{len(cardio)} films", flush=True)
    del pipe
    torch.cuda.empty_cache()

    _, (kind, param), _ = GENERATE["effusion"]
    for _, film in effusion.iterrows():
        mask = tuning_mask(landmarks_512(film), kind, param)
        original = Image.open(DATA / "nih512" / film["file"])
        for sigma in BLURS:
            out = DOSE / "effusion_blur" / f"sigma{sigma:g}" / film["image"]
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(blurred(original, mask, sigma, args.feather)).save(out)
            rows.append({"group": "blur", "finding": "effusion", "image": film["image"],
                         "patient_id": film["patient_id"], "level": sigma, "path": out})
        rows.append({"group": "sham", "finding": "effusion", "image": film["image"], "patient_id": film["patient_id"],
                     "level": np.nan, "path": OUTPUTS / "pairs" / "test" / "effusion" / "sham" / film["image"]})
        rows.append({"group": "original", "finding": "effusion", "image": film["image"],
                     "patient_id": film["patient_id"], "level": 0.0, "path": DATA / "nih512" / film["file"]})
    return rows


def measure(rows: list[dict], effusion: pd.DataFrame) -> pd.DataFrame:
    """Classifier score (-pc) and segmenter CTR of every image; for effusion rows, the aerated lung
    area inside the lung-base mask (the edit check's measure)."""
    from .checks import classify, ctr_of, load_classifiers, load_segmenter, segment

    images = [np.asarray(Image.open(r["path"]).convert("L")) for r in rows]
    p = classify({"densenet121-res224-pc": load_classifiers()["densenet121-res224-pc"]}, images)
    hearts, lungs = segment(load_segmenter(), images)
    _, (kind, param), _ = GENERATE["effusion"]
    masks = {film["image"]: tuning_mask(landmarks_512(film), kind, param) for _, film in effusion.iterrows()}
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in rows])
    table["classifier_cardiomegaly"] = p["densenet121-res224-pc:cardiomegaly"].to_numpy()
    table["classifier_effusion"] = p["densenet121-res224-pc:effusion"].to_numpy()
    table["ctr"] = [ctr_of(h, l) for h, l in zip(hearts, lungs)]
    table["lung_area_in_mask"] = [(lungs[i] & masks[r["image"]]).sum() if r["finding"] == "effusion" else np.nan
                                  for i, r in enumerate(rows)]
    return table


def run(args: argparse.Namespace) -> None:
    """Images, measures (results/dose_measures.csv), then both MedGemma models
    (results/dose_scores_<model>.csv). Each finished table is skipped on a rerun."""
    cardio = audit_sources("cardiomegaly", args.n, args.seed)
    effusion = audit_sources("effusion", args.n, args.seed)
    rows = make_images(args, cardio, effusion)
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    real = meta[meta["available"] & (meta["split"] == "test")]
    for _, film in real.iterrows():  # every real test film, for P(yes) against CTR
        rows.append({"group": "real", "finding": "cardiomegaly", "image": film["image"],
                     "patient_id": film["patient_id"], "level": np.nan, "path": DATA / "nih512" / film["file"],
                     "label_cardiomegaly": film["cardiomegaly"], "label_effusion": film["effusion"],
                     "no_finding": film["no_finding"]})
    print(f"{len(cardio)} cardiomegaly and {len(effusion)} effusion source films, {len(real)} real test films, "
          f"{len(rows)} images", flush=True)

    measures_file = RESULTS / "dose_measures.csv"
    if not measures_file.exists():
        measure(rows, effusion).round(4).to_csv(measures_file, index=False)
    print(pd.read_csv(measures_file).groupby(["group", "finding", "level"], dropna=False)[
        ["classifier_cardiomegaly", "classifier_effusion", "ctr", "lung_area_in_mask"]].median().round(3).to_string(),
        flush=True)

    for tag, model_id in MODELS.items():
        scores_file = RESULTS / f"dose_scores_{tag}.csv"
        if scores_file.exists():
            continue
        model, processor = load_medgemma(model_id)
        parts = []
        for finding in ["cardiomegaly", "effusion"]:  # each image gets its own finding's two questions
            idx = [i for i, r in enumerate(rows) if r["finding"] == finding]
            images = [Image.open(rows[i]["path"]).convert("L") for i in idx]
            s = score_images(model, processor, images, finding, args.batch_size)
            s["row"] = [idx[j] for j in s["row"]]
            parts.append(s)
            print(f"  {tag}: {finding}, {len(images)} images scored", flush=True)
        table = pd.concat(parts, ignore_index=True)
        info = pd.DataFrame([{k: v for k, v in r.items() if k in ("group", "finding", "image", "patient_id", "level")}
                             for r in rows]).reset_index(names="row")
        table.merge(info, on="row").drop(columns="row").round(5).to_csv(scores_file, index=False)
        del model
        torch.cuda.empty_cache()
    print("Median P(yes), first phrasing:")
    for tag in MODELS:
        s = pd.read_csv(RESULTS / f"dose_scores_{tag}.csv")
        s = s[s["phrasing"] == 0]
        print(tag, s.groupby(["group", "finding", "level"], dropna=False)["p_yes"].median().round(3).to_dict(), flush=True)


def make_gallery(args: argparse.Namespace) -> None:
    """Effusion shams with no measured change, as listed on the Mac (results/effusion_sham_gallery.csv):
    original, sham, |difference|, 10 per image; flipping shams first, then a random reference set."""
    listed = pd.read_csv(RESULTS / "effusion_sham_gallery.csv")
    films = load_films("test").set_index("image")
    _, (kind, param), _ = GENERATE["effusion"]
    for group, g in listed.groupby("group", sort=False):
        name = group.replace(" ", "_")
        for part, start in enumerate(range(0, len(g), 10), 1):
            chunk = g.iloc[start:start + 10]
            panels, labels = [], []
            for _, r in chunk.iterrows():
                film = films.loc[r["image"]]
                original = np.asarray(Image.open(DATA / "nih512" / film["file"]).convert("L"))
                sham = np.asarray(Image.open(OUTPUTS / "pairs" / "test" / "effusion" / "sham" / r["image"]).convert("L"))
                panels.append((original, sham, tuning_mask(landmarks_512(film), kind, param)))
                labels.append(f"{int(r['n_flips'])}/{int(r['n_scored'])} flips")
            gallery(panels, f"Effusion shams with no measured change: {group} ({part})",
                    RESULTS / f"effusion_sham_{name}_{part}.jpg", labels)
    print("galleries written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Dose-response edits, blur control, measures, MedGemma scores")
    r.add_argument("--n", type=int, default=100, help="Source films per finding")
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--guidance", type=float, default=15.0)
    r.add_argument("--feather", type=float, default=3.0)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=run)
    g = sub.add_parser("gallery", help="Effusion sham gallery from results/effusion_sham_gallery.csv")
    g.set_defaults(func=make_gallery)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
