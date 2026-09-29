"""Stage 3: does MedGemma look at the image?

    python -m cxr.audit smoke                     # day 1: P(yes) on val films -> AUROC
    python -m cxr.audit run --model 4b            # day 3: thresholds on val, then the test pairs
    python -m cxr.audit run --model 1.5-4b        #        same for MedGemma 1.5
    python -m cxr.audit run --model 4b --limit 4  #        quick test on a few finished pairs

MedGemma 4B (google/medgemma-4b-it, bf16) gets one film and a yes/no question. The score is
P("yes") read from the next-token logits (yes vs no tokens): continuous, no text parsing. A finding
is worth auditing only if this score separates films with the finding from normal films (AUROC
clearly above 0.5): grounding is meaningless if the model never sees the finding. The question
follows the MedGemma report's chest X-ray prompt (Table A7), without its "write out your argument"
part since we read the first answer token, and with no system message, as in the report.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import DATA, OUTPUTS, RESULTS
from .data import auroc

MODEL_ID = "google/medgemma-4b-it"
MODELS = {"4b": "google/medgemma-4b-it", "1.5-4b": "google/medgemma-1.5-4b-it"}  # same code for both
QUESTIONS = {  # two phrasings per finding; the first is the MedGemma report's wording
    "cardiomegaly": ["Is there cardiomegaly in this image?",
                     "Is the heart enlarged in this image?"],
    "effusion": ["Is there pleural effusion in this image?",
                 "Is there fluid in the pleural space in this image?"],
}


def load_medgemma(model_id: str = MODEL_ID):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    return model.eval(), AutoProcessor.from_pretrained(model_id)


def answer_ids(tokenizer) -> dict[str, list[int]]:
    """Ids of the one-token answers 'Yes', 'yes', ' Yes', ' yes' (and the same for no)."""
    ids = {}
    for answer in ["yes", "no"]:
        variants = [answer, answer.capitalize(), " " + answer, " " + answer.capitalize()]
        tokens = [tokenizer.encode(v, add_special_tokens=False) for v in variants]
        ids[answer] = sorted({t[0] for t in tokens if len(t) == 1})
    return ids


@torch.inference_mode()
def p_yes(model, processor, images: list, question: str, batch_size: int) -> np.ndarray:
    """Per image: P(yes) = p(yes tokens) / p(yes or no tokens) at the first answer position, and the
    mass p(yes or no tokens) itself (near 1 when the model really answers yes/no)."""
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": f"{question} Answer yes or no."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = answer_ids(processor.tokenizer)
    scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(text=[prompt] * len(batch), images=[[im] for im in batch], return_tensors="pt",
                           padding=True).to(model.device, dtype=torch.bfloat16)
        logp = model(**inputs, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        yes, no = logp[:, ids["yes"]].logsumexp(-1), logp[:, ids["no"]].logsumexp(-1)
        scores.append(torch.stack([torch.sigmoid(yes - no), yes.exp() + no.exp()], dim=1).cpu())
    return torch.cat(scores).numpy()


def auroc_ci(df: pd.DataFrame, score: str = "p_yes", n_boot: int = 1000,
             seed: int = 0) -> tuple[float, float, float]:
    """AUROC of the score column against label, with a 95% bootstrap CI that resamples patients."""
    df = df.reset_index(drop=True)
    point = auroc(df.loc[df["label"] == 1, score].to_numpy(), df.loc[df["label"] == 0, score].to_numpy())
    patients = [g.index.to_numpy() for _, g in df.groupby("patient_id")]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = df.loc[np.concatenate([patients[i] for i in rng.integers(len(patients), size=len(patients))])]
        pos, neg = s.loc[s["label"] == 1, score].to_numpy(), s.loc[s["label"] == 0, score].to_numpy()
        if len(pos) and len(neg):
            boots.append(auroc(pos, neg))
    low, high = np.percentile(boots, [2.5, 97.5])
    return point, low, high


def youden(positives: np.ndarray, negatives: np.ndarray) -> float:
    """The threshold that best separates the two groups: highest sensitivity + specificity - 1."""
    thresholds = np.unique(np.concatenate([positives, negatives]))
    j = [(positives >= t).mean() - (negatives >= t).mean() for t in thresholds]
    return float(thresholds[int(np.argmax(j))])


def smoke(args: argparse.Namespace) -> None:
    """Per finding: n val films with it and n normal films (one film per patient), both phrasings."""
    table = OUTPUTS / "medgemma_smoke.csv"
    if table.exists():  # finished work is skipped: delete the file to score again
        scores = pd.read_csv(table)
    else:
        meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
        val = meta[meta["available"] & (meta["split"] == "val")]
        normals = val[val["no_finding"] == 1].drop_duplicates("patient_id")
        normals = normals.sample(n=min(args.n, len(normals)), random_state=args.seed)
        model, processor = load_medgemma()
        parts = []
        for finding, questions in QUESTIONS.items():
            positives = val[val[finding] == 1].drop_duplicates("patient_id")
            positives = positives.sample(n=min(args.n, len(positives)), random_state=args.seed)
            films = pd.concat([positives.assign(label=1), normals.assign(label=0)], ignore_index=True)
            images = [Image.open(DATA / "nih512" / f).convert("RGB") for f in films["file"]]
            for k, question in enumerate(questions):
                s = p_yes(model, processor, images, question, args.batch_size)
                parts.append(films[["image", "patient_id", "label"]].assign(
                    finding=finding, phrasing=k, p_yes=s[:, 0], yes_no_mass=s[:, 1]))
        scores = pd.concat(parts, ignore_index=True)
        table.parent.mkdir(parents=True, exist_ok=True)
        scores.round(5).to_csv(table, index=False)

    rows = []
    for (finding, phrasing), g in scores.groupby(["finding", "phrasing"]):
        point, low, high = auroc_ci(g, seed=args.seed)
        rows.append({"finding": finding, "question": QUESTIONS[finding][phrasing],
                     "n_with": int((g["label"] == 1).sum()), "n_normal": int((g["label"] == 0).sum()),
                     "auroc": point, "ci_low": low, "ci_high": high,
                     "mean_p_yes_with": g.loc[g["label"] == 1, "p_yes"].mean(),
                     "mean_p_yes_normal": g.loc[g["label"] == 0, "p_yes"].mean(),
                     "min_yes_no_mass": g["yes_no_mass"].min()})
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(RESULTS / "medgemma_smoke.csv", index=False)
    print(summary.to_string(index=False))


def blank(film: Image.Image) -> Image.Image:
    """Control: uniform gray at the film's mean (as CORAL): nothing to see."""
    return Image.new("L", film.size, int(np.asarray(film).mean()))


def shuffled(film: Image.Image, seed: int, patch: int = 32) -> Image.Image:
    """Control: the film cut into 32-px patches (a 16 x 16 grid at 512 px) put back in a random
    order, fixed per film: same pixels, no anatomy."""
    a = np.asarray(film)
    n = a.shape[0] // patch
    tiles = a.reshape(n, patch, n, patch).swapaxes(1, 2).reshape(n * n, patch, patch)
    tiles = tiles[np.random.default_rng(seed).permutation(n * n)]
    return Image.fromarray(tiles.reshape(n, n, patch, patch).swapaxes(1, 2).reshape(n * patch, n * patch))


def score_images(model, processor, images: list, finding: str, batch_size: int) -> pd.DataFrame:
    """P(yes) and yes/no mass for each image and both phrasings of the finding's question."""
    parts = []
    for k, question in enumerate(QUESTIONS[finding]):
        s = p_yes(model, processor, [im.convert("RGB") for im in images], question, batch_size)
        parts.append(pd.DataFrame({"row": range(len(images)), "phrasing": k, "p_yes": s[:, 0],
                                   "yes_no_mass": s[:, 1]}))
    return pd.concat(parts, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    """The day-3 audit for one model.
    1. Thresholds (pre-registered flip rule): on val originals, n films with each finding and n
       normal films (one per patient, the smoke-test films), the Youden threshold per finding and
       phrasing -> results/audit_<model>_thresholds.csv.
    2. Test pairs: for every source film with its edit and sham on disk, six images: original,
       edit, sham, blank, shuffled, and another patient's real test film with the finding (the
       "real finding" group). Scores -> results/audit_<model>_test.csv (one row per image and
       phrasing, with the pair's validity from outputs/pairs_test.csv)."""
    tag = args.model
    thresholds_file, test_file = RESULTS / f"audit_{tag}_thresholds.csv", RESULTS / f"audit_{tag}_test.csv"
    if args.limit:
        test_file = OUTPUTS / f"audit_{tag}_test_limit{args.limit}.csv"
    model, processor = load_medgemma(MODELS[tag])
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta = meta[meta["available"]]

    val = meta[meta["split"] == "val"]
    normals = val[val["no_finding"] == 1].drop_duplicates("patient_id")
    normals = normals.sample(n=min(args.n_val, len(normals)), random_state=args.seed)
    rows = []
    for finding in QUESTIONS:
        positives = val[val[finding] == 1].drop_duplicates("patient_id")
        positives = positives.sample(n=min(args.n_val, len(positives)), random_state=args.seed)
        films = pd.concat([positives.assign(label=1), normals.assign(label=0)], ignore_index=True)
        s = score_images(model, processor, [Image.open(DATA / "nih512" / f) for f in films["file"]],
                         finding, args.batch_size)
        s = s.merge(films[["label"]].reset_index(names="row"), on="row")
        for phrasing, g in s.groupby("phrasing"):
            rows.append({"finding": finding, "phrasing": phrasing, "question": QUESTIONS[finding][phrasing],
                         "threshold": youden(g.loc[g["label"] == 1, "p_yes"].to_numpy(),
                                             g.loc[g["label"] == 0, "p_yes"].to_numpy()),
                         "auroc_val": auroc(g.loc[g["label"] == 1, "p_yes"].to_numpy(),
                                            g.loc[g["label"] == 0, "p_yes"].to_numpy())})
    thresholds = pd.DataFrame(rows).round(4)
    if not args.limit:
        thresholds.to_csv(thresholds_file, index=False)
    print(thresholds.to_string(index=False), flush=True)

    pairs_file = OUTPUTS / "pairs_test.csv"
    validity = (pd.read_csv(pairs_file).query("kind == 'edit'")[["finding", "image", "valid"]]
                if pairs_file.exists() else None)
    test = meta[meta["split"] == "test"]
    parts = []
    for finding in QUESTIONS:
        pair_dir = OUTPUTS / "pairs" / "test" / finding
        names = sorted(p.name for p in (pair_dir / "sham").glob("*.png") if (pair_dir / "edit" / p.name).exists())
        if args.limit:
            names = names[: args.limit]
        if not names:
            print(f"{finding}: no finished pairs yet, skipped", flush=True)
            continue
        sources = test.set_index("image").loc[names].reset_index()
        real = test[test[finding] == 1]
        rng = np.random.default_rng(args.seed)
        images, info = [], []
        for i, film in sources.iterrows():
            original = Image.open(DATA / "nih512" / film["file"]).convert("L")
            others = real[real["patient_id"] != film["patient_id"]]
            other = others.iloc[rng.integers(len(others))]
            # .convert() loads each image and closes its file: thousands of lazy Image.open() handles
            # hit the open-file limit (the first full run failed on 2026-09-29).
            variants = {"original": original,
                        "edit": Image.open(pair_dir / "edit" / film["image"]).convert("L"),
                        "sham": Image.open(pair_dir / "sham" / film["image"]).convert("L"),
                        "blank": blank(original),
                        "shuffled": shuffled(original, seed=args.seed + i),
                        "real_finding": Image.open(DATA / "nih512" / other["file"]).convert("L")}
            for variant, im in variants.items():
                images.append(im)
                info.append({"finding": finding, "image": film["image"], "patient_id": film["patient_id"],
                             "variant": variant, "shown_image": other["image"] if variant == "real_finding" else film["image"]})
        print(f"{finding}: {len(sources)} source films, {len(images)} images to score", flush=True)
        s = score_images(model, processor, images, finding, args.batch_size)
        parts.append(s.merge(pd.DataFrame(info).reset_index(names="row"), on="row").drop(columns="row"))
    table = pd.concat(parts, ignore_index=True)
    if validity is not None:
        table = table.merge(validity, on=["finding", "image"], how="left")
    table.round(5).to_csv(test_file, index=False)

    # Short summary of the primary flip rule (the full analysis, with CIs, is notebooks/03_audit)
    wide = table.pivot_table(index=["finding", "image", "phrasing"], columns="variant", values="p_yes").reset_index()
    if validity is not None:
        wide = wide.merge(validity, on=["finding", "image"]).query("valid == True")
    wide = wide.merge(thresholds[["finding", "phrasing", "threshold"]], on=["finding", "phrasing"])
    eligible = wide[wide["original"] < wide["threshold"]]
    summary = eligible.groupby(["finding", "phrasing"]).apply(lambda g: pd.Series({
        "n_eligible": len(g), "flip_rate_edit": (g["edit"] > g["threshold"]).mean(),
        "flip_rate_sham": (g["sham"] > g["threshold"]).mean()}), include_groups=False).round(3)
    print(f"Primary flip rule, {'valid pairs' if validity is not None else 'all pairs (no validity yet)'}:")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Day-3 audit of one model on the test pairs and controls")
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--n-val", type=int, default=100, help="Val films with the finding (and normal) for thresholds")
    r.add_argument("--limit", type=int, default=0, help="Only the first N pairs per finding (quick test)")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=run)
    s = sub.add_parser("smoke", help="P(yes) on real val films, AUROC per finding and phrasing")
    s.add_argument("--n", type=int, default=100, help="Films with the finding, and normal films, per finding")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=smoke)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
