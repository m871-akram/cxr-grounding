"""Stage 3: does MedGemma look at the image? Day 1 has only the smoke test:

    python -m cxr.audit smoke    # P(yes) on val films with and without each finding -> AUROC

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
QUESTIONS = {  # two phrasings per finding; the first is the MedGemma report's wording
    "cardiomegaly": ["Is there cardiomegaly in this image?",
                     "Is the heart enlarged in this image?"],
    "effusion": ["Is there pleural effusion in this image?",
                 "Is there fluid in the pleural space in this image?"],
}


def load_medgemma():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    return model.eval(), AutoProcessor.from_pretrained(MODEL_ID)


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


def auroc_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC of p_yes against label, with a 95% bootstrap CI that resamples patients."""
    df = df.reset_index(drop=True)
    point = auroc(df.loc[df["label"] == 1, "p_yes"].to_numpy(), df.loc[df["label"] == 0, "p_yes"].to_numpy())
    patients = [g.index.to_numpy() for _, g in df.groupby("patient_id")]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = df.loc[np.concatenate([patients[i] for i in rng.integers(len(patients), size=len(patients))])]
        pos, neg = s.loc[s["label"] == 1, "p_yes"].to_numpy(), s.loc[s["label"] == 0, "p_yes"].to_numpy()
        if len(pos) and len(neg):
            boots.append(auroc(pos, neg))
    low, high = np.percentile(boots, [2.5, 97.5])
    return point, low, high


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="P(yes) on real val films, AUROC per finding and phrasing")
    s.add_argument("--n", type=int, default=100, help="Films with the finding, and normal films, per finding")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=smoke)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
