"""Stage 3: does MedGemma look at the image?

    python -m cxr.audit smoke                     # day 1: P(yes) on val films -> AUROC
    python -m cxr.audit run --model 4b            # day 3: thresholds on val, then the test pairs
    python -m cxr.audit run --model 1.5-4b        #        same for MedGemma 1.5
    python -m cxr.audit run --model 4b --limit 4  #        quick test on a few finished pairs
    python -m cxr.audit decode --model 1.5-4b     # greedy answers against the score
    python -m cxr.audit boscheck                  # effect of the second <bos> used until 2026-09-30

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


def prompt_for(processor, question: str) -> str:
    """The audit's prompt: one image, the question and "Answer yes or no.", no system message."""
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": f"{question} Answer yes or no."}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def encode(processor, prompt: str, images: list, double_bos: bool = False) -> dict:
    """Tokenize a batch with the same prompt. The chat template's text already starts with <bos>, so
    the tokenizer must not add another (add_special_tokens=False). Until 2026-09-30 it did: every
    input started with two <bos> tokens. double_bos=True reproduces that, to measure its effect."""
    return processor(text=[prompt] * len(images), images=[[im] for im in images], return_tensors="pt",
                     padding=True, add_special_tokens=double_bos)


@torch.inference_mode()
def p_yes(model, processor, images: list, question: str, batch_size: int, double_bos: bool = False) -> np.ndarray:
    """Per image, at the first answer position: the log-odds d = logsumexp(yes-token logits) -
    logsumexp(no-token logits), P(yes) = sigmoid(d), and the mass p(yes or no tokens) (near 1 when
    the model really starts its answer with yes or no). Columns: d, P(yes), mass.
    The model's logits are bf16, which rounds them in steps of about 0.1 at the magnitudes seen here
    and made many scores tie; so the logits of the answer tokens are recomputed in float32 from the
    input of the output layer (captured by a hook) and that layer's rows for those tokens. The mass
    only needs bf16."""
    prompt = prompt_for(processor, question)
    ids = answer_ids(processor.tokenizer)
    answer, n_yes = ids["yes"] + ids["no"], len(ids["yes"])
    bos = processor.tokenizer.bos_token_id
    weight = model.lm_head.weight[answer].float()  # output-layer rows of the yes and no tokens
    captured = {}
    hook = model.lm_head.register_forward_hook(lambda module, args, output: captured.update(h=args[0]))
    scores = []
    try:
        for i in range(0, len(images), batch_size):
            inputs = encode(processor, prompt, images[i:i + batch_size], double_bos).to(model.device, dtype=torch.bfloat16)
            if i == 0:  # exactly one <bos> (two only when reproducing the old inputs)
                n_bos = int((inputs["input_ids"][0, :2] == bos).sum())
                assert n_bos == (2 if double_bos else 1), inputs["input_ids"][0, :4].tolist()
            out = model(**inputs, logits_to_keep=1)
            z = captured["h"][:, -1].float() @ weight.T  # answer-token logits in float32 (no cap in this model)
            if i == 0:  # the float32 logits must match the model's own bf16 logits up to bf16 rounding
                gap = (z - out.logits[:, -1, answer].float()).abs().max().item()
                print(f"  float32 vs bf16 answer logits, largest difference: {gap:.3f}", flush=True)
                assert gap < 0.5, gap
            d = z[:, :n_yes].logsumexp(-1) - z[:, n_yes:].logsumexp(-1)
            mass = out.logits[:, -1].float().log_softmax(-1)[:, answer].logsumexp(-1).exp()
            scores.append(torch.stack([d, torch.sigmoid(d), mass], dim=1).cpu())
    finally:
        hook.remove()
    return torch.cat(scores).numpy()


def auroc_ci(df: pd.DataFrame, score: str = "log_odds", n_boot: int = 1000,
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


def smoke_films(n: int, seed: int) -> dict[str, pd.DataFrame]:
    """Per finding: n val films with it (label 1) and the same n normal films (label 0), one per patient."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    val = meta[meta["available"] & (meta["split"] == "val")]
    normals = val[val["no_finding"] == 1].drop_duplicates("patient_id")
    normals = normals.sample(n=min(n, len(normals)), random_state=seed)
    films = {}
    for finding in QUESTIONS:
        positives = val[val[finding] == 1].drop_duplicates("patient_id")
        positives = positives.sample(n=min(n, len(positives)), random_state=seed)
        films[finding] = pd.concat([positives.assign(label=1), normals.assign(label=0)], ignore_index=True)
    return films


def boscheck(args: argparse.Namespace) -> None:
    """How much did the second <bos> change the scores? The smoke-test films scored with one <bos>
    (the fixed input) and with two (the input used before 2026-09-30), both models, both phrasings
    -> results/bos_check.csv."""
    rows = []
    for tag, model_id in MODELS.items():
        model, processor = load_medgemma(model_id)
        for finding, films in smoke_films(args.n, args.seed).items():
            images = [Image.open(DATA / "nih512" / f).convert("RGB") for f in films["file"]]
            pos, neg = films["label"].to_numpy() == 1, films["label"].to_numpy() == 0
            for k, question in enumerate(QUESTIONS[finding]):
                one = p_yes(model, processor, images, question, args.batch_size)
                two = p_yes(model, processor, images, question, args.batch_size, double_bos=True)
                rows.append({"model": tag, "finding": finding, "phrasing": k, "n": len(films),
                             "auroc_one_bos": auroc(one[pos, 0], one[neg, 0]), "auroc_two_bos": auroc(two[pos, 0], two[neg, 0]),
                             "mean_log_odds_change": (one[:, 0] - two[:, 0]).mean(),
                             "max_abs_log_odds_change": np.abs(one[:, 0] - two[:, 0]).max(),
                             "share_same_answer_at_05": ((one[:, 0] > 0) == (two[:, 0] > 0)).mean(),
                             "median_mass_one_bos": np.median(one[:, 2]), "median_mass_two_bos": np.median(two[:, 2])})
        del model
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "bos_check.csv", index=False)
    print(table.round(3).to_string(index=False))


def smoke(args: argparse.Namespace) -> None:
    """Per finding: n val films with it and n normal films (one film per patient), both phrasings."""
    table = OUTPUTS / "medgemma_smoke.csv"
    if table.exists() and "log_odds" in pd.read_csv(table, nrows=0).columns:  # finished work is skipped
        scores = pd.read_csv(table)
    else:  # (a table from before the log-odds re-score has no log_odds column: scored again)
        model, processor = load_medgemma()
        parts = []
        for finding, films in smoke_films(args.n, args.seed).items():
            images = [Image.open(DATA / "nih512" / f).convert("RGB") for f in films["file"]]
            questions = QUESTIONS[finding]
            for k, question in enumerate(questions):
                s = p_yes(model, processor, images, question, args.batch_size)
                parts.append(films[["image", "patient_id", "label"]].assign(
                    finding=finding, phrasing=k, log_odds=s[:, 0], p_yes=s[:, 1], yes_no_mass=s[:, 2]))
        scores = pd.concat(parts, ignore_index=True)
        table.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(table, index=False)  # unrounded: rounding only in display tables

    rows = []
    for (finding, phrasing), g in scores.groupby(["finding", "phrasing"]):
        point, low, high = auroc_ci(g, score="log_odds", seed=args.seed)
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
    """Log-odds, P(yes) and yes/no mass for each image and both phrasings of the finding's question."""
    parts = []
    for k, question in enumerate(QUESTIONS[finding]):
        s = p_yes(model, processor, [im.convert("RGB") for im in images], question, batch_size)
        parts.append(pd.DataFrame({"row": range(len(images)), "phrasing": k, "log_odds": s[:, 0],
                                   "p_yes": s[:, 1], "yes_no_mass": s[:, 2]}))
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

    rows = []
    for finding, films in smoke_films(args.n_val, args.seed).items():
        s = score_images(model, processor, [Image.open(DATA / "nih512" / f) for f in films["file"]],
                         finding, args.batch_size)
        s = s.merge(films[["label"]].reset_index(names="row"), on="row")
        for phrasing, g in s.groupby("phrasing"):
            pos, neg = g.loc[g["label"] == 1, "log_odds"].to_numpy(), g.loc[g["label"] == 0, "log_odds"].to_numpy()
            t = youden(pos, neg)  # fitted on the log-odds: P(yes) saturates and would tie
            rows.append({"finding": finding, "phrasing": phrasing, "question": QUESTIONS[finding][phrasing],
                         "threshold_log_odds": t, "threshold": 1 / (1 + np.exp(-t)), "auroc_val": auroc(pos, neg)})
    thresholds = pd.DataFrame(rows)
    if not args.limit:
        thresholds.to_csv(thresholds_file, index=False)  # unrounded: a threshold saved as 0.0 emptied a stratum
    print(thresholds.round(4).to_string(index=False), flush=True)

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
    table.to_csv(test_file, index=False)  # unrounded

    # Short summary of the primary flip rule on the log-odds (the full analysis is notebooks/03_audit)
    wide = table.pivot_table(index=["finding", "image", "phrasing"], columns="variant", values="log_odds").reset_index()
    if validity is not None:
        wide = wide.merge(validity, on=["finding", "image"]).query("valid == True")
    wide = wide.merge(thresholds[["finding", "phrasing", "threshold_log_odds"]], on=["finding", "phrasing"])
    eligible = wide[wide["original"] < wide["threshold_log_odds"]]
    summary = eligible.groupby(["finding", "phrasing"]).apply(lambda g: pd.Series({
        "n_eligible": len(g), "flip_rate_edit": (g["edit"] > g["threshold_log_odds"]).mean(),
        "flip_rate_sham": (g["sham"] > g["threshold_log_odds"]).mean()}), include_groups=False).round(3)
    print(f"Primary flip rule, {'valid pairs' if validity is not None else 'all pairs (no validity yet)'}:")
    print(summary.to_string())


@torch.inference_mode()
def decode(args: argparse.Namespace) -> None:
    """Does the score say what the model answers? For the first n dose films (test normals): the
    original, its growth-0 and growth-0.06 cardiomegaly edits (dose run) and its blank. MedGemma
    answers "Is there cardiomegaly in this image?" by greedy decoding; the first word of the answer
    (yes / no / other) is compared with the score read at the first answer position
    -> results/decode_<model>.csv (MedGemma 1.5 rarely starts with yes or no on real films)."""
    model, processor = load_medgemma(MODELS[args.model])
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv").set_index("image")
    dose = OUTPUTS / "dose" / "cardiomegaly"
    names = sorted(p.name for p in (dose / "growth0.00").glob("*.png"))[: args.n]
    images, rows = [], []
    for name in names:
        original = Image.open(DATA / "nih512" / meta.loc[name, "file"]).convert("L")
        for condition, im in [("original", original),
                              ("growth 0", Image.open(dose / "growth0.00" / name).convert("L")),
                              ("growth 0.06", Image.open(dose / "growth0.06" / name).convert("L")),
                              ("blank", blank(original))]:
            images.append(im.convert("RGB"))
            rows.append({"image": name, "condition": condition})
    question = QUESTIONS["cardiomegaly"][0]
    prompt = prompt_for(processor, question)
    texts = []
    for i in range(0, len(images), args.batch_size):  # same prompt in a batch, so no padding
        inputs = encode(processor, prompt, images[i:i + args.batch_size]).to(model.device, dtype=torch.bfloat16)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts += processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    s = p_yes(model, processor, images, question, args.batch_size)
    table = pd.DataFrame(rows).assign(text=texts, log_odds=s[:, 0], p_yes=s[:, 1], yes_no_mass=s[:, 2])
    text = table["text"].str.lower()
    first = text.str.extract(r"([a-z]+)", expand=False)  # the answer's first word
    table["answer"] = first.where(first.isin(["yes", "no"]), "other")
    # MedGemma 1.5 often answers in a sentence ("Based on the chest X-ray image, there is no ..."):
    # the first standalone yes or no anywhere in the answer, as a second reading
    table["answer_in_text"] = text.str.extract(r"\b(yes|no)\b", expand=False).fillna("other")
    table.to_csv(RESULTS / f"decode_{args.model}.csv", index=False)
    agree = lambda g, col: ((g[col] == "yes") == (g["p_yes"] > 0.5))[g[col] != "other"].mean()
    summary = table.groupby("condition", sort=False).apply(lambda g: pd.Series({
        "n": len(g), "share_answer_yes_or_no": (g["answer"] != "other").mean(),
        "share_answer_yes": (g["answer"] == "yes").mean(), "share_p_yes_above_05": (g["p_yes"] > 0.5).mean(),
        "agreement_where_yes_or_no": agree(g, "answer"),
        "share_yes_or_no_in_text": (g["answer_in_text"] != "other").mean(),
        "agreement_in_text": agree(g, "answer_in_text"),
        "median_yes_no_mass": g["yes_no_mass"].median()}), include_groups=False)
    print(summary.round(3).to_string())
    print("Most common first words of the other answers:", first[table["answer"] == "other"].value_counts().head(5).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("decode", help="Greedy answers against the score, per condition")
    g.add_argument("--model", choices=list(MODELS), required=True)
    g.add_argument("--n", type=int, default=100, help="Dose films (each gives 4 images)")
    g.add_argument("--max-new-tokens", type=int, default=32)
    g.add_argument("--batch-size", type=int, default=16)
    g.set_defaults(func=decode)
    b = sub.add_parser("boscheck", help="Scores with one vs two <bos> tokens on the smoke films, both models")
    b.add_argument("--n", type=int, default=100, help="Films with the finding, and normal films, per finding")
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=boscheck)
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
