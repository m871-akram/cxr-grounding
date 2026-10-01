"""H3 (PLAN.md, section 4): LoRA fine-tuning of MedGemma 4B on real films (R), RadEdit edits (E) or
both (RE), and the evaluation of every model through one path.

    python -m cxr.lora overfit                         # 32 examples, 50 steps: the checks of the wiring
    python -m cxr.lora train --arm R --seed 0          # one arm (R, E, RE) and seed (SEEDS)
    python -m cxr.lora evaluate --model base --set test   # the base model on the audit's test split
    python -m cxr.lora evaluate --model R0 --set fresh --final   # the fresh test set: final run only

Inputs are built with cxr.audit.encode (the scorer's own function, one <bos> asserted) and scored
with cxr.audit.p_yes (float32 log-odds), for validation and evaluation alike; the training loss uses
the model's bf16 logits of the same tokens (a difference of at most ~0.1 in d, which only changes the
gradient slightly). LoRA adapts only the language model (module paths matched by LORA_TARGETS); the
trainable parameter names are logged and checked before training. Patients: training pool, val
split, fresh test set and test split come from the patient hash and are asserted disjoint; every
training, validation and evaluation image is checked against its set. Finished runs and scores are
skipped; the fresh set is scored only with --final, by committed code, once every adapter exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import DATA, OUTPUTS, RESULTS
from .audit import QUESTIONS, answer_ids, blank, encode, load_medgemma, p_yes, prompt_for
from .data import auroc, patient_hash
from .h3 import FRESH_BELOW, MARGIN, cached_revision, commit, segment_ctr

MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_REVISION = None  # set to the pod's cached commit (printed by the test session) before training
QUESTION = QUESTIONS["cardiomegaly"][0]  # the trained question
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
ARMS, SEEDS = ["R", "E", "RE"], [0, 1, 2]
N_PER_CLASS = 1000
MAX_STEPS, EVAL_EVERY, PATIENCE, BATCH, LR, WARMUP = 750, 50, 3, 16, 1e-4, 0.05
MICRO_BATCH = 16  # one pass per step on the RTX PRO 6000; 8 gives the same batch in two passes (48-GB cards)
H3 = OUTPUTS / "h3"
RUNS = H3 / "runs"
GROWTHS = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30]
EXPECTED = {  # images of each evaluation set: a missing folder must not shrink a set silently
    "test": {"real": 1883, "edit": 600, "sham": 580, "blank": 380},
    "fresh": {"real": 2335, "edit": 600, "sham": 100, "blank": 100},
}


def medgemma_revision() -> str:
    """The pinned MedGemma commit (MEDGEMMA_REVISION), or before it is set, the cached one."""
    return MEDGEMMA_REVISION or cached_revision(MODEL_ID)


def patient_sets() -> dict[str, set]:
    """Patients of each set H3 uses, from the patient hash (the hash of the splits): test split
    (h < 0.1), val split (0.1-0.2), fresh test set (0.2-0.32) and training pool (h >= 0.32). They
    are asserted disjoint and consistent with the split column of the metadata."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta["h"] = meta["patient_id"].map(patient_hash)
    sets = {"test": meta["h"] < 0.1, "val": (meta["h"] >= 0.1) & (meta["h"] < 0.2),
            "fresh": (meta["h"] >= 0.2) & (meta["h"] < FRESH_BELOW), "pool": meta["h"] >= FRESH_BELOW}
    assert (meta.loc[sets["test"], "split"] == "test").all() and (meta.loc[sets["val"], "split"] == "val").all()
    assert (meta.loc[sets["fresh"] | sets["pool"], "split"] == "train").all()
    sets = {name: set(meta.loc[mask, "patient_id"]) for name, mask in sets.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], f"patients in both {a} and {b}"
    return sets


def labelled(d: pd.DataFrame) -> pd.DataFrame:
    """Label = segmenter CTR > 0.5; kept only when CTR > 0.525 ("yes") or CTR < 0.475 ("no"), as in
    h3 measure (images without a CTR, or exactly at 0.475 or 0.525, are dropped)."""
    d = d[(d["ctr"] > 0.5 + MARGIN) | (d["ctr"] < 0.5 - MARGIN)]
    return d.assign(label=(d["ctr"] > 0.5).astype(int))


def training_data(arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The arm's training images, its validation images and the real validation films (path, label,
    patient_id). Drawn once (seed 0), the same for every training seed: R and E take N_PER_CLASS of
    each class (fewer if a pool is short, then for every arm alike), RE takes all of R and all of E.
    The class counts must equal results/h3_supply.csv."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv").set_index("image")
    real = pd.read_csv(H3 / "ctr_real.csv")
    real["path"] = [DATA / "nih512" / f for f in real["file"]]
    edits = pd.read_csv(H3 / "ctr_edits.csv")
    edits["path"] = [H3 / s / k / f"growth{g:.2f}" / i for s, k, g, i in zip(edits["set"], edits["kind"], edits["growth"], edits["image"])]
    edits["patient_id"] = meta.loc[edits["image"], "patient_id"].to_numpy()
    real, edits = labelled(real), labelled(edits)
    supply = pd.read_csv(RESULTS / "h3_supply.csv").set_index(["arm", "set"])
    for arm_name, d in [("R", real), ("E", edits)]:
        for which, g in d.groupby("set"):
            row = supply.loc[(arm_name, which)]
            assert (int(g["label"].sum()), int((g["label"] == 0).sum())) == (row["yes"], row["no"]), (arm_name, which)
    n = min([N_PER_CLASS] + [int(((d["set"] == "train") & (d["label"] == c)).sum()) for d in (real, edits) for c in (0, 1)])
    if n < N_PER_CLASS:
        print(f"Only {n} examples in the smallest class: N = {2 * n} for every arm (PLAN.md, H3)", flush=True)
    balanced = lambda d: pd.concat([d[d["label"] == c].sample(n=n, random_state=0) for c in (1, 0)])
    r_train, e_train = balanced(real[real["set"] == "train"]), balanced(edits[edits["set"] == "train"])
    rv = real[real["set"] == "val"]
    r_val = pd.concat([rv[rv["label"] == 1], rv[rv["label"] == 0].sample(n=int(rv["label"].sum()), random_state=0)])
    e_val = edits[edits["set"] == "val"]
    train = {"R": r_train, "E": e_train, "RE": pd.concat([r_train, e_train])}[arm]
    val = e_val if arm == "E" else r_val
    sets = patient_sets()
    for d, where in [(train, "pool"), (val, "val"), (r_val, "val")]:
        assert set(d["patient_id"]) <= sets[where], f"{arm}: images from outside the {where} patients"
    return train.reset_index(drop=True), val.reset_index(drop=True), r_val.reset_index(drop=True)


def draw_hash(d: pd.DataFrame) -> str:
    """Fingerprint of a set of examples: every seed of an arm must have the same one."""
    return hashlib.sha256("\n".join(sorted(f"{p}\t{y}" for p, y in zip(d["path"], d["label"]))).encode()).hexdigest()


def lora_model(seed: int, revision: str):
    """MedGemma 4B at the given commit with LoRA on the language model only. Returns the model, the
    processor and the names of the trainable parameters (checked: none belongs to the vision encoder
    or the multimodal projector)."""
    from peft import LoraConfig, get_peft_model

    model, processor = load_medgemma(MODEL_ID, revision)
    torch.manual_seed(seed)  # the LoRA initialisation
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and not any("vision" in n or "multi_modal_projector" in n for n in names), names[:5]
    return model, processor, names


def batch_loss(model, processor, prompt: str, ids: dict, rows: pd.DataFrame) -> torch.Tensor:
    """Binary cross-entropy on the log-odds d = logsumexp(yes logits) - logsumexp(no logits) at the
    first answer position (the quantity the audit scores); no other position is trained."""
    images = [Image.open(p).convert("RGB") for p in rows["path"]]
    inputs = encode(processor, prompt, images).to(model.device, dtype=torch.bfloat16)
    z = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, ids["yes"] + ids["no"]].float()
    d = z[:, :len(ids["yes"])].logsumexp(-1) - z[:, len(ids["yes"]):].logsumexp(-1)
    y = torch.tensor(rows["label"].to_numpy(), dtype=torch.float32, device=d.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(d, y)


def val_scores(model, processor, val: pd.DataFrame) -> np.ndarray:
    """Log-odds of the validation images, scored by the audit's own path (cxr.audit.p_yes, eval mode)."""
    model.eval()
    d = p_yes(model, processor, [Image.open(p).convert("RGB") for p in val["path"]], QUESTION, BATCH)[:, 0]
    model.train()
    return d


def validate(model, processor, val: pd.DataFrame) -> tuple[float, float]:
    """Validation loss (the training loss with each class weighted 1/2) and CTR-AUROC."""
    d, y = val_scores(model, processor, val), val["label"].to_numpy()
    bce = np.logaddexp(0, -d) * y + np.logaddexp(0, d) * (1 - y)  # BCE with logits, stable
    return 0.5 * bce[y == 1].mean() + 0.5 * bce[y == 0].mean(), auroc(d[y == 1], d[y == 0])


def run_training(name: str, train: pd.DataFrame, val: pd.DataFrame, seed: int, steps: int, lr: float,
                 overfit: bool = False, real_val: pd.DataFrame | None = None):
    """The training loop. Training runs: AdamW, cosine schedule over `steps` (MAX_STEPS for every arm)
    with 5% warmup, batch 16 (in micro-batches of MICRO_BATCH), validation every EVAL_EVERY steps
    and at the last step, patience PATIENCE, the best adapter kept; real_val (arm E only): the real
    validation films, scored at every evaluation and kept as a separate checkpoint for H3.1's
    robustness check, never for selection. Overfit test: constant learning rate after 2 warmup steps,
    validation before the first step and after the last one, no early stopping.
    Writes outputs/h3/runs/<name>/: log.csv (after every evaluation), run.json (last), trainable.txt,
    best/ (and best_real/); returns the model and processor."""
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    revision = medgemma_revision()
    model, processor, names = lora_model(seed, revision)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (out / "trainable.txt").write_text("\n".join(names) + "\n")
    print(f"{name}: {len(names)} trainable tensors, {count:,} parameters (language model only); "
          f"MedGemma revision {revision}; code commit {commit()}; GPU {torch.cuda.get_device_name()}", flush=True)
    prompt, ids = prompt_for(processor, QUESTION), answer_ids(processor.tokenizer)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    schedule = (get_constant_schedule_with_warmup(optimizer, 2) if overfit
                else get_cosine_schedule_with_warmup(optimizer, int(WARMUP * steps), steps))
    rng = np.random.default_rng(seed)  # the data order
    order = np.array([], dtype=int)
    log, losses, best, best_real, waited = [], [], np.inf, np.inf, 0
    if overfit:
        loss0, auroc0 = validate(model, processor, val)
        log.append({"step": 0, "val_loss": loss0, "val_auroc": auroc0, "seconds": 0.0})
        print(f"  before training: eval loss {loss0:.4f}, AUROC {auroc0:.3f}", flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for step in range(1, steps + 1):
        if len(order) < BATCH:
            order = np.concatenate([order, rng.permutation(len(train))])
        rows, order = train.iloc[order[:BATCH]], order[BATCH:]
        step_loss = 0.0
        for i in range(0, BATCH, MICRO_BATCH):  # the mean loss over the batch, in micro-batches
            part = rows.iloc[i:i + MICRO_BATCH]
            loss = batch_loss(model, processor, prompt, ids, part) * len(part) / len(rows)
            loss.backward()
            step_loss += loss.item()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        losses.append(step_loss)
        if overfit:
            log.append({"step": step, "train_loss": step_loss, "seconds": time.time() - start})
            print(f"  step {step}: loss {step_loss:.4f}", flush=True)
            continue
        if step % EVAL_EVERY and step != steps:
            continue
        row = {"step": step, "train_loss": float(np.mean(losses)), "seconds": time.time() - start, "lr": schedule.get_last_lr()[0]}
        losses = []
        row["val_loss"], row["val_auroc"] = validate(model, processor, val)
        if row["val_loss"] < best:
            best, waited, row["kept"] = row["val_loss"], 0, True
            model.save_pretrained(out / "best")
        else:
            waited += 1
        if real_val is not None:
            row["real_val_loss"], row["real_val_auroc"] = validate(model, processor, real_val)
            if row["real_val_loss"] < best_real:
                best_real = row["real_val_loss"]
                model.save_pretrained(out / "best_real")
        log.append(row)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)  # a killed run keeps its curve
        print(f"  {name} step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)), flush=True)
        if waited >= PATIENCE:
            break
    seconds = time.time() - start
    if overfit:
        loss1, auroc1 = validate(model, processor, val)
        log.append({"step": steps, "val_loss": loss1, "val_auroc": auroc1, "seconds": seconds})
    log = pd.DataFrame(log)
    log.to_csv(out / "log.csv", index=False)
    kept = log.loc[log["kept"].fillna(False).astype(bool), "step"] if "kept" in log else pd.Series(dtype=int)
    (out / "run.json").write_text(json.dumps({
        "name": name, "seed": seed, "lr": lr, "n_train": len(train), "n_val": len(val),
        "train_draw": draw_hash(train), "val_draw": draw_hash(val), "revision": revision, "commit": commit(),
        "trainable_tensors": len(names), "trainable_parameters": count, "micro_batch": MICRO_BATCH,
        "gpu": torch.cuda.get_device_name(), "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "steps_run": int(log["step"].max()), "best_step": int(kept.max()) if len(kept) else None,
        "seconds": seconds}, indent=1))
    return model, processor


def train(args: argparse.Namespace) -> None:
    """One H3 run: arm R, E or RE, training seed in SEEDS (outputs/h3/runs/<arm><seed>/). A finished
    run (run.json written) is skipped; an interrupted one is redone from the start."""
    name = f"{args.arm}{args.seed}"
    if (RUNS / name / "run.json").exists():
        print(f"{name}: done, skipped", flush=True)
        return
    data, val, real_val = training_data(args.arm)
    print(f"{name}: {len(data)} training images ({int(data['label'].sum())} yes), {len(val)} validation images", flush=True)
    run_training(name, data, val, args.seed, MAX_STEPS, LR, real_val=real_val if args.arm == "E" else None)


def overfit(args: argparse.Namespace) -> None:
    """32 examples of R (16 per class), args.steps steps at a constant learning rate. Passes if: (1)
    before training, the AUROC on the 32 films is above 0.5 (the labels are not inverted, and the
    eval-mode scoring of the LoRA model works); (2) after training, the eval-mode loss is below 0.05;
    (3) the saved adapter, reloaded on a fresh base, gives the same scores. Also measures the seconds
    per step and the peak memory -> results/h3_overfit.csv."""
    from peft import PeftModel

    data, _, _ = training_data("R")
    data = pd.concat([data[data["label"] == c].sample(n=16, random_state=0) for c in (1, 0)]).reset_index(drop=True)
    model, processor = run_training("overfit", data, data, 0, args.steps, args.lr, overfit=True)
    final = val_scores(model, processor, data)
    model.save_pretrained(RUNS / "overfit" / "final")
    del model
    torch.cuda.empty_cache()
    base, processor = load_medgemma(MODEL_ID, medgemma_revision())
    reloaded = PeftModel.from_pretrained(base, RUNS / "overfit" / "final").eval()
    gap = float(np.abs(val_scores(reloaded, processor, data) - final).max())
    log = pd.read_csv(RUNS / "overfit" / "log.csv")
    run = json.loads((RUNS / "overfit" / "run.json").read_text())
    log["seconds_per_step"] = log["seconds"].diff()
    log = log.assign(gpu=run["gpu"], peak_memory_gb=run["peak_memory_gb"], micro_batch=run["micro_batch"],
                     reload_max_log_odds_difference=gap)
    log.to_csv(RESULTS / "h3_overfit.csv", index=False)
    first, last = log[log["step"] == 0].iloc[0], log.dropna(subset=["val_loss"]).iloc[-1]
    steps = log.dropna(subset=["train_loss"])
    checks = {"AUROC before training > 0.5": first["val_auroc"] > 0.5, "eval loss after training < 0.05": last["val_loss"] < 0.05,
              "reloaded adapter gives the same scores (max |d difference| < 0.01)": gap < 0.01}
    print(f"overfit: eval loss {first['val_loss']:.4f} -> {last['val_loss']:.4f}, AUROC {first['val_auroc']:.3f} -> "
          f"{last['val_auroc']:.3f}; reload gap {gap:.5f}; {steps['seconds_per_step'].iloc[1:].median():.2f} s per step "
          f"(median, first step excluded); peak memory {run['peak_memory_gb']:.1f} GB on {run['gpu']}", flush=True)
    for check, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {check}", flush=True)


def eval_images(which: str) -> pd.DataFrame:
    """The images of an evaluation set: 'test' (the audit's test split: real films, the dose run's
    edits and shams, the audit's shams and blanks; secondary) or 'fresh' (H3's fresh test set: its
    real films, the fresh edits and shams, blanks of the fresh source films; primary). The counts
    must equal EXPECTED."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta = meta[meta["available"]]
    sets = patient_sets()
    real = meta[meta["patient_id"].isin(sets[which])]
    rows = [{"kind": "real", "image": r.image, "growth": np.nan, "path": DATA / "nih512" / r.file} for r in real.itertuples()]
    if which == "test":
        edit_path = lambda g: OUTPUTS / "dose" / "cardiomegaly" / f"growth{g:.2f}"
        sham_dirs = [(OUTPUTS / "pairs" / "test" / "cardiomegaly" / "sham", 0.30),
                     (OUTPUTS / "dose" / "cardio_sham" / "level0.00", 0.0), (OUTPUTS / "dose" / "cardio_sham" / "level0.06", 0.06)]
    else:
        edit_path = lambda g: H3 / "fresh" / "edit" / f"growth{g:.2f}"
        sham_dirs = [(H3 / "fresh" / "sham" / "growth0.30", 0.30)]
    for g in GROWTHS:
        rows += [{"kind": "edit", "image": p.name, "growth": g, "path": p} for p in sorted(edit_path(g).glob("*.png"))]
    for folder, g in sham_dirs:
        rows += [{"kind": "sham", "image": p.name, "growth": g, "path": p} for p in sorted(folder.glob("*.png"))]
    sources = sorted({r["image"] for r in rows if r["kind"] == "sham" and r["growth"] == 0.30})
    files = meta.set_index("image")
    rows += [{"kind": "blank", "image": i, "growth": np.nan, "path": DATA / "nih512" / files.loc[i, "file"]} for i in sources]
    table = pd.DataFrame(rows)
    table["patient_id"] = files.loc[table["image"], "patient_id"].to_numpy()
    assert set(table["patient_id"]) <= sets[which], f"{which}: images from outside its patients"
    counts = table["kind"].value_counts().to_dict()
    assert counts == EXPECTED[which], f"{which}: {counts} images, expected {EXPECTED[which]}"
    return table


def check_final() -> None:
    """Before the fresh set is touched: committed code, and every planned adapter on disk."""
    assert "uncommitted" not in commit() and commit() != "unknown", f"code commit: {commit()}"
    missing = [f"{a}{s}/best" for a in ARMS for s in SEEDS if not (RUNS / f"{a}{s}" / "best" / "adapter_model.safetensors").exists()]
    missing += [f"E{s}/best_real" for s in SEEDS if not (RUNS / f"E{s}" / "best_real" / "adapter_model.safetensors").exists()]
    assert not missing, f"adapters missing: {missing}"


def evaluate(args: argparse.Namespace) -> None:
    """Scores of one model on one evaluation set, through the audit's path (encode, p_yes): every image
    with both cardiomegaly questions, real films also with the first effusion question
    -> results/h3_scores_<model>_<set>.csv (+ .json: revision, commit, draws). With --model base it
    also measures the set (segmenter CTR of every real film, edit and sham; CheXmask CTR and NIH
    labels of the real films) -> results/h3_measures_<set>.csv, and on the test set writes the val
    split's shares of CTR > 0.5 (the prior correction's pi) -> results/h3_val_prior.csv.
    Finished scores are skipped. The fresh set needs --final (PLAN.md, H3)."""
    from peft import PeftModel

    if args.set == "fresh":
        assert args.final, "the fresh test set is scored only in the final run: add --final"
        check_final()
    scores_file = RESULTS / f"h3_scores_{args.model}_{args.set}.csv"
    measures_file = RESULTS / f"h3_measures_{args.set}.csv"
    if scores_file.exists() and (args.model != "base" or measures_file.exists()):
        print(f"{args.model} on the {args.set} set: done, skipped", flush=True)
        return
    images = eval_images(args.set)
    revision, info = medgemma_revision(), {"model": args.model, "set": args.set}
    if args.model != "base":  # e.g. R0, or E0real for E's best checkpoint on the real validation films
        name, folder = (args.model[:-4], "best_real") if args.model.endswith("real") else (args.model, "best")
        run = json.loads((RUNS / name / "run.json").read_text())  # a run without run.json was interrupted
        assert run["revision"] == revision, f"{name} was trained on MedGemma {run['revision']}, not {revision}"
        info.update(train_draw=run["train_draw"], val_draw=run["val_draw"], best_step=run["best_step"])
    model, processor = load_medgemma(MODEL_ID, revision)
    if args.model != "base":
        model = PeftModel.from_pretrained(model, RUNS / name / folder).eval()
    loaded = [Image.open(p).convert("L") for p in images["path"]]
    loaded = [(blank(im) if k == "blank" else im).convert("RGB") for im, k in zip(loaded, images["kind"])]
    parts = []
    jobs = [(f"cardiomegaly_{k}", q, images.index) for k, q in enumerate(QUESTIONS["cardiomegaly"])]
    jobs.append(("effusion_0", QUESTIONS["effusion"][0], images.index[images["kind"] == "real"]))
    for code, question, idx in jobs:
        s = p_yes(model, processor, [loaded[i] for i in idx], question, args.batch_size)
        parts.append(images.loc[idx, ["kind", "image", "patient_id", "growth"]].assign(
            question=code, log_odds=s[:, 0], p_yes=s[:, 1], yes_no_mass=s[:, 2]))
        print(f"  {args.model} {args.set}: {question} - {len(idx)} images scored", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(scores_file, index=False)  # float32 scores, written exactly
    info.update(revision=revision, commit=commit(), n_images=len(images), gpu=torch.cuda.get_device_name())
    scores_file.with_suffix(".json").write_text(json.dumps(info, indent=1))
    if args.model == "base":
        measured = images[images["kind"] != "blank"].copy()
        measured["ctr"] = segment_ctr(measured["path"].tolist())
        meta = pd.read_csv(DATA / "nih512" / "metadata.csv").set_index("image")
        chexmask = pd.read_csv(OUTPUTS / "ctr_nih.csv", usecols=["image", "ctr", "good_mask"]).set_index("image")
        real = measured["kind"] == "real"
        cm = chexmask.reindex(measured.loc[real, "image"])
        measured.loc[real, "ctr_chexmask"] = cm["ctr"].where(cm["good_mask"].fillna(False).astype(bool)).to_numpy()
        for col in ["cardiomegaly", "effusion", "no_finding"]:
            measured.loc[real, col] = meta.loc[measured.loc[real, "image"], col].to_numpy()
        measured.drop(columns="path").to_csv(measures_file, index=False)
    if args.model == "base" and args.set == "test":
        real_val = pd.read_csv(H3 / "ctr_real.csv").query("set == 'val'").dropna(subset=["ctr"])
        edits_val = pd.read_csv(H3 / "ctr_edits.csv").query("set == 'val'").dropna(subset=["ctr"])
        prior = [{"images": name, "n": len(d), "share_ctr_above_05": (d["ctr"] > 0.5).mean()} for name, d in
                 [("real films", real_val), ("additions", edits_val[edits_val["kind"] == "edit"]),
                  ("shams", edits_val[edits_val["kind"] == "sham"]), ("additions and shams", edits_val)]]
        pd.DataFrame(prior).to_csv(RESULTS / "h3_val_prior.csv", index=False)
    print(f"{args.model} on the {args.set} set: done; code commit {commit()}, MedGemma revision {revision}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("overfit", help="32 examples, a few steps: checks of the training wiring")
    o.add_argument("--steps", type=int, default=50)
    o.add_argument("--lr", type=float, default=LR)
    o.set_defaults(func=overfit)
    t = sub.add_parser("train", help="One arm and seed of H3")
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--seed", type=int, choices=SEEDS, required=True)
    t.set_defaults(func=train)
    e = sub.add_parser("evaluate", help="Score a model on an evaluation set (fresh: final run only)")
    e.add_argument("--model", required=True, help="base, or <arm><seed> (e.g. R0), or E<seed>real")
    e.add_argument("--set", choices=["test", "fresh"], required=True)
    e.add_argument("--final", action="store_true", help="Required for the fresh set: the final run")
    e.add_argument("--batch-size", type=int, default=16)
    e.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
