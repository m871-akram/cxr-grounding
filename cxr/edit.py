"""Stage 2: counterfactual chest X-rays with RadEdit. So far:

    python -m cxr.edit smoke     # 10 val films per finding, remove and add -> results/radedit_smoke_*.jpg
    python -m cxr.edit bugcheck  # two quirks of RadEdit's released pipeline -> results/radedit_bugcheck.csv
    python -m cxr.edit tune [--round removals]  # day-2 tuning rounds, scored by the three checks
                                 # -> results/radedit_tuning*  (run python -m cxr.checks validate first)

RadEdit (microsoft/radedit; research-only weights, never redistributed) edits a 512-px chest X-ray
following a text prompt, inside an edit mask, while the film is copied back inside a keep mask.
The masks are drawn from each film's CheXmask landmarks: the heart, dilated, for cardiomegaly (room
for the heart border to move either way), and the lower half of both lungs, dilated, for pleural
effusion (fluid collects at the lung bases). The keep mask is everything else minus a free band
around the edit mask, where RadEdit may blend the edit into the film (paper, p.28-29).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from . import DATA, OUTPUTS, RESULTS
from .data import ACCENT, HEART, INK, INK_2, SURFACE

RIGHT_LUNG, LEFT_LUNG = slice(0, 44), slice(44, 94)  # CheXmask landmark order (see data.py)
NORMAL_PROMPT = "No acute cardiopulmonary process"   # removal prompt of the paper and model card
# RadEdit saw NIH films conditioned on their NIH label names (paper, p.9), so additions use them.
ADD_PROMPTS = {"cardiomegaly": "Cardiomegaly.", "effusion": "Effusion."}
DILATION_PX = {"cardiomegaly": 25, "effusion": 15}   # at 512 px
FREE_BAND_PX = 16  # two 8-px latent cells between edit and keep masks (no seam at the border)


def load_films(split: str) -> pd.DataFrame:
    """Converted 512-px films of one split, with their CheXmask landmarks (good masks only)."""
    meta = pd.read_csv(DATA / "nih512" / "metadata.csv")
    meta = meta[meta["available"] & (meta["split"] == split)]
    marks = pd.read_csv(DATA / "chexmask_nih_landmarks.csv").rename(columns={"Image Index": "image"})
    marks = marks[marks["Dice RCA (Mean)"] >= 0.7]
    return meta.merge(marks, on="image")


def landmarks_512(film: pd.Series, size: int = 512) -> np.ndarray:
    """CheXmask landmarks (original pixels) -> (120, 2) array of (x, y) in the 512-px film.

    Same geometry as data.to_square_resized: the film is padded to a centred square, then resized.
    """
    xy = np.array(film["Landmarks"].split(","), dtype=float).reshape(120, 2)
    h, w = film["Height"], film["Width"]
    side = max(h, w)
    return (xy + [(side - w) // 2, (side - h) // 2]) * size / side


def polygon_mask(points: np.ndarray, size: int = 512) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in points], fill=1)
    return np.array(img, dtype=bool)


def edit_mask(xy: np.ndarray, finding: str, size: int = 512) -> np.ndarray:
    """Where RadEdit may change the film: the dilated heart, or the dilated lower half of both lungs."""
    from skimage.morphology import dilation, disk

    if finding == "cardiomegaly":
        mask = polygon_mask(xy[HEART], size)
    else:
        mask = np.zeros((size, size), dtype=bool)
        for lung in (RIGHT_LUNG, LEFT_LUNG):
            lung_mask = polygon_mask(xy[lung], size)
            ys = xy[lung][:, 1]
            lung_mask[: int((ys.min() + ys.max()) / 2)] = False  # keep the lower half
            mask |= lung_mask
    return dilation(mask, disk(DILATION_PX[finding]))


def keep_mask(edit: np.ndarray) -> np.ndarray:
    """Where the film must stay as it is: everything but the edit mask and a free band around it
    (the paper saw a dark seam at the border when changes were confined to the mask alone)."""
    from skimage.morphology import dilation, disk

    return ~dilation(edit, disk(FREE_BAND_PX))


def grown_heart_mask(xy: np.ndarray, growth: float, size: int = 512) -> np.ndarray:
    """Room for a heart that enlarges like a real one: mostly sideways toward the image's right (the
    patient's left, where the left ventricle grows), less toward the other side, and with the apex
    moving down and out; never upward, and the rest of the lower border stays on the diaphragm.
    The heart contour is stretched away from its centre; growth 0.15 widens it by ~15%."""
    heart = xy[HEART]
    cx, cy = heart.mean(axis=0)
    grown = heart.copy()
    right, below = grown[:, 0] > cx, grown[:, 1] > cy
    grown[right, 0] = cx + (grown[right, 0] - cx) * (1 + 1.5 * growth)
    grown[~right, 0] = cx + (grown[~right, 0] - cx) * (1 + 0.5 * growth)
    toward_apex = np.clip((heart[:, 0] - cx) / (heart[:, 0].max() - cx), 0, 1)  # 0 at the centre, 1 at the apex
    grown[below, 1] = cy + (grown[below, 1] - cy) * (1 + growth * toward_apex[below])  # smooth, apex moves most
    return polygon_mask(grown, size) | polygon_mask(heart, size)


def heart_band_mask(xy: np.ndarray, shrink: float, size: int = 512) -> np.ndarray:
    """For removing cardiomegaly: only the band between the current heart border and a heart shrunk
    sideways (mostly on the apex side, the inverse of grown_heart_mask), i.e. the part that should
    become lung. RadEdit fills whatever shape it is given. The heart is widened by 8 px (one latent
    cell) so that its current border lies inside the band."""
    from skimage.morphology import dilation, disk

    heart = xy[HEART]
    cx = heart[:, 0].mean()
    shrunk = heart.copy()
    right = shrunk[:, 0] > cx
    shrunk[right, 0] = cx + (shrunk[right, 0] - cx) / (1 + 1.5 * shrink)
    shrunk[~right, 0] = cx + (shrunk[~right, 0] - cx) / (1 + 0.5 * shrink)
    return dilation(polygon_mask(heart, size), disk(8)) & ~polygon_mask(shrunk, size)


def lung_bases_mask(xy: np.ndarray, medial: float, lateral: float, dilate_px: int,
                    size: int = 512) -> np.ndarray:
    """Lower part of each lung under a curved upper edge that rises toward the chest wall, like the
    meniscus of a pleural effusion. medial / lateral: height of that edge as a fraction of the
    lung's height (0 = apex, 1 = base) on the inner and on the outer side of the lung."""
    from skimage.morphology import dilation, disk

    rows, cols = np.arange(size)[:, None], np.arange(size)
    mask = np.zeros((size, size), dtype=bool)
    for lung, wall_on_left in [(RIGHT_LUNG, True), (LEFT_LUNG, False)]:  # the right lung is on the image's left
        points = xy[lung]
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        t = np.clip((cols - x0) / (x1 - x0), 0, 1)
        to_wall = 1 - t if wall_on_left else t  # 0 on the inner side, 1 at the chest wall
        edge = y0 + (medial + (lateral - medial) * to_wall**2) * (y1 - y0)  # rises fastest near the wall
        mask |= polygon_mask(points, size) & (rows >= edge[None, :])
    return dilation(mask, disk(dilate_px))


def paste_back(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, feather_px: float) -> np.ndarray:
    """The edit inside the mask, the original film outside, blended across the border over a few
    pixels: the blending weight is the mask blurred by a Gaussian of feather_px."""
    from PIL import ImageFilter

    weight = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather_px))
    weight = np.asarray(weight, dtype=float) / 255
    return weight * edited + (1 - weight) * original


def edit_film(pipe, original: Image.Image, mask: np.ndarray, prompt: str, guidance: float,
              skip_ratio: float, steps: int, seed: int, feather_px: float) -> np.ndarray:
    """One RadEdit edit with the original pixels pasted back outside the mask (512 x 512, uint8).
    Used for edits and sham edits alike. The keep mask is 1 - edit: the released pipeline ignores
    its content anyway (bugcheck), and the paste-back removes its off-by-one noise outside the mask."""
    torch.manual_seed(seed)
    edited = pipe(prompt, weights=[guidance], image=original.convert("RGB"),
                  edit_mask=Image.fromarray(mask.astype(np.uint8) * 255),
                  keep_mask=Image.fromarray((~mask).astype(np.uint8) * 255), num_inference_steps=steps,
                  invert_prompt="", skip_ratio=skip_ratio, output_type="pil")[0]
    before = np.asarray(original.convert("L"), dtype=float)
    after = np.asarray(edited.convert("L"), dtype=float)
    return np.clip(paste_back(before, after, mask, feather_px), 0, 255).round().astype(np.uint8)


def load_radedit(device: str = "cuda", revisions: dict | None = None):
    """RadEdit exactly as on its model card (UNet, SDXL VAE, BioViL-T text encoder, DDIM), then
    RadEdit's own editing pipeline on top (custom code from the Hub, hence trust_remote_code).
    revisions pins each Hugging Face repo to a commit (H3); without it, the cached main is used."""
    from diffusers import (AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline,
                           UNet2DConditionModel)
    from transformers import AutoModel, AutoTokenizer

    rev = lambda repo: {"revision": revisions[repo]} if revisions else {}
    unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet", **rev("microsoft/radedit"))
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", **rev("stabilityai/sdxl-vae"))
    text_encoder = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True,
                                             **rev("microsoft/BiomedVLP-BioViL-T"))
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", model_max_length=128,
                                              trust_remote_code=True, **rev("microsoft/BiomedVLP-BioViL-T"))
    scheduler = DDIMScheduler(beta_schedule="linear", clip_sample=False, prediction_type="epsilon",
                              timestep_spacing="trailing", steps_offset=1)
    base = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                   scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
                                   feature_extractor=None).to(device)
    pinned = {"custom_revision": revisions["microsoft/radedit"]} if revisions else {}
    return DiffusionPipeline.from_pipe(base, custom_pipeline="microsoft/radedit", trust_remote_code=True, **pinned)


def gallery(panels: list, title: str, path, labels: list | None = None) -> None:
    """Rows: original, edited, |difference| (x4); one column per film; the edit mask outlined.
    Optional column labels ("valid" is drawn in the accent colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(panels), figsize=(1.5 * len(panels), 5.0), facecolor=SURFACE,
                             squeeze=False)
    for j, (original, edited, mask) in enumerate(panels):
        diff = np.abs(edited.astype(float) - original)
        for i, (img, vmax) in enumerate([(original, 255), (edited, 255), (diff, 64)]):
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax)
            axes[i, j].contour(mask, levels=[0.5], colors=ACCENT, linewidths=0.5)
            axes[i, j].set_axis_off()
        if labels:
            axes[0, j].set_title(labels[j], fontsize=8, color=ACCENT if labels[j] == "valid" else INK_2)
    for i, label in enumerate(["original", "edited", "|difference| x4"]):
        axes[i, 0].text(-0.06, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=9, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.01, top=0.9, wspace=0.03, hspace=0.03)
    # JPEG: a 10-film gallery is ~2 MB as PNG, ~0.4 MB here, at the same 200 dpi.
    fig.savefig(path, dpi=200, facecolor=SURFACE, pil_kwargs={"quality": 80, "optimize": True})
    plt.close(fig)


def smoke(args: argparse.Namespace) -> None:
    """Remove each finding from n val films that have it, add it to n normal films (one film per
    patient); save the edits, a gallery per finding and direction, and how much changed inside the
    edit mask and inside the keep mask."""
    torch.backends.cuda.matmul.allow_tf32 = True  # A100: fp32 matmuls on tensor cores
    films = load_films("val").drop_duplicates("patient_id")
    normals = films[films["no_finding"] == 1].sample(n=args.n, random_state=args.seed)
    pipe = None
    rows = []
    for finding in ["cardiomegaly", "effusion"]:
        positives = films[films[finding] == 1].sample(n=args.n, random_state=args.seed)
        for direction, group, prompt in [("remove", positives, NORMAL_PROMPT),
                                         ("add", normals, ADD_PROMPTS[finding])]:
            out_dir = OUTPUTS / "radedit_smoke" / f"{finding}_{direction}"
            out_dir.mkdir(parents=True, exist_ok=True)
            panels = []
            for k, (_, film) in enumerate(group.iterrows()):
                original = Image.open(DATA / "nih512" / film["file"]).convert("L")
                mask = edit_mask(landmarks_512(film), finding)
                keep = keep_mask(mask)
                out = out_dir / film["image"]
                if not out.exists():  # finished edits are skipped
                    if pipe is None:
                        pipe = load_radedit()
                    torch.manual_seed(args.seed + k)
                    edited = pipe(prompt, weights=[args.guidance], image=original.convert("RGB"),
                                  edit_mask=Image.fromarray(mask.astype(np.uint8) * 255),
                                  keep_mask=Image.fromarray(keep.astype(np.uint8) * 255),
                                  num_inference_steps=args.steps, invert_prompt="",
                                  skip_ratio=args.skip_ratio, output_type="pil")[0]  # a list of images
                    edited.convert("L").save(out)
                before = np.asarray(original, dtype=float)
                after = np.asarray(Image.open(out).convert("L"), dtype=float)
                diff = np.abs(after - before)
                rows.append({"image": film["image"], "patient_id": film["patient_id"], "finding": finding,
                             "direction": direction, "prompt": prompt, "seed": args.seed + k,
                             "change_in_edit": diff[mask].mean(), "change_in_keep": diff[keep].mean()})
                panels.append((before, after, mask))
            gallery(panels, f"RadEdit smoke test: {direction} {finding} (prompt: \"{prompt}\")",
                    RESULTS / f"radedit_smoke_{finding}_{direction}.jpg")

    table = pd.DataFrame(rows).round(2)
    table.to_csv(RESULTS / "radedit_smoke.csv", index=False)
    print("Mean absolute change (gray levels 0-255), inside the edit mask and inside the keep mask:")
    print(table.groupby(["finding", "direction"])[["change_in_edit", "change_in_keep"]].mean().round(2)
          .to_string())
    print(f"Galleries: {RESULTS}/radedit_smoke_*.jpg")


def bugcheck(args: argparse.Namespace) -> None:
    """Show the two quirks of RadEdit's released pipeline with its own public call, on one val film.

    1. Off-by-one. Outside the edit mask, the pipeline pastes back the inverted latent of the step
       it has just done instead of the next one, so one step of noise is left there. With an empty
       prompt and guidance 1, RadEdit only reconstructs the film, and pasting the film back must
       then change nothing: the run with paste-back should match a run that edits everywhere.
       With the bug they differ outside the mask, less with more steps (less noise per step).
    2. Keep mask ignored. A second `if keep_mask is not None` block pastes the film back everywhere
       outside the edit mask. Then an empty keep mask gives exactly the same edit as
       keep = 1 - edit, while passing no keep mask at all gives a different one.
    """
    from skimage.morphology import dilation, disk

    pipe = load_radedit(args.device)
    films = load_films("val").drop_duplicates("patient_id")
    film = films[films["cardiomegaly"] == 1].iloc[0]
    original = Image.open(DATA / "nih512" / film["file"]).convert("RGB")
    edit = edit_mask(landmarks_512(film), "cardiomegaly")
    far = ~dilation(edit, disk(32))  # outside, away from the border (the VAE decoder mixes ~32 px)
    everything, nothing = np.ones_like(edit), np.zeros_like(edit)

    def run(prompt: str, guidance: float, steps: int, edit_region, keep_region) -> np.ndarray:
        to_image = lambda m: None if m is None else Image.fromarray(m.astype(np.uint8) * 255)
        torch.manual_seed(0)  # the same inversion noise in every run
        out = pipe(prompt, weights=[guidance], image=original, edit_mask=to_image(edit_region),
                   keep_mask=to_image(keep_region), num_inference_steps=steps, invert_prompt="",
                   skip_ratio=0.3, output_type="pil")[0]
        return np.asarray(out.convert("L"), dtype=float)

    rows = []
    for steps in args.steps:
        plain = run("", 1.0, steps, everything, nothing)  # reconstruction, nothing pasted back
        pasted = run("", 1.0, steps, edit, ~edit)          # reconstruction, film pasted back outside
        diff = np.abs(pasted - plain)
        rows.append({"check": "1: paste-back vs plain reconstruction", "steps": steps,
                     "mean_abs_diff_outside": diff[far].mean(), "mean_abs_diff_inside": diff[edit].mean()})
    steps, prompt = max(args.steps), ADD_PROMPTS["cardiomegaly"]
    with_keep = run(prompt, 7.5, steps, edit, ~edit)
    empty_keep = run(prompt, 7.5, steps, edit, nothing)
    no_keep = run(prompt, 7.5, steps, edit, None)
    for name, a, b in [("2: keep = 1 - edit vs empty keep", with_keep, empty_keep),
                       ("2: empty keep vs no keep", empty_keep, no_keep)]:
        diff = np.abs(a - b)
        rows.append({"check": name, "steps": steps, "mean_abs_diff_outside": diff[far].mean(),
                     "mean_abs_diff_inside": diff[edit].mean()})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "radedit_bugcheck.csv", index=False)
    print("Mean absolute difference (gray levels 0-255) between two runs, outside and inside the edit mask:")
    print(table.to_string(index=False))


HEART_REMOVAL = ("heart", None)
EFFUSION_REMOVAL = ("lung_bases", (0.55, 0.3, 20))
TUNE_ROUNDS = {  # name, finding, direction, prompt, (mask kind, its parameters), skip ratio
    "first": [
        ("cardiomegaly_add_small", "cardiomegaly", "add", "Cardiomegaly", ("grown_heart", 0.15), 0.5),
        ("cardiomegaly_add_large", "cardiomegaly", "add", "Cardiomegaly", ("grown_heart", 0.30), 0.5),
        ("cardiomegaly_remove_generic", "cardiomegaly", "remove", NORMAL_PROMPT, HEART_REMOVAL, 0.5),
        ("cardiomegaly_remove_specific", "cardiomegaly", "remove", "Normal heart size", HEART_REMOVAL, 0.5),
        ("effusion_add", "effusion", "add", ADD_PROMPTS["effusion"], ("lung_bases", (0.75, 0.5, 10)), 0.5),
        ("effusion_remove_generic", "effusion", "remove", NORMAL_PROMPT, EFFUSION_REMOVAL, 0.5),
        ("effusion_remove_specific", "effusion", "remove", "No pleural effusion", EFFUSION_REMOVAL, 0.5),
    ],
    "removals": [  # follow-up: lower skip ratios leave RadEdit more room to erase a finding
        ("cardiomegaly_remove_generic_skip0.3", "cardiomegaly", "remove", NORMAL_PROMPT, HEART_REMOVAL, 0.3),
        ("cardiomegaly_remove_generic_skip0.2", "cardiomegaly", "remove", NORMAL_PROMPT, HEART_REMOVAL, 0.2),
        ("cardiomegaly_remove_specific_skip0.3", "cardiomegaly", "remove", "Normal heart size", HEART_REMOVAL, 0.3),
        ("cardiomegaly_remove_specific_skip0.2", "cardiomegaly", "remove", "Normal heart size", HEART_REMOVAL, 0.2),
        ("cardiomegaly_remove_band_skip0.3", "cardiomegaly", "remove", NORMAL_PROMPT, ("heart_band", 0.30), 0.3),
        ("cardiomegaly_remove_band_skip0.2", "cardiomegaly", "remove", NORMAL_PROMPT, ("heart_band", 0.30), 0.2),
        ("effusion_remove_generic_skip0.3", "effusion", "remove", NORMAL_PROMPT, EFFUSION_REMOVAL, 0.3),
        ("effusion_remove_generic_skip0.2", "effusion", "remove", NORMAL_PROMPT, EFFUSION_REMOVAL, 0.2),
        ("effusion_remove_specific_skip0.3", "effusion", "remove", "No pleural effusion", EFFUSION_REMOVAL, 0.3),
        ("effusion_remove_specific_skip0.2", "effusion", "remove", "No pleural effusion", EFFUSION_REMOVAL, 0.2),
    ],
}


def tuning_mask(xy: np.ndarray, kind: str, param) -> np.ndarray:
    if kind == "grown_heart":
        return grown_heart_mask(xy, param)
    if kind == "heart_band":
        return heart_band_mask(xy, param)
    if kind == "heart":  # removal: the whole heart and a 25-px margin
        return edit_mask(xy, "cardiomegaly")
    return lung_bases_mask(xy, *param)  # removals use a higher edge and a wider margin


def score_pairs(rows: list, used: dict, feather: float) -> tuple[list, list]:
    """The three checks for every (original, edit) pair; results are added to each row.
    Rows need: file (original), edit_file, mask, finding, direction. Returns the images."""
    from skimage.morphology import dilation, disk

    from .checks import classify, ctr_of, load_classifiers, load_segmenter, segment

    originals = [np.asarray(Image.open(DATA / "nih512" / r["file"]).convert("L")) for r in rows]
    edits = [np.asarray(Image.open(r["edit_file"]).convert("L")) for r in rows]
    models = load_classifiers()
    p_before, p_after = classify(models, originals), classify(models, edits)
    seg = load_segmenter()
    (hearts0, lungs0), (hearts1, lungs1) = segment(seg, originals), segment(seg, edits)
    for i, r in enumerate(rows):
        finding, sign, mask = r["finding"], (1 if r["direction"] == "add" else -1), r["mask"]
        # 1. every validated classifier ends on the target side of its threshold, having moved that way
        classifiers_ok = True
        for name, threshold in used[finding].items():
            before, after = p_before.loc[i, f"{name}:{finding}"], p_after.loc[i, f"{name}:{finding}"]
            r[f"p_before:{name}"], r[f"p_after:{name}"] = before, after
            classifiers_ok &= bool(sign * (after - threshold) > 0 and sign * (after - before) > 0)
        # 2. anatomy (same segmenter before and after): the CTR crosses 0.5, or the aerated lung
        #    inside the mask shrinks / grows by 10%
        if finding == "cardiomegaly":
            ctr0, ctr1 = ctr_of(hearts0[i], lungs0[i]), ctr_of(hearts1[i], lungs1[i])
            r["ctr_before"], r["ctr_after"] = ctr0, ctr1
            anatomy_ok = bool(ctr0 <= 0.5 < ctr1) if sign > 0 else bool(ctr1 <= 0.5 < ctr0)
        else:
            area0, area1 = (lungs0[i] & mask).sum(), (lungs1[i] & mask).sum()
            change = (area1 - area0) / max(area0, 1)
            r["lung_area_change"] = change
            anatomy_ok = bool(change <= -0.10) if sign > 0 else bool(change >= 0.10)
        # 3. nothing changed beyond the mask and its blended border
        far = ~dilation(mask, disk(int(4 * feather) + 1))
        r["max_change_outside"] = int(np.abs(edits[i].astype(int) - originals[i].astype(int))[far].max())
        outside_ok = r["max_change_outside"] <= 1
        r.update(classifiers_ok=classifiers_ok, anatomy_ok=anatomy_ok, outside_ok=outside_ok,
                 valid=classifiers_ok and anatomy_ok and outside_ok)
    return originals, edits


GENERATE = {  # the addition configs kept after tuning: prompt, (mask kind, parameters), skip ratio
    "cardiomegaly": ("Cardiomegaly", ("grown_heart", 0.30), 0.5),
    "effusion": (ADD_PROMPTS["effusion"], ("lung_bases", (0.75, 0.5, 10)), 0.5),
}


def generate(args: argparse.Namespace) -> None:
    """Addition and sham edit for each source film of one split: normal films (for cardiomegaly
    with CTR < 0.5 by CheXmask), one per patient, up to n per finding. The sham uses the same mask,
    seed and edit path with the normal prompt, so only the prompt differs. Both are scored by the
    three checks (a sham that passes would reveal an artifact). Edits: outputs/pairs/<split>/;
    per-pair table: outputs/pairs_<split>.csv; pass rates: results/pairs_<split>_summary.csv."""
    from .checks import used_classifiers

    used = used_classifiers()
    torch.backends.cuda.matmul.allow_tf32 = True
    films = load_films(args.split).merge(pd.read_csv(OUTPUTS / "ctr_nih.csv", usecols=["image", "ctr"]), on="image")
    normals = films[films["no_finding"] == 1]
    sources = {"cardiomegaly": normals[normals["ctr"] < 0.5], "effusion": normals}
    counts = {"cardiomegaly": args.n_cardiomegaly, "effusion": args.n_effusion}
    pipe, rows = None, []
    for finding, (prompt, (kind, param), skip) in GENERATE.items():
        chosen = sources[finding].drop_duplicates("patient_id")
        chosen = chosen.sample(n=min(counts[finding], len(chosen)), random_state=args.seed)
        print(f"{finding}: {len(chosen)} source films ({args.split})", flush=True)
        for k, (_, film) in enumerate(chosen.iterrows()):
            mask = tuning_mask(landmarks_512(film), kind, param)
            for pair_kind, pair_prompt in [("edit", prompt), ("sham", NORMAL_PROMPT)]:
                out = OUTPUTS / "pairs" / args.split / finding / pair_kind / film["image"]
                if not out.exists():  # finished edits are skipped
                    if pipe is None:
                        pipe = load_radedit()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    original = Image.open(DATA / "nih512" / film["file"])
                    Image.fromarray(edit_film(pipe, original, mask, pair_prompt, args.guidance, skip,
                                              args.steps, args.seed + k, args.feather)).save(out)
                rows.append({"split": args.split, "finding": finding, "kind": pair_kind, "direction": "add",
                             "prompt": pair_prompt, "image": film["image"], "patient_id": film["patient_id"],
                             "file": film["file"], "edit_file": out, "mask": mask})
            if (k + 1) % 50 == 0:
                print(f"  {finding}: {k + 1}/{len(chosen)} films edited", flush=True)
    del pipe
    torch.cuda.empty_cache()

    score_pairs(rows, used, args.feather)
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("mask", "file", "edit_file")} for r in rows])
    table.round(4).to_csv(OUTPUTS / f"pairs_{args.split}.csv", index=False)
    checks = ["classifiers_ok", "anatomy_ok", "outside_ok", "valid"]
    summary = table.groupby(["finding", "kind"])[checks].mean().round(3)
    summary.insert(0, "n", table.groupby(["finding", "kind"]).size())
    summary.insert(1, "n_valid", table.groupby(["finding", "kind"])["valid"].sum())
    summary.to_csv(RESULTS / f"pairs_{args.split}_summary.csv")
    print(f"Pairs of the {args.split} split (a valid sham would mean the edit path alone adds the finding):")
    print(summary.to_string())


def tune(args: argparse.Namespace) -> None:
    """One tuning round on val films (TUNE_ROUNDS[args.round]): every config edits the same n films of
    its finding and direction (one per patient), then the three checks score every edit. For
    cardiomegaly the source films must be able to cross CTR 0.5 (by CheXmask): normal films below it
    for additions, cardiomegaly films above it for removals. Both CTRs of the check itself come from
    the segmenter, so its offset from CheXmask cancels."""
    from .checks import used_classifiers, val_films

    used = used_classifiers()  # stops here if a finding has no validated classifier
    torch.backends.cuda.matmul.allow_tf32 = True
    films = load_films("val").merge(val_films()[["image", "ctr"]], on="image")
    sources = {
        ("cardiomegaly", "add"): films[(films["no_finding"] == 1) & (films["ctr"] < 0.5)],
        ("cardiomegaly", "remove"): films[(films["cardiomegaly"] == 1) & (films["ctr"] > 0.5)],
        ("effusion", "add"): films[films["no_finding"] == 1],
        ("effusion", "remove"): films[films["effusion"] == 1],
    }
    sources = {k: v.drop_duplicates("patient_id").sample(n=args.n, random_state=args.seed) for k, v in sources.items()}

    configs = TUNE_ROUNDS[args.round]
    pipe, rows = None, []
    for name, finding, direction, prompt, (kind, param), skip in configs:
        out_dir = OUTPUTS / "radedit_tuning" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, (_, film) in enumerate(sources[(finding, direction)].iterrows()):
            mask = tuning_mask(landmarks_512(film), kind, param)
            out = out_dir / film["image"]
            if not out.exists():  # finished edits are skipped
                if pipe is None:
                    pipe = load_radedit()
                original = Image.open(DATA / "nih512" / film["file"])
                Image.fromarray(edit_film(pipe, original, mask, prompt, args.guidance, skip,
                                          args.steps, args.seed + k, args.feather)).save(out)
            rows.append({"config": name, "finding": finding, "direction": direction, "prompt": prompt,
                         "skip_ratio": skip, "image": film["image"], "patient_id": film["patient_id"],
                         "file": film["file"], "edit_file": out, "mask": mask})
    del pipe
    torch.cuda.empty_cache()

    originals, edits = score_pairs(rows, used, args.feather)
    suffix = "" if args.round == "first" else f"_{args.round}"
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("mask", "file", "edit_file")} for r in rows])
    table.round(4).to_csv(RESULTS / f"radedit_tuning{suffix}.csv", index=False)
    checks = ["classifiers_ok", "anatomy_ok", "outside_ok", "valid"]
    summary = table.groupby("config", sort=False)[checks].mean().round(2)
    summary.insert(0, "n", table.groupby("config", sort=False).size())
    summary.to_csv(RESULTS / f"radedit_tuning{suffix}_summary.csv")
    print(f"Share of edits passing each check (guidance {args.guidance}, {args.steps} steps; "
          f"classifiers used: {used}):")
    print(summary.to_string())

    for name, finding, direction, prompt, _, skip in configs:
        idx = [i for i, r in enumerate(rows) if r["config"] == name]
        labels = ["valid" if rows[i]["valid"] else "fails: " + ", ".join(
            c.removesuffix("_ok") for c in checks[:3] if not rows[i][c]) for i in idx]
        gallery([(originals[i].astype(float), edits[i].astype(float), rows[i]["mask"]) for i in idx],
                f"Tuning: {direction} {finding} (prompt \"{prompt}\", guidance {args.guidance}, skip {skip})",
                RESULTS / f"radedit_tuning_{name}.jpg", labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Addition and sham edit per source film of a split, scored")
    g.add_argument("--split", choices=["test", "train"], required=True)
    g.add_argument("--n-cardiomegaly", type=int, default=380, help="Source films (250 valid at a 0.8 pass rate, + margin)")
    g.add_argument("--n-effusion", type=int, default=600, help="Source films (250 valid at a 0.5 pass rate, + margin)")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--guidance", type=float, default=15.0)
    g.add_argument("--feather", type=float, default=3.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=generate)
    t = sub.add_parser("tune", help="Day-2 tuning round on val films, scored by the three checks")
    t.add_argument("--n", type=int, default=10, help="Films per finding and direction")
    t.add_argument("--steps", type=int, default=100)
    t.add_argument("--round", choices=list(TUNE_ROUNDS), default="first",
                   help="first: additions and removals at skip 0.5; removals: follow-up at skip 0.3 and 0.2")
    t.add_argument("--guidance", type=float, default=15.0, help="Paper: 15; model card: 7.5")
    t.add_argument("--feather", type=float, default=3.0, help="Blur (px) of the paste-back border")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=tune)
    b = sub.add_parser("bugcheck", help="Demonstrate the two quirks of RadEdit's released pipeline")
    b.add_argument("--steps", type=int, nargs="+", default=[4, 16, 64], help="Step counts for check 1")
    b.add_argument("--device", default="cuda", help="cuda, or cpu (then use few steps, e.g. 4 8)")
    b.set_defaults(func=bugcheck)
    s = sub.add_parser("smoke", help="A few RadEdit edits per finding, as galleries")
    s.add_argument("--n", type=int, default=10, help="Films per finding and direction")
    s.add_argument("--steps", type=int, default=100, help="Denoising steps (model card: 100-200)")
    s.add_argument("--guidance", type=float, default=7.5)
    s.add_argument("--skip-ratio", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=smoke)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
