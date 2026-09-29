"""Stage 2: counterfactual chest X-rays with RadEdit. So far:

    python -m cxr.edit smoke     # 10 val films per finding, remove and add -> results/radedit_smoke_*.jpg
    python -m cxr.edit bugcheck  # two quirks of RadEdit's released pipeline -> results/radedit_bugcheck.csv

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


def load_radedit(device: str = "cuda"):
    """RadEdit exactly as on its model card (UNet, SDXL VAE, BioViL-T text encoder, DDIM), then
    RadEdit's own editing pipeline on top (custom code from the Hub, hence trust_remote_code)."""
    from diffusers import (AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline,
                           UNet2DConditionModel)
    from transformers import AutoModel, AutoTokenizer

    unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    text_encoder = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", model_max_length=128,
                                              trust_remote_code=True)
    scheduler = DDIMScheduler(beta_schedule="linear", clip_sample=False, prediction_type="epsilon",
                              timestep_spacing="trailing", steps_offset=1)
    base = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                   scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
                                   feature_extractor=None).to(device)
    return DiffusionPipeline.from_pipe(base, custom_pipeline="microsoft/radedit", trust_remote_code=True)


def gallery(panels: list, title: str, path) -> None:
    """Rows: original, edited, |difference| (x4); one column per film; the edit mask outlined."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
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
