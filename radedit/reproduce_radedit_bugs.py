"""Reproduce two bugs in RadEdit's editing pipeline (microsoft/radedit, pipeline.py).

    python reproduce_radedit_bugs.py --image 00000001_000.png                      # the Hub's pipeline.py
    python reproduce_radedit_bugs.py --image 00000001_000.png --pipeline ./patched  # a folder with a patched pipeline.py
    python reproduce_radedit_bugs.py --image 00000001_000.png --compare-with ./patched  # also: no keep mask, Hub vs patched

Any frontal NIH ChestX-ray14 film works; it is resized to 512 x 512. The edit mask is a fixed box
over the heart; "far outside" means more than 32 px from it (the VAE decoder mixes nearby pixels).

1. One-step-late paste-back. After each step the pipeline should paste back, inside the keep mask,
   the inverted latent of the timestep just reached (Algorithm 3 of the paper), so that the kept
   region ends as the film's own latent x0, i.e. its VAE round trip (encode, then decode). The
   released code pastes back the latent one timestep noisier, so noise is left there, more with fewer
   steps.
2. Keep mask ignored. A second block pastes the inverted latent back everywhere outside the edit mask,
   so keep = 1 - edit and an empty keep mask give the same image, while the paper lets the region
   outside both masks change ("we allow the area outside m_edit to be modified").
3. With --compare-with: without a keep mask, the Hub's pipeline and the other one (e.g. the fixed
   pipeline.py) should give identical images, since the fix only changes what is pasted back inside
   a keep mask.
"""
import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REVISION = "e8ebd31396ff8553c34084b98b5defb8ebea2817"  # microsoft/radedit main as of 2026-10-01


def load(pipeline: str, device: str, base=None):
    """RadEdit as on its model card (UNet, SDXL VAE, BioViL-T, DDIM), with its editing pipeline on top;
    pass `base` to put a second editing pipeline on the same models."""
    if base is None:
        unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet", revision=REVISION)
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        text_encoder = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", model_max_length=128, trust_remote_code=True)
        scheduler = DDIMScheduler(beta_schedule="linear", clip_sample=False, prediction_type="epsilon",
                                  timestep_spacing="trailing", steps_offset=1)
        base = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                       safety_checker=None, requires_safety_checker=False, feature_extractor=None).to(device)
    pinned = {"custom_revision": REVISION} if pipeline == "microsoft/radedit" else {}
    return DiffusionPipeline.from_pipe(base, custom_pipeline=pipeline, trust_remote_code=True, **pinned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="a frontal chest X-ray (PNG), e.g. from NIH ChestX-ray14")
    parser.add_argument("--pipeline", default="microsoft/radedit", help="the Hub's pipeline, or a folder with pipeline.py")
    parser.add_argument("--compare-with", help="a second pipeline (folder with pipeline.py): check 3")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pipe = load(args.pipeline, args.device)
    film = Image.open(args.image).convert("RGB").resize((512, 512))
    edit = np.zeros((512, 512), bool)
    edit[230:430, 150:360] = True                      # a box over the heart
    near = np.zeros_like(edit)
    near[230 - 32:430 + 32, 150 - 32:360 + 32] = True
    far = ~near                                        # far outside the edit mask
    nothing = np.zeros_like(edit)
    as_image = lambda m: None if m is None else Image.fromarray(m.astype(np.uint8) * 255)

    def round_trip():
        torch.manual_seed(0)                           # the pipeline samples the latent x0 first, with this seed
        with torch.no_grad():
            x0 = pipe.encode_images_vae(pipe.image_processor.preprocess(film))
            image = pipe.vae.decode(x0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
        out = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]
        return np.asarray(out.convert("L"), dtype=float)

    def run(prompt, guidance, steps, edit_mask, keep_mask, pipeline=None):
        torch.manual_seed(0)                           # the same inversion noise in every run
        out = (pipeline or pipe)(prompt, weights=[guidance], image=film, edit_mask=as_image(edit_mask), keep_mask=as_image(keep_mask),
                   num_inference_steps=steps, invert_prompt="", skip_ratio=0.3, output_type="pil")[0]
        return np.asarray(out.convert("L"), dtype=float)

    print(f"pipeline: {args.pipeline}  (mean absolute difference, gray levels 0-255, far outside the edit mask)")
    reference = round_trip()
    for steps in args.steps:                           # empty prompt, guidance 1: the box is only reconstructed
        kept = run("", 1.0, steps, edit, ~edit)
        print(f"1. keep = 1 - edit vs the film's VAE round trip, {steps:3d} steps: {np.abs(kept - reference)[far].mean():7.3f}"
              "   (expected ~0)")
    steps = max(args.steps)
    kept = run("Cardiomegaly", 7.5, steps, edit, ~edit)
    empty = run("Cardiomegaly", 7.5, steps, edit, nothing)
    none = run("Cardiomegaly", 7.5, steps, edit, None)
    print(f"2. keep = 1 - edit vs empty keep mask, {steps} steps:           {np.abs(kept - empty)[far].mean():7.3f}"
          "   (expected > 0: an empty keep mask keeps nothing)")
    print(f"   empty keep mask vs no keep mask, {steps} steps:            {np.abs(empty - none)[far].mean():7.3f}"
          "   (expected 0: both keep nothing)")
    if args.compare_with:
        other = load(args.compare_with, args.device, base=pipe)
        other_none = run("Cardiomegaly", 7.5, steps, edit, None, pipeline=other)
        diff = np.abs(other_none - none)
        print(f"3. no keep mask, {args.pipeline} vs {args.compare_with}, {steps} steps: everywhere {diff.mean():7.3f}, "
              f"largest {diff.max():.0f}   (expected 0: identical images)")


if __name__ == "__main__":
    main()
