# cxr-grounding

Do medical vision-language models actually look at the X-ray? This project edits chest X-rays with
a diffusion model (remove or add one finding, nothing else), checks each edit with anatomy
(cardiothoracic ratio from a segmentation network), then tests whether MedGemma's answer follows
the image. Finally it fine-tunes MedGemma with LoRA on the edited pairs.

> Research project on public data. Not for clinical use.

Plan, metrics and progress: [PLAN.md](PLAN.md).

## Results

| Week | Result | Files |
|---|---|---|
| 1 | Physiology check: cardiothoracic ratio of cardiomegaly vs normal films, PA vs AP views | `results/ctr_by_view.png`, `results/ctr_summary.csv` |

## Layout

```
cxr/                all the Python, one file per stage, run as: python -m cxr.<stage>
  __init__.py       shared paths
  data.py           1. NIH subset, CheXmask masks, cardiothoracic ratio (CTR)
  seg.py            2. heart/lung U-Net                                   (week 2)
  edit.py           2. RadEdit counterfactual pairs + validation checks   (week 2)
  audit.py          3. MedGemma P(yes), flip rate, controls               (week 3)
  lora.py           4. LoRA fine-tuning + re-evaluation                   (week 4)
slurm/
  setup_env.sh      creates .venv (once, on the cluster login node)
  data.sbatch       stage 1 in one job: downloads, CTR + figure, NIH subset
  job.sbatch        any GPU run: sbatch slurm/job.sbatch -m cxr.<stage> [options]
results/            small tables and figures (in git)
PLAN.md             plan and lab notebook
```

Only on the cluster, not in git: `data/` (datasets), `outputs/` (per-image tables, checkpoints,
generated images), `logs/` (Slurm logs), `.venv/`.

## Run it (Slurm cluster)

```bash
git clone git@github.com:m871-akram/cxr-grounding.git ~/cxr-grounding && cd ~/cxr-grounding
bash slurm/setup_env.sh                        # once: Python environment
source .venv/bin/activate && hf auth login     # once: gated models (RadEdit, MedGemma)
sbatch slurm/data.sbatch                       # stage 1: data, CTR figure, 512-px subset
```

Every partition stops jobs after 4 h: long stages save checkpoints and resume when resubmitted.
If the project is not in `~/cxr-grounding`, set `export PROJECT_DIR=/path/to/cxr-grounding` first.

## Data and models

- **NIH ChestX-ray14** (Wang et al., CVPR 2017), from Kaggle `nih-chest-xrays/data` (needs a
  Kaggle API token in `~/.kaggle/kaggle.json`). PA views only for the edits: the CTR rule does not
  hold on AP films.
- **CheXmask** v1.0 (Gaggion et al., Scientific Data 2024; PhysioNet, CC BY 4.0): heart and lung
  contours for every NIH image. The job downloads the NIH file from a Hugging Face mirror and checks
  it against PhysioNet's official SHA-256.
- **RadEdit** (`microsoft/radedit`): research use only; weights are not redistributed.
- **MedGemma** (`google/medgemma-4b-it`): Health AI Developer Foundations terms.
- **TorchXRayVision** classifiers as independent checks of the edits.
- **VQA-RAD** and **SLAKE** (Hugging Face) to check that fine-tuning does not hurt general answers.
