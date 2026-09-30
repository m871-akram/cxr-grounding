# cxr-grounding

Do medical vision-language models look at the X-ray? This project adds a finding (cardiomegaly or
pleural effusion) to normal chest X-rays with a diffusion editor (RadEdit), keeps only the edits
that pass automatic checks, and asks MedGemma the same yes/no question before and after the edit.

> Research project on public data. Not for clinical use.

**MedGemma's answers depend on the X-ray: of the edits that passed our automatic checks and started
from a "no", 98.6-100% flip its answer to "yes". But it also reacts to how the editor paints a
finding, not only to the anatomy the edit was meant to change, so flip rates on generated
counterfactuals show that a model uses the image, not that it reads the anatomy.**

An edit *passes our automatic checks* when an independent classifier (TorchXRayVision; the
PadChest-trained model, and for effusion also the CheXpert-trained one) scores it above its threshold
and higher than the original film, the anatomy agrees (cardiomegaly: the cardiothoracic ratio, CTR,
crosses 0.5; effusion: the aerated lung in the lung-base mask shrinks by at least 10%), and a
compositing check finds no change outside the edit mask (true by construction: the original pixels
are pasted back). Plan, pre-registered rules and lab notebook: [PLAN.md](PLAN.md).

## Results

MedGemma's score is P(yes) = sigmoid(d), with d the log-odds of the yes tokens against the no tokens
at the first answer position (float32, unrounded). Test split of NIH ChestX-ray14 (PA films), one
source film per patient; 308 of 380 cardiomegaly additions and 179 of 600 effusion additions passed
the checks (`results/pairs_test_summary.csv`).

**Flip rates (pre-registered, `results/audit_flips.csv`).** Pairs that passed the checks and whose
original answer is "no" (P(yes) < 0.5); a pair flips when the edited film's P(yes) is above 0.5.
Shams use the same mask and edit path with a normal prompt.

| Model | Question | Additions that flip | One-sided 95% lower bound | Shams that flip |
|---|---|---|---|---|
| MedGemma 4B | Is there cardiomegaly? | 266/266 (100%) | 98.9% | 20.3% |
| MedGemma 4B | Is the heart enlarged? | 289/293 (98.6%) | 96.9% | 9.2% |
| MedGemma 4B | Is there pleural effusion? | 168/169 (99.4%) | 97.2% | 2.4% |
| MedGemma 4B | Is there fluid in the pleural space? | 163/163 (100%) | 98.2% | 4.9% |
| MedGemma 1.5 | Is there cardiomegaly? | 265/265 (100%) | 98.9% | 22.6% |
| MedGemma 1.5 | Is the heart enlarged? | 284/285 (99.6%) | 98.3% | 11.9% |
| MedGemma 1.5 | Is there pleural effusion? | 163/163 (100%) | 98.2% | 4.3% |
| MedGemma 1.5 | Is there fluid in the pleural space? | 158/158 (100%) | 98.1% | 3.8% |

With the primary pre-registered threshold (the Youden threshold on validation films) every eligible
addition flips, in all 8 rows (n = 137-274). Neither model says "yes" to any of the 980 blank or 980
shuffled images (largest P(yes) on a blank: 0.0045; `results/audit_controls.csv`).

**What the edits change (exploratory, decided after the audit).**
- Asked for cardiomegaly inside the heart's own outline, RadEdit makes the heart denser (+12 gray
  levels inside the outline) but not wider (median CTR 0.430 -> 0.434), and the share of films
  MedGemma-4B calls cardiomegaly rises from 14% to 39% (sham, same mask: 12%). After the smallest
  enlargement (CTR 0.456, still normal) it reaches 54% (`results/followup_heart_intensity.csv`,
  `dose_effect_by_growth.csv`, `followup_cardio_shams.csv`).
- On 31% of those films it answers "yes" to "Is there cardiomegaly?" but "no" to "Is the heart
  enlarged?", against 16% of real films, and never the reverse
  (`results/dose_phrasing_contradiction.csv`).
- At the same measured CTR, edited films get a higher P(yes) than real films (CI above 0 in 15 of 16
  comparisons; `results/dose_matched_ctr.csv`).
- The thinnest effusion edit raises the share called effusion from 3% to 56% while the aerated lung
  shrinks by 2.6%, below our own 10% check (`results/followup_effusion_dose.csv`).

![P(yes) against measured CTR for real and edited films](results/dose_psychometric.png)

## Limitations

- **MedGemma 1.5's score is a ranking score, not its answer.** It rarely starts its answer with yes
  or no (4-33% of real and edited films); read at the first yes or no of its full answer, the answer
  agrees with its score for 78-93% of films. MedGemma 4B agrees for 98.9-100%
  (`results/decode_1.5-4b.csv`, `decode_4b.csv`).
- **The effusion screen is not specific to fluid**: it passes 22% of blur-only images (a Gaussian
  blur of the lung bases, sigma 8 px, no RadEdit; `results/followup_effusion_dose.csv`).
- **A tokenization bug, fixed and measured.** Until 2026-09-30 every MedGemma input carried two <bos>
  tokens. Re-scored with one: for 4B, AUROC moved by at most 0.013 and 97-99% of answers stayed on the
  same side of 0.5; for 1.5 the log-odds fell by 1.1-1.5 on average and 3.5-10.5% of answers changed
  side (`results/bos_check.csv`). Every number here comes from the re-scored run.
- **RadEdit's released pipeline** ignores the keep mask and, outside the edit mask, pastes back a
  latent one timestep too noisy (`results/radedit_bugcheck.csv`). We paste the original pixels back
  after decoding, which removes both effects outside the mask but not inside it. RadEdit was trained
  on NIH ChestX-ray14 without a held-out split, so it may have seen our films.
- **The test split is exploratory**: the edit settings and the analyses above were chosen after
  looking at it. The next experiment (H3) keeps a fresh test set for that reason.
- Only additions: removals failed the checks (PLAN.md, lab notebook). The checks rely on models
  (classifiers, a segmenter whose CTR agrees with CheXmask at r = 0.893 on real films,
  `results/segmenter_ctr.csv`). One dataset, two findings, two models.

## Next: H3

Does LoRA fine-tuning on counterfactuals transfer to real films? Pre-registered in PLAN.md (section
4) before any training: MedGemma 4B trained on real films, on RadEdit edits, or on both, and tested
on patients reserved for it.

## Layout

```
cxr/                all the Python, one file per stage, run as: python -m cxr.<stage>
  __init__.py       shared paths
  data.py           NIH subset, CheXmask landmarks, cardiothoracic ratio (CTR), patient splits
  checks.py         edit checks: TorchXRayVision classifiers and segmenter, validated on real films
  edit.py           RadEdit additions and shams, masks from CheXmask, the three checks
  audit.py          MedGemma scores (log-odds), flip rule, controls, decode and <bos> checks
  dose.py           dose-response, blur control, follow-up measures
  h3.py             H3 training data (manifests, pinned model revisions)
notebooks/          analysis on the Mac, reading only results/ (01 data, 03 audit, 03b dose, 03c follow-up, 04 H3)
pod/                the GPU pod (RunPod): sync, setup, run in tmux, watchdog, provenance
results/            small tables and figures (in git)
PLAN.md             plan, pre-registrations and lab notebook
```

Only on the pod's network volume, not in git: `data/` (datasets), `outputs/` (per-image tables,
generated images), the Python environment, the Hugging Face cache and the logs.

## Run it (RunPod)

One GPU pod from the Runpod PyTorch 2.8.0 template (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`)
with a network volume at `/workspace`; put the pod's SSH address in `~/.ssh/config` as host `cxr-pod`.

```bash
bash pod/sync.sh push                                    # Mac: code -> pod
ssh cxr-pod bash /workspace/cxr-grounding/pod/setup.sh   # every new pod
ssh -t cxr-pod 'source /workspace/cxr-grounding/pod/env.sh && hf auth login'   # once: gated models
R="bash /workspace/cxr-grounding/pod/run.sh"             # long runs in tmux, logged in /workspace/logs
ssh cxr-pod "$R data 'bash pod/data.sh'"                                    # data, CTR, 512-px subset
ssh cxr-pod "$R checks 'python -m cxr.checks validate'"                     # checks on real films
ssh cxr-pod "$R generate 'python -m cxr.edit generate --split test'"        # additions and shams
ssh cxr-pod "$R audit 'python -m cxr.audit run --model 4b && python -m cxr.audit run --model 1.5-4b'"
ssh cxr-pod "$R dose 'python -m cxr.dose run && python -m cxr.dose followup'"
bash pod/sync.sh pull                                    # Mac: results/ <- pod, then run the notebooks
```

Every stage skips finished work, so an interrupted run is simply started again.

## Data and models

- **NIH ChestX-ray14** (Wang et al., CVPR 2017), from a Hugging Face mirror of the NIH release
  (`alkzar90/NIH-Chest-X-ray-dataset`); PA films only.
- **CheXmask** v1.0 (Gaggion et al., Scientific Data 2024; PhysioNet, CC BY 4.0): heart and lung
  contours; the downloaded file is checked against PhysioNet's SHA-256.
- **RadEdit** (`microsoft/radedit`): research use only; weights are not redistributed.
- **MedGemma** (`google/medgemma-4b-it`, `google/medgemma-1.5-4b-it`): Health AI Developer
  Foundations terms.
- **TorchXRayVision**: PadChest- and CheXpert-trained classifiers and the ChestX-Det segmenter, as
  independent checks of the edits.
