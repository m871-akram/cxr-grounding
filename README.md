# cxr-grounding

**Do medical vision-language models actually look at the X-ray?**

A model that answers "is there cardiomegaly?" correctly could be reading the image, or it could be
guessing from the question. To tell the two apart, I take a normal chest X-ray, **add** a finding to
it with a diffusion editor (RadEdit), and ask MedGemma the same yes/no question before and after.
If the model looks at the image, its answer should change from "no" to "yes".

> Research project on public data (NIH ChestX-ray14). Not for clinical use.

## Key results

- **MedGemma does use the image.** Edits that pass my checks flip MedGemma 4B's answer from "no" to
  "yes" in 98.6-100% of cases.
- **But it also reacts to *how* the editor paints.** When RadEdit is asked for cardiomegaly inside
  the heart's own outline, the heart gets denser but not wider. The share of "yes" still rises from
  14% to 39% (a sham edit with the same mask: 12%).
- **Training on edits partly transfers to real films.** Fine-tuning MedGemma 4B (LoRA) only on
  edited images recovers about two-thirds of the gain obtained by training on real films (AUROC
  0.900 -> 0.925, against 0.937). Adding the edits to the real films brings nothing extra.

So a flip on a generated counterfactual shows that a model *uses the image*, not that it *reads the
anatomy*.

## How it works

1. **Data.** Frontal (PA) films from NIH ChestX-ray14, at 512 px, with heart and lung contours from
   CheXmask. Splits are by patient.
2. **Edit.** RadEdit adds cardiomegaly (a heart enlarged toward the apex) or pleural effusion (fluid
   at the lung bases) inside a mask drawn from the contours. Outside the mask, the original pixels
   are pasted back.
3. **Check every edit.** An edit is kept only if:
   - an independent classifier (TorchXRayVision) sees the finding more than on the original;
   - the anatomy agrees: the cardiothoracic ratio (CTR, heart width / chest width) crosses 0.5, or
     the aerated lung at the base shrinks by at least 10%;
   - nothing changed outside the mask.
4. **Ask MedGemma.** The score is P(yes), read from the logits of "yes" vs "no" at the first answer
   token, so no text parsing is needed. An answer *flips* when P(yes) goes from below 0.5 to above.
5. **Controls.** *Sham* edits use the same mask and pipeline with a "normal" prompt: they should not
   flip. Blank and pixel-shuffled images should get no "yes".

All CIs are 95% bootstrap intervals that resample patients. The plan, the rules fixed before each
result, and a dated lab notebook are in [PLAN.md](PLAN.md).

## Results: the audit

Test split, one film per patient. 308 of 380 cardiomegaly edits and 179 of 600 effusion edits passed
the checks (`results/pairs_test_summary.csv`). Flips at 0.5, among pairs where the original was
answered "no" (`results/audit_flips.csv`):

| Model | Question | Edits that flip | Shams that flip |
|---|---|---|---|
| MedGemma 4B | Is there cardiomegaly? | 266/266 (100%) | 20.3% |
| MedGemma 4B | Is the heart enlarged? | 289/293 (98.6%) | 9.2% |
| MedGemma 4B | Is there pleural effusion? | 168/169 (99.4%) | 2.4% |
| MedGemma 4B | Is there fluid in the pleural space? | 163/163 (100%) | 4.9% |
| MedGemma 1.5 | Is there cardiomegaly? | 265/265 (100%) | 22.6% |
| MedGemma 1.5 | Is the heart enlarged? | 284/285 (99.6%) | 11.9% |
| MedGemma 1.5 | Is there pleural effusion? | 163/163 (100%) | 4.3% |
| MedGemma 1.5 | Is there fluid in the pleural space? | 158/158 (100%) | 3.8% |

No blank or shuffled image gets P(yes) > 0.5 (980 of each). Cardiomegaly shams flip more often
because some of them enlarge the heart a little without reaching CTR 0.5.

**What does the model react to?** These analyses were decided after the audit, so they are
exploratory:
- **Denser, not wider.** In the heart's own outline, RadEdit makes the heart brighter (+12 gray
  levels) without widening it (median CTR 0.430 -> 0.434), and MedGemma 4B's "yes" rate goes from
  14% to 39% (sham: 12%). After the smallest enlargement (CTR 0.456, still normal) it reaches 54%
  (`results/followup_heart_intensity.csv`, `dose_effect_by_growth.csv`, `followup_cardio_shams.csv`).
- **Two phrasings, two answers.** After the smallest enlargement, on 31% of the films 4B says "yes" to "Is there cardiomegaly?"
  but "no" to "Is the heart enlarged?" (16% of real films; never the reverse;
  `results/dose_phrasing_contradiction.csv`).
- **Edited films look sicker than real ones.** At the same measured CTR, edited films get a higher
  P(yes) than real films (CI above 0 in 15 of 16 comparisons; `results/dose_matched_ctr.csv`).
- **Effusion too.** The thinnest effusion edit raises 4B's "yes" rate from 3% to 56%, while the lung
  shrinks by only 2.6%, below my own 10% check (`results/followup_effusion_dose.csv`).

![P(yes) against measured CTR for real and edited films](results/dose_psychometric.png)

## Results: can fine-tuning on edits teach the model? (H3)

Pre-registered in [PLAN.md](PLAN.md) before any training. MedGemma 4B was fine-tuned with LoRA on
"Is there cardiomegaly in this image?", with the label CTR > 0.5, in three ways: on **real films
(R)**, on **edits only (E)**, or on **both (RE)**, three seeds each. It was tested once, at the end,
on patients kept aside for it: 2,335 real films of 1,187 patients, plus edits, shams and blanks made
from 100 of their normal films.

Mean of 3 seeds (range), `results/h3_arms_fresh.csv`:

| | Base | R: real | E: edits only | RE: both |
|---|---|---|---|---|
| Real films, AUROC for CTR > 0.5 (primary) | 0.900 | 0.937 (0.934-0.940) | 0.925 (0.920-0.927) | 0.937 (0.936-0.937) |
| Real films, AUROC against NIH labels | 0.931 | 0.935 (0.932-0.938) | 0.931 (0.921-0.936) | 0.933 (0.932-0.935) |
| Edits with CTR <= 0.5 answered "yes" | 50% | 28% (23-33%) | 9% (7-10%) | 13% (9-15%) |
| Shams answered "yes" | 19% | 16% (16-17%) | 5% (4-6%) | 9% (8-10%) |
| Blank images answered "yes" | 0% | 0% | 0% | 0% |

("Answered yes" = P(yes) > 0.5.)

- **H3.1, "edits help real films much less than real films do": inconclusive.** E's gain on real
  films (+0.025, CI 0.015 to 0.035) is 0.67 of R's (+0.037, CI 0.025 to 0.050). I predicted less
  than half. The estimate leans the other way, but the test cannot rule out half
  (`results/h3_contrasts_fresh.csv`, `h3_verdicts_fresh.csv`).
- **H3.2, "adding edits to real films doesn't help": supported.** RE - R = -0.0005 AUROC (CI -0.005
  to +0.004), below the 0.02 margin fixed in advance.
- **Why a fresh test set matters:** on the test split I had already looked at, E seemed to recover
  0.79 of R's gain, against 0.67 on the fresh patients (`results/h3_verdicts_test.csv`).
- No fine-tuned model says "yes" to a blank image, and effusion AUROC does not drop.

## Limitations

- **Only additions.** RadEdit could not remove findings convincingly, so removals were dropped.
- **The checks are models too.** The classifiers and the segmenter can be fooled. For example, the
  effusion check passes 22% of images whose lung bases were only blurred
  (`results/followup_effusion_dose.csv`). The segmenter's CTR agrees with CheXmask at r = 0.893
  (`results/segmenter_ctr.csv`).
- **MedGemma 1.5's score is not its answer.** It rarely starts its reply with yes or no, so for it
  P(yes) is a ranking score. For MedGemma 4B, the score agrees with the written answer in 98.9-100%
  of cases (`results/decode_4b.csv`, `decode_1.5-4b.csv`).
- **RadEdit saw NIH films in training.** Its split was patient-disjoint but had no test set, so some
  of my patients may have been in it.
- **RadEdit's released pipeline has two bugs** in its keep-mask handling. I paste the original
  pixels back, which removes their effect outside the mask. A reproduction and a fix (proposed
  upstream) are in [`radedit/`](radedit/) and `results/radedit_repro.txt`.
- **A tokenization bug, fixed.** Early scores had two `<bos>` tokens. Everything here comes from the
  corrected re-score (effect of the bug: `results/bos_check.csv`).
- **H3:** the label and the main metric both come from the segmenter's CTR. Cardiomegaly only, one
  editor, one model, three seeds. LoRA trains only the language model, with the vision encoder
  frozen.
- **Scope:** one dataset, two findings, two models.

## Layout

```
cxr/          the pipeline, one file per stage, run as python -m cxr.<stage>
  data.py       NIH subset, CheXmask contours, CTR, patient splits
  checks.py     TorchXRayVision classifiers and segmenter for the edit checks
  edit.py       RadEdit additions, shams and the three checks
  audit.py      MedGemma scores, flip rule, controls
  dose.py       dose-response, blur control, follow-up measures
  h3.py         H3 training images
  lora.py       H3 LoRA training and evaluation
  reader.py     pilot reader study (image selection, rating page)
  provenance.py records of images, manifests and environment
notebooks/    analysis on the Mac, reading only results/
radedit/      standalone reproduction of the RadEdit pipeline bugs
pod/          scripts for the RunPod GPU pod
results/      small tables and figures
PLAN.md       plan, pre-registrations and lab notebook
```

`data/` and `outputs/` (datasets and generated images) stay on the pod, not in git.

## Run it (RunPod)

One GPU pod from the Runpod PyTorch 2.8.0 template, a network volume at `/workspace`, and the pod's
SSH address in `~/.ssh/config` as host `cxr-pod`.

```bash
bash pod/sync.sh push                                    # Mac: code -> pod
ssh cxr-pod bash /workspace/cxr-grounding/pod/setup.sh   # every new pod
ssh -t cxr-pod 'source /workspace/cxr-grounding/pod/env.sh && hf auth login'   # once: gated models
R="bash /workspace/cxr-grounding/pod/run.sh"             # long runs in tmux, logged
ssh cxr-pod "$R data 'bash pod/data.sh'"                                    # data, CTR, 512-px subset
ssh cxr-pod "$R checks 'python -m cxr.checks validate'"                     # checks on real films
ssh cxr-pod "$R generate 'python -m cxr.edit generate --split test'"        # edits and shams
ssh cxr-pod "$R audit 'python -m cxr.audit run --model 4b && python -m cxr.audit run --model 1.5-4b'"
ssh cxr-pod "$R dose 'python -m cxr.dose run && python -m cxr.dose followup'"
ssh cxr-pod "$R h3gen 'python -m cxr.h3 generate && python -m cxr.h3 measure'"   # H3 training data
ssh cxr-pod "$R h3train 'python -m cxr.lora train --arm R --seed 0'"             # each arm R, E, RE, seed 0-2
ssh cxr-pod "$R h3final 'python -m cxr.lora runs && python -m cxr.lora evaluate --model base --set fresh --final'"  # then each model
bash pod/sync.sh pull                                    # Mac: results/ <- pod, then run the notebooks
```

Every stage skips finished work, so an interrupted run can just be restarted. Most stages consider
work finished when its output files exist, so delete a stage's outputs before rerunning it with
other settings. H3's generation and final evaluation check settings, code and inputs by hash.

## License, data and models

Code under the MIT License ([LICENSE](LICENSE)). No data or model weights are included.

- **NIH ChestX-ray14** (Hugging Face mirror `alkzar90/NIH-Chest-X-ray-dataset`): Wang X, Peng Y,
  Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks
  on weakly-supervised classification and localization of common thorax diseases. CVPR 2017.
- **CheXmask** v1.0 (PhysioNet, CC BY 4.0): Gaggion N, Mosquera C, Mansilla L, et al. CheXmask: a
  large-scale dataset of anatomical segmentation masks for multi-center chest x-ray images.
  Scientific Data 11, 511 (2024).
- **RadEdit** (`microsoft/radedit`): research use only, weights not redistributed.
- **MedGemma** (`google/medgemma-4b-it`, `google/medgemma-1.5-4b-it`): Health AI Developer
  Foundations terms.
- **TorchXRayVision**: PadChest- and CheXpert-trained classifiers and the PSPNet segmenter. The
  segmenter was trained on 1,000 images from unnamed external data, so an overlap with NIH can be
  neither ruled out nor confirmed.
