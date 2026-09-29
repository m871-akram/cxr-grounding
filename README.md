# cxr-grounding

Do medical vision-language models actually look at the X-ray? This project edits chest X-rays with
a diffusion model (remove or add one finding, nothing else), checks each edit with anatomy
(cardiothoracic ratio from a segmentation network), then tests whether MedGemma's answer follows
the image. Finally it fine-tunes MedGemma with LoRA on the edited pairs.

> Research project on public data. Not for clinical use.

Plan, metrics and progress: [PLAN.md](PLAN.md).

## Results

| Day | Result | Files |
|---|---|---|
| 1 | Physiology check: cardiothoracic ratio of cardiomegaly vs normal films, PA vs AP views | `results/ctr_by_view.png`, `results/ctr_summary.csv` |

## Layout

```
cxr/                all the Python, one file per stage, run as: python -m cxr.<stage>
  __init__.py       shared paths
  data.py           1. NIH subset, CheXmask landmarks, cardiothoracic ratio (CTR)
  edit.py           2. RadEdit counterfactual pairs + validation checks   (day 2)
  audit.py          3. MedGemma P(yes), flip rate, controls               (day 3)
  lora.py           4. LoRA fine-tuning + re-evaluation                   (day 4)
pod/                the GPU pod (RunPod)
  sync.sh           on the Mac: code to the pod (push), results/ back (pull)
  setup.sh          on the pod: tmux, Python environment on the network volume
  env.sh            on the pod: environment of every command (venv, Hugging Face cache)
  run.sh            on the pod: any long run inside tmux, logged in /workspace/logs
  data.sh           on the pod: stage 1 (downloads, CTR figure, 512-px subset)
results/            small tables and figures (in git)
PLAN.md             plan and lab notebook
```

Only on the pod's network volume, not in git: `data/` (datasets), `outputs/` (per-image tables,
checkpoints, generated images), the Python environment, the Hugging Face cache and the logs.

## Run it (RunPod)

One GPU pod from the Runpod PyTorch 2.8.0 template (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`)
with a network volume mounted at `/workspace`. Code, Python environment, data, Hugging Face cache and
logs all live on the volume, so the pod can be deleted at the end of a session and a new one created
the next day. Put the pod's SSH address in `~/.ssh/config` as host `cxr-pod` (the IP and port change
with every pod), then:

```bash
bash pod/sync.sh push                                    # Mac: code -> pod
ssh cxr-pod bash /workspace/cxr-grounding/pod/setup.sh   # every new pod (first time: Python env)
ssh -t cxr-pod 'source /workspace/cxr-grounding/pod/env.sh && hf auth login'  # once: gated models
ssh cxr-pod 'bash /workspace/cxr-grounding/pod/run.sh data "bash pod/data.sh"'  # stage 1, in tmux
ssh cxr-pod tail -f /workspace/logs/data.log             # follow the run
bash pod/sync.sh pull                                    # Mac: results/ <- pod
```

Big downloads (NIH zips, CheXmask) stay on the pod's container disk and are deleted after use; only
the ~2 GB image subset is kept. Every stage skips finished work, so an interrupted run is simply
started again.

## Data and models

- **NIH ChestX-ray14** (Wang et al., CVPR 2017). Labels (`Data_Entry_2017_v2020.csv`) and images
  from a Hugging Face mirror of the NIH release (`alkzar90/NIH-Chest-X-ray-dataset`; 12 image zips
  read one at a time on the pod's container disk, so only the ~2 GB subset is stored). PA views only
  for the edits: the CTR rule does not hold on AP films.
- **CheXmask** v1.0 (Gaggion et al., Scientific Data 2024; PhysioNet, CC BY 4.0): heart and lung
  contours for every NIH image. `pod/data.sh` downloads the NIH file from a Hugging Face mirror,
  checks it against PhysioNet's official SHA-256 and keeps only the landmark columns (~115 MB).
- **RadEdit** (`microsoft/radedit`): research use only; weights are not redistributed.
- **MedGemma** (`google/medgemma-4b-it`): Health AI Developer Foundations terms.
- **TorchXRayVision** classifiers as independent checks of the edits.
- **VQA-RAD** and **SLAKE** (Hugging Face) to check that fine-tuning does not hurt general answers.
