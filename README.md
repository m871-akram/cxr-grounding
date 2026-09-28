# cxr-grounding

Do medical vision-language models actually look at the X-ray? This project edits chest X-rays with
a diffusion model (remove or add one finding, nothing else), checks the edits with anatomy
(cardiothoracic ratio from a segmentation network), then tests whether MedGemma's answers follow
the image. Finally it fine-tunes MedGemma with LoRA on the edited pairs.

The full plan is in [PLAN.md](PLAN.md).

> Research project, not for clinical use.

## Layout

```
PLAN.md                     week-by-week plan, metrics, risks
requirements.txt            Python dependencies (install PyTorch first, see below)
setup/create_env.sh         creates .venv on the cluster
scripts/check_cluster.sh    quota, partitions, GPU, internet access
scripts/smoke_test.py       GPU test (fp32/fp16/bf16 speed)
scripts/prepare_nih_subset.py   NIH ChestX-ray14 -> 512 px subset + metadata, split by patient
slurm/job.sbatch            GPU job template: sbatch slurm/job.sbatch <script.py> [args]
slurm/download_nih.sbatch   CPU job: Kaggle download to /tmp -> data/nih512
logs/                       Slurm logs
data/ outputs/ .cache/      not tracked by git
```

## Quick start on Ensicompute

On your Mac:

```bash
git init -b main && git add . && git commit -m "Starter"
# create an empty private repo on GitHub, then:
git remote add origin git@github.com:m871-akram/cxr-grounding.git && git push -u origin main
```

On nash (VPN if off campus). To clone a private repo from nash, add an SSH key generated on nash
to your GitHub account (`ssh-keygen -t ed25519`, then paste `~/.ssh/id_ed25519.pub` in GitHub settings).

```bash
git clone git@github.com:m871-akram/cxr-grounding.git ~/cxr-grounding   # or another folder with space
cd ~/cxr-grounding

# 1. What do I have? (quota, partitions, internet)
bash scripts/check_cluster.sh | tee logs/check_nash.txt
srun --gres=shard:1 --cpus-per-task=2 --mem=4GB bash scripts/check_cluster.sh | tee logs/check_node.txt

# 2. Environment (once)
bash setup/create_env.sh

# 3. GPU smoke test (edit the partition name in slurm/job.sbatch first: see `sinfo`)
sbatch slurm/job.sbatch scripts/smoke_test.py
squeue -u $USER          # then read logs/cxr-<jobid>.out
```

If the project lives outside `~/cxr-grounding`, set `export PROJECT_DIR=/path/to/cxr-grounding`
before running the scripts.

## Data

- **NIH ChestX-ray14** (Kaggle `nih-chest-xrays/data`, ~45 GB). `sbatch slurm/download_nih.sbatch`
  downloads it into the node's `/tmp`, reads the zip directly (no unzip) and keeps only the 512-px
  subset (~5 GB) in `data/nih512`. Needs a Kaggle API token in `~/.kaggle/kaggle.json`.
- **CheXmask** heart/lung masks: PhysioNet `chexmask-cxr-segmentation-data` (open access); only the
  ChestX-ray8 file is needed.
- **VQA-RAD** and **SLAKE** (Hugging Face) for the regression check after fine-tuning.

## Where things live

| Place | Role |
|---|---|
| Mac (Claude Code) | Write code, small tests on a few images, figures, writing. Push to GitHub. |
| GitHub (private repo) | The only way code moves between the Mac and the cluster. |
| Ensicompute | Data, model weights and every GPU job. Pull the repo, submit with `sbatch`. |
| Google Drive (2 TB) | Archive only: the processed subset as one tarball, final checkpoints, results. Never train from it. |

For interactive debugging (notebooks on a GPU), use VS Code Remote-SSH to a `vmgpuXXX.ensimag.fr`
machine, as described in the Ensicompute guide. Long runs go through `sbatch`.

Google Drive from the cluster, if it has internet access: install the `rclone` binary in `~/bin`,
run `rclone config` (choose Google Drive, answer "n" to auto config), then run
`rclone authorize "drive"` on the Mac and paste the token back. Afterwards:
`rclone copy data/nih512.tar gdrive:cxr-grounding/`.

## Models

- `microsoft/radedit`: gated, research use only. Do not redistribute the weights.
- `google/medgemma-4b-it`: gated (accept the terms on Hugging Face).
- TorchXRayVision classifiers as independent verifiers of the edits.

On the cluster, log in once inside the venv so the gated downloads work: `hf auth login`
(older versions: `huggingface-cli login`), with a read token from your Hugging Face settings.
