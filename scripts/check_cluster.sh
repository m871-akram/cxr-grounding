#!/usr/bin/env bash
# What resources do I have? Run it twice:
#   on nash:              bash scripts/check_cluster.sh | tee logs/check_nash.txt
#   on a compute node:    srun --gres=shard:1 --cpus-per-task=2 --mem=4GB bash scripts/check_cluster.sh | tee logs/check_node.txt
set -u

section() { printf '\n== %s\n' "$1"; }

section "Host"
echo "host=$(hostname)  user=${USER:-$(id -un)}  date=$(date '+%Y-%m-%d %H:%M')"

section "Disk quota and free space"
quota -s 2>/dev/null || echo "(no 'quota' command: rely on df below)"
df -h "$HOME" 2>/dev/null | sed 's/^/home  /'
for d in /tmp /scratch /local /matieres; do
  [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|$d  |"
done
echo "home usage (may take a few seconds):"
du -sh "$HOME" 2>/dev/null

section "Slurm partitions: name, time limit, GRES, node count, nodes"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%P %l %G %D %N"
else
  echo "(sinfo not available on this host)"
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
  echo "(no GPU visible on this host)"
fi

section "Python and PyTorch"
command -v python3 >/dev/null 2>&1 && python3 --version
python3 - <<'EOF' 2>/dev/null || echo "(torch not installed yet)"
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, f"| compute capability {p.major}.{p.minor}",
          "| native bf16:", p.major >= 8)
EOF

section "Internet access (any HTTP code other than 000 = reachable; 000 = blocked or timeout)"
for url in https://huggingface.co https://www.kaggle.com https://physionet.org \
           https://download.pytorch.org https://pypi.org https://www.googleapis.com; do
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$url" 2>/dev/null)
  printf '%-32s %s\n' "$url" "${code:-000}"
done

section "Tools"
for t in git rclone kaggle huggingface-cli tmux; do
  printf '%-16s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo 'not found')"
done
