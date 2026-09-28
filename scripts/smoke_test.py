"""GPU smoke test.

Run on the cluster:  sbatch slurm/job.sbatch scripts/smoke_test.py
Prints the GPU, whether bf16 is native, and matmul speed per precision.
"""
import time

import torch


def matmul_tflops(dtype: torch.dtype, n: int = 8192, reps: int = 10) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    a @ b  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / reps
    return 2 * n**3 / seconds / 1e12


def main() -> None:
    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Did the job request --gres?")

    props = torch.cuda.get_device_properties(0)
    native_bf16 = props.major >= 8  # Ampere (A40) and newer
    print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB | "
          f"compute capability {props.major}.{props.minor} | native bf16: {native_bf16}")

    dtypes = [torch.float32, torch.float16] + ([torch.bfloat16] if native_bf16 else [])
    for dtype in dtypes:
        print(f"{str(dtype):16s} matmul: {matmul_tflops(dtype):6.1f} TFLOPS")

    if not native_bf16:
        print("Turing GPU: train with fp16 autocast + GradScaler; run MedGemma on the A40.")


if __name__ == "__main__":
    main()
