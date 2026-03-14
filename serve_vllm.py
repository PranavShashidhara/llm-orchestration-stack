"""
serve_vllm.py — Launch vLLM inference server on dual RTX 5090s

Usage (on the RunPod pod after SSH):
    python serve_vllm.py --model /workspace/model
    python serve_vllm.py --model /workspace/model --port 8000 --dry-run

Requirements (pre-installed in vllm/vllm-openai Docker image):
    vllm >= 0.4.0
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    "model"                   : "/workspace/model",       # HF format or merged LoRA
    "draft_model"             : "meta-llama/Llama-3.2-1B",# Speculative decoding draft
    "num_speculative_tokens"  : 5,                        # Tokens guessed per step
    "tensor_parallel_size"    : 2,                        # TP=2 across dual 5090s
    "max_model_len"           : 131072,                   # 128k context window
    "gpu_memory_utilization"  : 0.92,                     # Leave 8% headroom
    "kv_cache_dtype"          : "fp8",                    # FP8 KV-cache — fits 128k context in 64GB VRAM
    "port"                    : 8000,
    "host"                    : "0.0.0.0",
    "dtype"                   : "bfloat16",
}


# ── Build vLLM Command ────────────────────────────────────────────────────────
def build_vllm_command(args) -> list[str]:
    """Constructs the full vllm serve command as a list of arguments."""
    cmd = [
        "vllm", "serve", args.model,

        # ── Parallelism ──────────────────────────────────────────────────────
        "--tensor-parallel-size",       str(args.tensor_parallel_size),

        # ── Speculative Decoding ─────────────────────────────────────────────
        # Llama 3.2 1B guesses 5 tokens at a time; 8B verifies them in parallel.
        # Effective throughput: 3,000+ tokens/sec on dual 5090s.
        "--speculative-model",          args.draft_model,
        "--num-speculative-tokens",     str(args.num_speculative_tokens),

        # ── Memory & Context ─────────────────────────────────────────────────
        "--kv-cache-dtype",             args.kv_cache_dtype,  # FP8 halves KV memory vs bfloat16
        "--max-model-len",              str(args.max_model_len),
        "--gpu-memory-utilization",     str(args.gpu_memory_utilization),

        # ── Performance ──────────────────────────────────────────────────────
        "--dtype",                      args.dtype,
        "--enable-prefix-caching",      # Cache repeated SQL schema prefixes
        "--disable-log-requests",       # Reduce log noise in production

        # ── Server ───────────────────────────────────────────────────────────
        "--host",                       args.host,
        "--port",                       str(args.port),

        # ── OpenAI-compatible API ─────────────────────────────────────────────
        "--served-model-name",          "sql-genie",
    ]
    return cmd


# ── Validate Environment ──────────────────────────────────────────────────────
def validate_environment(args):
    """Checks GPU count, model path, and vLLM installation."""
    errors = []

    # Check vLLM is installed
    result = subprocess.run(["vllm", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("vLLM not found. Install with: pip install vllm")

    # Check model path exists
    if not Path(args.model).exists():
        errors.append(
            f"Model path '{args.model}' not found.\n"
            f"  Upload your merged HF model to /workspace/model on the pod,\n"
            f"  or set --model to the HuggingFace repo ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct')."
        )

    # Check GPU count
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count < args.tensor_parallel_size:
            errors.append(
                f"TP={args.tensor_parallel_size} requires {args.tensor_parallel_size} GPUs "
                f"but only {gpu_count} detected."
            )
        else:
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(gpu_count)
            )
            print(f"✅ GPUs detected : {gpu_count}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {name} ({vram:.1f} GB)")
            print(f"   Total VRAM    : {total_vram:.1f} GB")
    except ImportError:
        print("⚠️  PyTorch not available for GPU check — proceeding anyway.")

    if errors:
        print("\n❌ Validation errors:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    print("✅ Environment validated\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM inference server for SQL-Genie (dual RTX 5090, TP=2)"
    )
    parser.add_argument("--model",                  default=DEFAULTS["model"])
    parser.add_argument("--draft-model",            default=DEFAULTS["draft_model"])
    parser.add_argument("--num-speculative-tokens", default=DEFAULTS["num_speculative_tokens"], type=int)
    parser.add_argument("--tensor-parallel-size",   default=DEFAULTS["tensor_parallel_size"],   type=int)
    parser.add_argument("--max-model-len",          default=DEFAULTS["max_model_len"],           type=int)
    parser.add_argument("--gpu-memory-utilization", default=DEFAULTS["gpu_memory_utilization"],  type=float)
    parser.add_argument("--kv-cache-dtype",         default=DEFAULTS["kv_cache_dtype"])
    parser.add_argument("--dtype",                  default=DEFAULTS["dtype"])
    parser.add_argument("--port",                   default=DEFAULTS["port"],                    type=int)
    parser.add_argument("--host",                   default=DEFAULTS["host"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the vLLM command without executing it")
    args = parser.parse_args()

    print("=" * 60)
    print("SQL-Genie vLLM Inference Server")
    print("=" * 60)
    print(f"  Model              : {args.model}")
    print(f"  Draft model        : {args.draft_model}")
    print(f"  Tensor parallelism : TP={args.tensor_parallel_size}")
    print(f"  Speculative tokens : {args.num_speculative_tokens}")
    print(f"  KV-cache dtype     : {args.kv_cache_dtype}")
    print(f"  Max context        : {args.max_model_len:,} tokens ({args.max_model_len // 1024}k)")
    print(f"  GPU memory util    : {args.gpu_memory_utilization * 100:.0f}%")
    print(f"  Serving at         : http://{args.host}:{args.port}")
    print()

    validate_environment(args)

    cmd = build_vllm_command(args)

    if args.dry_run:
        print("🔍 Dry run — command that would be executed:")
        print("   " + " \\\n     ".join(cmd))
        return

    print("🚀 Starting vLLM server...")
    print("   Press Ctrl+C to stop\n")

    # Stream output directly to terminal
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ vLLM exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
