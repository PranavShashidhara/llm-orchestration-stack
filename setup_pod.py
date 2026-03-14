"""
setup_pod.py — Pull model from Google Drive onto the RunPod pod

Run this ONCE after SSH-ing into your pod, before starting serve_vllm.py.

Usage:
    python setup_pod.py
    python setup_pod.py --gdrive-path "models/Infra" --local-path /workspace/model

Requirements:
    pip install gdown huggingface_hub
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
# Mirror of your Google Drive path: /content/gdrive/MyDrive/models/Infra
DEFAULT_GDRIVE_FOLDER = "models/Infra"
DEFAULT_LOCAL_PATH    = "/workspace/model"


# ── Install dependencies if missing ──────────────────────────────────────────
def ensure_deps():
    pkgs = ["gdown", "huggingface_hub"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)


# ── Option A: Download via gdown (Google Drive direct) ───────────────────────
def download_from_gdrive_folder(gdrive_folder_id: str, local_path: str):
    """
    Downloads a Google Drive folder by its folder ID.

    To get your folder ID:
      1. Open the folder in Google Drive
      2. Copy the ID from the URL:
         https://drive.google.com/drive/folders/<FOLDER_ID>
    """
    import gdown

    os.makedirs(local_path, exist_ok=True)
    print(f"📥 Downloading from Google Drive folder ID: {gdrive_folder_id}")
    print(f"   Destination: {local_path}")
    print("   This may take a while for large models...\n")

    url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"
    gdown.download_folder(url, output=local_path, quiet=False, use_cookies=False)

    print(f"\n✅ Download complete → {local_path}")


# ── Option B: Mount via rclone (most reliable for large files) ────────────────
def setup_rclone(local_path: str):
    """
    Uses rclone to copy from Google Drive. More reliable than gdown for
    large model files (>2GB). Requires one-time rclone OAuth setup.
    """
    print("Setting up rclone for Google Drive access...")

    # Install rclone
    subprocess.run(
        "curl https://rclone.org/install.sh | bash",
        shell=True, check=True
    )

    print("\n" + "=" * 60)
    print("rclone config — follow the prompts:")
    print("  1. Type 'n' for new remote")
    print("  2. Name it 'gdrive'")
    print("  3. Choose Google Drive (option 13 or search 'drive')")
    print("  4. Leave client_id and secret blank")
    print("  5. Choose scope 1 (full access)")
    print("  6. Use auto config: NO (you're on a headless server)")
    print("  7. Paste the auth URL into your browser, then paste the token back")
    print("=" * 60 + "\n")

    subprocess.run(["rclone", "config"], check=True)

    # Copy model files
    gdrive_src = f"gdrive:models/Infra"
    os.makedirs(local_path, exist_ok=True)

    print(f"\n📥 Copying {gdrive_src} → {local_path}")
    subprocess.run(
        ["rclone", "copy", gdrive_src, local_path,
         "--progress", "--transfers", "4", "--checkers", "8"],
        check=True
    )
    print(f"✅ Model copied to {local_path}")


# ── Verify model files ────────────────────────────────────────────────────────
def verify_model(local_path: str) -> bool:
    """
    Checks that the model directory contains expected files.
    Supports both HF format and GGUF format.
    """
    path = Path(local_path)
    if not path.exists():
        print(f"❌ Path does not exist: {local_path}")
        return False

    files = list(path.rglob("*"))
    if not files:
        print(f"❌ Directory is empty: {local_path}")
        return False

    # Check for HF format
    has_config     = any(f.name == "config.json"           for f in files)
    has_weights    = any(f.suffix in (".bin", ".safetensors") for f in files)
    has_gguf       = any(f.suffix == ".gguf"               for f in files)
    has_adapter    = any(f.name == "adapter_config.json"   for f in files)

    total_size_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9

    print(f"\n── Model Directory Contents ─────────────────────────────────")
    print(f"  Path         : {local_path}")
    print(f"  Total size   : {total_size_gb:.2f} GB")
    print(f"  Files found  : {len([f for f in files if f.is_file()])}")
    print(f"  HF config    : {'✅' if has_config  else '❌'}")
    print(f"  HF weights   : {'✅' if has_weights else '❌'}")
    print(f"  GGUF file    : {'✅' if has_gguf    else '❌ (not required for vLLM)'}")
    print(f"  LoRA adapter : {'✅' if has_adapter else '❌ (not required if merged)'}")

    if has_gguf and not has_config:
        print("\n⚠️  GGUF detected but no HF config found.")
        print("   vLLM requires HF format (config.json + safetensors).")
        print("   You need to merge your LoRA adapter into the base model first.")
        print("   See: merge_adapter.py")
        return False

    if not has_config:
        print("\n❌ No config.json found — not a valid HF model directory.")
        return False

    print("\n✅ Model looks valid for vLLM\n")
    return True


# ── Merge LoRA adapter into base model (if needed) ───────────────────────────
def merge_lora_if_needed(adapter_path: str, base_model: str, output_path: str):
    """
    If you saved only the LoRA adapter (not merged), merge it into the
    base model before serving with vLLM.

    vLLM can serve LoRA adapters directly with --enable-lora, but merging
    is simpler and faster for single-adapter setups.
    """
    print(f"🔀 Merging LoRA adapter into base model...")
    print(f"   Base model   : {base_model}")
    print(f"   LoRA adapter : {adapter_path}")
    print(f"   Output       : {output_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        print("   Loading base model (this takes ~2 min)...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",     # Merge on CPU to avoid VRAM pressure
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        print("   Applying LoRA weights...")
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()

        print(f"   Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        print(f"✅ Merged model saved to {output_path}")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install transformers peft accelerate")
        sys.exit(1)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pull model from Google Drive onto RunPod pod")
    parser.add_argument("--gdrive-folder-id", default=None,
                        help="Google Drive folder ID (from the URL of your Drive folder)")
    parser.add_argument("--local-path",       default=DEFAULT_LOCAL_PATH,
                        help=f"Where to save the model on the pod (default: {DEFAULT_LOCAL_PATH})")
    parser.add_argument("--method",           default="rclone", choices=["rclone", "gdown"],
                        help="Download method: rclone (recommended) or gdown")
    parser.add_argument("--merge-lora",       action="store_true",
                        help="Merge LoRA adapter into base model after download")
    parser.add_argument("--base-model",       default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model ID for LoRA merging")
    parser.add_argument("--verify-only",      action="store_true",
                        help="Only verify an already-downloaded model, skip download")
    args = parser.parse_args()

    ensure_deps()

    print("=" * 60)
    print("SQL-Genie — Pod Model Setup")
    print("=" * 60)
    print(f"  Method       : {args.method}")
    print(f"  Local path   : {args.local_path}")
    print()

    if not args.verify_only:
        if args.method == "gdown":
            if not args.gdrive_folder_id:
                print("❌ --gdrive-folder-id is required for gdown method.")
                print()
                print("To find your folder ID:")
                print("  1. Open your Google Drive folder in a browser")
                print("  2. Copy the ID from the URL:")
                print("     https://drive.google.com/drive/folders/<THIS_PART>")
                sys.exit(1)
            download_from_gdrive_folder(args.gdrive_folder_id, args.local_path)
        else:
            setup_rclone(args.local_path)

    # Merge LoRA if needed
    if args.merge_lora:
        adapter_path  = args.local_path
        merged_path   = args.local_path + "_merged"
        merge_lora_if_needed(adapter_path, args.base_model, merged_path)
        args.local_path = merged_path

    # Verify
    valid = verify_model(args.local_path)

    if valid:
        print("🚀 Ready to serve! Run:")
        print(f"   python serve_vllm.py --model {args.local_path}")
    else:
        print("\n⚠️  Fix the issues above before running serve_vllm.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
