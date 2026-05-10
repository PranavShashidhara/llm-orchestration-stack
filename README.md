# LLM Orchestration Platform (LOP)

An end-to-end LLM fine-tuning and inference platform for SQL generation built around **Llama-3.1 8B-Instruct**, using QLoRA fine-tuning, 4-bit quantization, and scalable multi-GPU inference with vLLM.

This project demonstrates production-grade infrastructure for fine-tuning and deploying large language models at scale, with focus on memory efficiency, throughput optimization, and parallel inference across multiple GPUs.

## Project Structure

```
infra/
├── main_fine_tuning_pipeline.ipynb        # QLoRA fine-tuning pipeline (Colab A100/H100)
│                                          # Loads Gretel dataset, trains adapter, saves to GDrive
│
├── model_load_to_HF.ipynb                 # Model export workflow
│                                          # Loads fine-tuned model from GDrive, pushes to HuggingFace Hub
│                                          # Includes test inference from HF hub
│
├── Inference_Quantization_SQL.ipynb       # Quantization and inference testing
│                                          # Quantizes model to 4-bit GGUF format
│                                          # Demonstrates inference from HF hub
│
├── tokenizer.py                           # Utility for prompt formatting
│                                          # Handles ChatML template application
│
├── requirements.txt                       # Python dependencies for local/RunPod environment
├── setup.sh                               # Local environment bootstrap (optional, not needed for Colab)
│
├── checkpoints/                           # Training checkpoints and LoRA adapters (GDrive)
├── data/                                  # Training dataset
│   ├── unified/master_dataset             # Preprocessed ChatML data
│   └── tokenized_master                   # Tokenized for training
│
└── venv/                                  # Python virtual environment (local only)
```

**Workflow:**

1. **Fine-tune** on Colab with `main_fine_tuning_pipeline.ipynb` → saved to GDrive
2. **Export GGUF** with `model_load_to_HF.ipynb` → first artifact to HuggingFace Hub
3. **Merge & Quantize** with `Inference_Quantization_SQL.ipynb` → create NVFP4 Safetensors
4. **Deploy** on RunPod (A5090 Blackwell) using the NVFP4 quantized model from HF hub


## Quick Start

### Training Phase (Google Colab)

1. Open `main_fine_tuning_pipeline.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → GPU → A100 or H100**
3. Add `HF_TOKEN` secret (left sidebar) for model push
4. Run all cells top to bottom
5. Fine-tuned adapter is saved to your GDrive

**What happens:**
- Loads Gretel SQL dataset (~100k samples)
- Converts to ChatML format for Llama-3
- Fine-tunes with QLoRA (4-bit) + Liger Kernels
- Evaluates on holdout SQL samples
- Saves checkpoint to `checkpoints/` in GDrive

**Training time:**
| GPU | VRAM | Runtime |
|-----|------|---------|
| A100 40GB | 39GB | ~4 hours |
| H100 95GB | 78GB | ~1.5 hours |

### Export Phase

1. Open `model_load_to_HF.ipynb` in Colab
2. Loads fine-tuned adapter from GDrive
3. Exports to GGUF format
4. Tests inference on sample queries
5. Pushes GGUF model to HuggingFace Hub

### Quantization & Testing

1. Open `Inference_Quantization_SQL.ipynb` in Colab
2. Loads GGUF model from HF hub
3. Merges adapter weights with base model
4. Applies NVFP4 quantization (4-bit)
5. Tests inference performance
6. Exports quantized model as Safetensors
7. Pushes final quantized artifact to HuggingFace Hub

### Inference Deployment (RunPod)

1. Launch RunPod GPU pod
2. SSH in and install vLLM: `pip install vllm`
3. Start vLLM server (see [Deploying on RunPod](#deploying-on-runpod))
4. Run Python inference client with your RunPod URL
5. Get SQL generation results via API

#### Single GPU Deployment (A5000 24GB VRAM)

**Hardware Limitation:** A5000 uses Ada architecture, which does not support NVFP4 quantization. Deploy with full-precision fp16 model instead:

```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model_fp16 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32000 \
  --dtype float16 \
  --enable-prefix-caching \
  --served-model-name sql-genie \
  --port 8000 \
  > /workspace/logs/vllm_gpu0.log 2>&1 &
```

**Configuration for A5000 (fp16):**
- Full precision model (~19GB) required due to NVFP4 incompatibility
- `--max-model-len 32000` (reduced from 128000 to fit KV-cache in 1-2GB headroom)
- `--gpu-memory-utilization 0.95` to use available 24GB
- Limited to single concurrent request
- Expected throughput: 18-22 tokens/sec

#### Dual GPU Deployment (A5000 24GB VRAM × 2)

Deploy two parallel vLLM instances on separate A5000 GPUs, each running the fp16 (non-quantized) model:

```bash
# GPU 0: Instance 1 on port 8000
CUDA_VISIBLE_DEVICES=0 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model_fp16 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32000 \
  --dtype float16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --served-model-name sql-genie-gpu0 \
  --port 8000 \
  > /workspace/logs/vllm_gpu0.log 2>&1 &

# GPU 1: Instance 2 on port 8001
CUDA_VISIBLE_DEVICES=1 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model_fp16 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32000 \
  --dtype float16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --served-model-name sql-genie-gpu1 \
  --port 8001 \
  > /workspace/logs/vllm_gpu1.log 2>&1 &
```

**Key configuration for A5000 dual deployment (Ada, fp16):**
- `--max-model-len 32000` (tight memory constraints)
- `--gpu-memory-utilization 0.95` maximizes available 24GB per GPU
- `--kv-cache-dtype fp8` reduces KV-cache memory by 2×
- Two independent full model instances (Ada lacks quantization support)
- Each instance serves requests independently from port 8000 and 8001
- Load balancer distributes requests round-robin across both instances
- Combined throughput: ~36-44 tokens/sec (2× single GPU)
- Model is fp16 (no Blackwell quantization available)

### Deploying on RunPod

RunPod pods do not come with vLLM pre-installed. Set up the inference environment as follows:

```bash
# SSH into your RunPod instance
ssh root@<pod-ip>

# Install vLLM (required for inference)
pip install vllm

# Start vLLM server with your quantized model
CUDA_VISIBLE_DEVICES=0 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 128000 \
  --dtype float16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --served-model-name sql-genie \
  --port 8000 \
  > /workspace/logs/vllm_server.log 2>&1 &

# Check if server is running
tail -f /workspace/logs/vllm_server.log
```

The vLLM server will be accessible at your RunPod URL (e.g., `http://<runpod-url>:8000/v1/completions`).

#### Dual GPU Deployment (A5090 32GB VRAM × 2 - Blackwell)

Deploy two parallel vLLM instances on separate A5090 GPUs (Blackwell architecture), each running the NVFP4 quantized model independently:

```bash
# GPU 0: Instance 1 on port 8000
CUDA_VISIBLE_DEVICES=0 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 128000 \
  --dtype float16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --served-model-name sql-genie-gpu0 \
  --port 8000 \
  > /workspace/logs/vllm_gpu0.log 2>&1 &

# GPU 1: Instance 2 on port 8001
CUDA_VISIBLE_DEVICES=1 nohup vllm serve \
  pshashid/llama3.1B_8B_SQL_Finetuned_model \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 128000 \
  --dtype float16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --served-model-name sql-genie-gpu1 \
  --port 8001 \
  > /workspace/logs/vllm_gpu1.log 2>&1 &
```

**Key configuration for A5090 parallel deployment (Blackwell NVFP4):**
- `--max-model-len 128000` (4× longer than A5000's 32000)
- `--gpu-memory-utilization 0.90` leverages 28-29GB per GPU
- `--kv-cache-dtype fp8` saves additional headroom for batch processing
- Two independent full model instances maximize request concurrency
- Each instance serves requests independently from port 8000 and 8001
- Load balancer distributes requests round-robin across both instances
- Model is NVFP4 quantized (Blackwell hardware acceleration required)


## Architecture

### Training Pipeline

```
Dataset (Gretel SQL 100k)
    ↓
ChatML Preprocessing
    ↓
QLoRA Fine-tuning (Unsloth + Liger)
    ↓
GGUF Export + HuggingFace Push (from GDrive)
    ↓
Adapter Merge
    ↓
4-bit Quantization (NVFP4)
    ↓
Safetensors + HuggingFace Push
```

**Workflow breakdown:**

1. **Fine-tuning** - QLoRA adapters trained on Gretel SQL dataset
2. **Checkpoint Save** - Saved to GDrive for persistence
3. **GGUF Export** - First export format pushed to HuggingFace Hub
4. **Adapter Merge** - Merge LoRA weights with base model
5. **Quantization** - Apply NVFP4 (Blackwell-optimized) quantization
6. **Final Push** - Quantized model uploaded as Safetensors format to HF Hub

The quantized Safetensors artifact is what's deployed on Blackwell GPUs (A5090) via vLLM.

### Inference Architecture

```
Client Request
    ↓
Load Balancer / Router
    ↓
    ├─→ GPU0: vLLM Server (Port 8000) - Full Model Instance
    ├─→ GPU1: vLLM Server (Port 8001) - Full Model Instance
    └─→ GPU2: vLLM Server (Port 8002) - Full Model Instance [optional]
    ↓
SQL Generation Response (OpenAI-compatible API)
```

Each GPU runs a complete, independent model instance. Requests are load-balanced across instances for parallel processing.

**Parallel vs. Distributed Inference:**

This architecture uses **parallel inference** with multiple full model instances rather than distributed inference with model sharding. Each GPU has a complete copy of the model and operates independently. This approach is optimal for models that fit on individual GPUs, providing simpler deployment, better fault isolation, and easier horizontal scaling compared to tensor-parallelism-based distributed inference.


## Training Configuration

### Actual Hyperparameters Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `meta-llama/Llama-3.1-8B-Instruct` | Base model from Meta |
| **LoRA Rank (R)** | 128 | Optimized for 8B model, balances capacity vs VRAM |
| **LoRA Alpha** | 128 | Scaling factor (R = Alpha for 1.0 scaling) |
| **LoRA Dropout** | 0.0 | Disabled (large batch size provides regularization) |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + feedforward projections |
| **Max Seq Length** | 1024 | Reduced from 2048 (quadratic attention cost) |
| **Batch Size** | 128 | Linear VRAM cost = fast training |
| **Gradient Accumulation** | 2 | Effective batch size: 256 |
| **Learning Rate** | 8e-4 | Scaled with batch size |
| **Warmup Ratio** | 0.05 | 5% linear warmup |
| **Num Epochs** | 1 | Single pass over dataset |
| **Save Interval** | Every 200 steps | Checkpoint frequency |

**Effective batch size:** `128 × 2 = 256` per gradient update

**Why MAX_SEQ_LENGTH=1024?**

The SQL dataset's longest sequences are ~500 tokens. Using 1024 provides comfortable headroom without the quadratic overhead of 2048. Initial estimate with 2048 was 159+ hours; reducing to 1024 cut training time dramatically while maintaining full dataset coverage.

### Dataset Configuration

| Dataset | Samples | Purpose |
|---------|---------|---------|
| Gretel SQL | 100,000 | SQL generation pairs |
| Finetome | 100,000 | General instruction following |
| UltraChat | 200,000 | Conversational diversity |
| **Total** | **400,000** | Mixed training corpus |

All datasets converted to Llama-3.1 chat template format before training.

### Output Artifacts

| Artifact | Location | Size | Format | Purpose |
|----------|----------|------|--------|---------|
| **LoRA Adapter** | `sql-genie-lora/` | ~33 MB | PyTorch | Fine-tuned weights only |
| **GGUF Model** | HuggingFace Hub | ~9-10 GB | GGUF | Intermediate checkpoint |
| **Merged + Quantized** | HuggingFace Hub | ~4-5 GB | Safetensors (NVFP4) | Production deployment |
| **Checkpoints** | GDrive `/models/Infra/` | ~33 MB each | LoRA | Training saves (every 200 steps) |

**Deployment Artifact:** Use the Safetensors NVFP4 model from HuggingFace Hub. Requires Blackwell architecture (A5090) for hardware-accelerated inference.

## Quantization Strategy

The model uses a two-stage quantization approach optimized for Blackwell architecture inference:

### Weights: NVFP4 (4-bit)

NVFP4 is NVIDIA's proprietary 4-bit floating-point format native to Blackwell GPUs. It provides hardware-accelerated inference with minimal accuracy loss compared to fp16.

**Compression achieved:** ~16GB (fp16) → ~4-5GB NVFP4 (4× reduction)

### KV-Cache: FP8 (8-bit)

During token generation, attention key-value caches are stored in FP8 format with tensor-wise dynamic quantization. This further reduces memory consumption during inference without affecting output quality.

**Configuration:**
- Format: FP8 (8-bit floating point)
- Strategy: Tensor-wise (per-tensor quantization)
- Dynamic: Disabled (static quantization)
- Symmetric: True (symmetric range)

### Calibration

Quantization uses a calibration dataset of 128 representative SQL generation samples to determine optimal quantization parameters. Samples cover diverse SQL patterns (aggregations, joins, filtering, grouping) to ensure the quantized model generalizes well across different schemas.

### Result

Combined NVFP4 weights + FP8 KV-cache achieves:
- Model weights: 4× smaller (16GB → 4-5GB)
- Peak inference VRAM: 27-28GB → 14-16GB on A5090
- No degradation in SQL generation quality
- Hardware-accelerated inference on Blackwell GPUs

## Inference Benchmarks

### Single GPU Performance (A5000 24GB VRAM)

**Hardware Limitation:** A5000 uses Ada architecture which does not support NVFP4 quantization format. Requires full-precision fp16 model.

**Model Instance:** 1× Llama-3.1 8B-Instruct (fp16, unquantized)

| Metric | Value | Notes |
|--------|-------|-------|
| **Base Load VRAM** | 23-24 GB | Full precision weights |
| **Peak VRAM** (256 tokens) | 23.8 GB | Limited headroom for KV-cache |
| **Throughput** (batch=1) | 18-22 tokens/sec | Slower than A5090 due to precision |
| **Latency (128 tokens)** | 5.8-6.4 sec | Higher latency vs quantized |
| **Latency (256 tokens)** | 11.6-12.8 sec | Memory bandwidth limited |
| **Max Batch Size** | 1 | No room for batching |
| **Max Seq Length** | 32000 | Limited by tight memory margins |

**Note:** For A5000 deployment, use unquantized models or alternative quantization formats (8-bit) instead of NVFP4.

### Dual GPU Performance (A5090 32GB VRAM × 2)

**Hardware Advantage:** A5090 uses Blackwell architecture with full support for NVFP4 quantization, enabling 3-4× better performance than A5000.

**Model Configuration:** 2× parallel instances of NVFP4 quantized models (full capacity)

| Metric | A5000 Single | A5000 Dual (fp16) | A5090 Dual (NVFP4) | A5090 Improvement |
|--------|-----------|----------|-------------|---|
| **Quantization** | Full precision | Full precision | NVFP4 (Blackwell) | Hardware-accelerated |
| **Base Load VRAM/GPU** | 23-24 GB | 23-24 GB | 4-5 GB | **4.8-5.5× smaller** |
| **Available for KV-cache** | <1 GB | <1 GB | 27-28 GB | **27× more headroom** |
| **Throughput** | 18-22 tok/sec | 36-44 tok/sec | 68-75 tok/sec | **1.9× vs A5000 dual** |
| **Latency P99** | 6.4 sec | 3.2 sec | 1.7 sec | **1.9× vs A5000 dual** |
| **Max Concurrent Requests** | 1 | 2-4 | 20-24 | **5-10× vs A5000 dual** |
| **Max Seq Length** | 32000 | 32000 | 128000 | **4× longer contexts** |

*A5090 NVFP4 compression (4-5GB) vs A5000 fp16 (23-24GB) enables dramatically better scaling and concurrency

### Memory Breakdown

### Memory Breakdown

**Model Load Phase (A5000 24GB - Full Precision):**
```
Base Model Weights (fp16):   ~19 GB
Attention Buffers:           ~2.5 GB
CUDA Runtime Overhead:       ~0.5-1 GB
─────────────────────────────────
Total:                       ~22-24 GB (max capacity)
Headroom for KV-cache:       <1 GB
```

**Model Load Phase (A5090 32GB - NVFP4 Quantized):**

NVFP4 weights combined with FP8 KV-cache quantization:

```
Base Model Weights (NVFP4):  ~4-5 GB (4× compression vs fp16)
Attention Buffers:           ~1-1.5 GB
CUDA Runtime Overhead:       ~0.5-1 GB
─────────────────────────────────
Total Model:                 ~6-7 GB
Available for KV-cache (FP8): ~25-26 GB
```

**Generation Phase (KV-Cache Growth - A5090 NVFP4):**
```
Model Load:                  ~6-7 GB
Prompt Processing (512 tok): ~10-12 GB (FP8 KV-cache)
Token-by-token Generation:   ~12-24 GB (grows linearly, FP8)

Formula (with FP8): VRAM_gen ≈ base_load + (seq_len × layers × hidden_size / 8)

A5090 Blackwell advantage: 25GB+ available for KV-cache enables:
                           - Very long sequences (128k tokens)
                           - Higher batch sizes (4-6 concurrent requests)
                           - No memory spillover
```

**Architecture Requirement:**

NVFP4 weights quantization **only works on Blackwell architecture** (A5090). Ada architecture (A5000) cannot load NVFP4 models. This is a hardware-level limitation due to Blackwell's native FP4 support.

## Inference Deployment Details

### Prompt Formatting

All prompts must be formatted using the Llama-3.1 chat template before sending to vLLM. This is done on the **client side**, not the server.

**Why:** vLLM receives pre-formatted strings and does not apply tokenization templates itself. The template is automatically pulled from the model's `tokenizer_config.json` when loading.

**Llama-3.1 Chat Template:**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

This template is automatically applied when you load the tokenizer from HuggingFace and call `apply_chat_template()` with your messages. The `add_generation_prompt=True` parameter appends the assistant header to signal the model to start generating.

The formatted prompt string is then sent to vLLM's API endpoint.

### vLLM Configuration Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--tensor-parallel-size 1` | 1 | Single GPU inference (no model sharding) |
| `--kv-cache-dtype fp8` | fp8 | Reduces KV-cache memory by 2× |
| `--max-model-len` | 128000 (A5090) / 32000 (A5000) | Supports long context windows |
| `--dtype float16` | float16 | Fast mixed-precision compute |
| `--enable-prefix-caching` | enabled | Reuses KV-cache for repeated prompts |
| `--served-model-name` | custom | OpenAI-compatible model identifier |

**Important:** Prompts sent to vLLM must be pre-tokenized using the Llama-3 tokenizer's `apply_chat_template()` method on the client side. vLLM expects already-formatted strings, not raw text.

### API Usage

The vLLM server exposes an OpenAI-compatible API. The inference workflow is:

1. **Load tokenizer** from HuggingFace Hub
2. **Build messages** with system prompt (schema) and user query
3. **Apply template** using `apply_chat_template(messages, add_generation_prompt=True)`
4. **POST** formatted prompt to vLLM endpoint: `http://<pod-url>:8000/v1/completions`
5. **Receive** generated SQL in response

**Request format:**

```json
{
  "model": "sql-genie",
  "prompt": "[ChatML formatted prompt from tokenizer]",
  "max_tokens": 200,
  "temperature": 0.0
}
```

**Response format:**

```json
{
  "id": "cmpl-12345",
  "object": "text_completion",
  "created": 1710510234,
  "model": "sql-genie",
  "choices": [
    {
      "text": "SELECT customer_id, SUM(amount) FROM orders WHERE created_at >= NOW() - INTERVAL 30 DAY GROUP BY customer_id;",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 32,
    "total_tokens": 77
  }
}
```

**Key requirements:**

- Tokenize prompts locally using the Llama-3.1 tokenizer's `apply_chat_template()` method
- Follow Llama-3.1 ChatML format (system + user roles, generation prompt appended)
- Send formatted string (not raw text) to vLLM API
- Use temperature 0.0 for deterministic SQL generation

## Dataset

**Primary Dataset:** Gretel SQL

| Dataset | Samples | Format | Use Case |
|---------|---------|--------|----------|
| `gretelai/synthetic_text_to_sql` | ~100k | ChatML | SQL generation |

**Data Pipeline:**
```
Raw Dataset (JSON)
    ↓
ChatML Conversion
    ↓
Tokenization (Llama-3 template)
    ↓
Arrow/Parquet (vectorized training)
    ↓
Streaming to GPU during training
```

## Model Outputs

| Artifact | Location | Size | Use Case |
|----------|----------|------|----------|
| **LoRA Adapter** | `sql-genie-lora/` | ~33 MB | Fine-tuned weights |
| **Merged Model** | HuggingFace Hub | ~16 GB | Full model (unquantized) |
| **Quantized Model (GGUF)** | `sql-genie-gguf/*.gguf` | ~4-8 GB | Q4_K_M deployment artifact |
| **Checkpoints** | `checkpoints/` | ~33 MB each | Mid-training saves (every 200 steps) |

## Local Setup (Optional)

For local development without Colab:

```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
pip install -r requirements.txt

# Run fine-tuning locally (requires NVIDIA GPU with 24GB+ VRAM)
jupyter notebook main_fine_tuning_pipeline.ipynb
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model** | Llama-3.1 8B-Instruct | Base LLM for SQL generation |
| **Fine-tuning** | Unsloth + Liger Kernels | 2-10× training speedup |
| **Quantization** | NVFP4 (Blackwell) | Hardware-accelerated 4-bit format |
| **Inference** | vLLM | High-throughput batch processing |
| **Parallel Inference** | Multi-GPU instances | Independent servers per GPU |
| **Deployment** | OpenAI-compatible API | Standardized client interface |
| **Model Hub** | HuggingFace | Model versioning & distribution |

**NVFP4 Note:** Quantization format requires Blackwell architecture for hardware-accelerated inference (A5090, newer H100 variants). Ada architecture (A5000, A6000) requires fp16 deployment.

## Performance Tuning

### For Throughput Optimization

1. Increase batch size if VRAM allows
2. Enable prefix caching for repeated prompts
3. Use fp8 KV-cache instead of fp16
4. Add more GPU instances for parallel request handling

### For Latency Reduction

1. Reduce `max-model-len` if extended context windows are not required
2. Use smaller batch size (1-2) for single concurrent requests
3. Pre-warm the model with dummy requests to stabilize execution
4. Enable prefix caching for commonly repeated prompts

## Troubleshooting

### Out of Memory During Training

**Issue:** CUDA OOM error during fine-tuning

**Resolution:** Reduce `BATCH_SIZE` from 8 to 4, reduce `MAX_SEQ_LENGTH` from 2048 to 1024, or reduce `LORA_R` from 64 to 32. Gradient checkpointing is enabled by default in Unsloth and provides additional memory savings.

### Out of Memory During Inference

**Issue:** vLLM fails to load model on target GPU

**Resolution:** Verify `--kv-cache-dtype` is set to `fp8`. Consider reducing `--max-model-len` from 64000 to 32000 if extended context is not required. Lower batch size in inference client to reduce peak memory consumption. Switch to quantized GGUF artifact if available.

### Slow Inference Performance

**Issue:** Inference throughput below 20 tokens/second

**Resolution:** Check GPU utilization with `nvidia-smi` to determine if memory bandwidth is the bottleneck. Verify vLLM process is bound to correct GPU via logs. Increase batch size if request latency allows. Profile CPU-GPU communication for potential bottlenecks.

## References

- [Llama-3.1 Documentation](https://llama.meta.com/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [vLLM Documentation](https://docs.vllm.ai/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Liger Kernels](https://github.com/linkedin/Liger-Kernel)
