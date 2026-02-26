# LeLM

Fine-tuned LLM for NBA hot takes. Built on Qwen3-8B with LoRA, trained on Reddit r/nba posts and synthetic data.

## Overview

LeLM is a Qwen3-8B model fine-tuned to generate and analyze NBA hot takes with conviction, stats backing, and personality. The full pipeline covers data collection, processing, training, and inference.

**Model**: [KenWu/LeLM](https://huggingface.co/KenWu/LeLM) (LoRA adapter) | [KenWu/LeLM-GGUF](https://huggingface.co/KenWu/LeLM-GGUF) (quantized, ready for inference)

## Example Output

```
>> Is KD the best scorer in NBA history?

KD's midrange efficiency is a game-changer. He's got the perfect blend of
technique and touch, with a 45% midrange clip that puts him in elite company.
You can't teach that level of consistency, it's all about feel and execution.
KD delivers.
```

## Project Structure

```
LeLM/
├── configs/
│   └── train_config.yaml        # Model, LoRA, and training hyperparameters
├── data/
│   ├── raw/                     # Scraped Reddit posts and comments
│   ├── processed/               # Train/val splits (2,434 / 129 examples)
│   └── prompts.txt              # System prompt for the model persona
├── scripts/
│   ├── 01_scrape_reddit.py      # Scrape r/nba hot takes via public JSON API
│   ├── 01b_scrape_kd.py         # KD-targeted Reddit scrape
│   ├── 01c_generate_kd_synthetic.py  # Generate synthetic KD training data
│   ├── 02_process_data.py       # Clean, filter, dedup, format, and split
│   ├── 03_train.py              # QLoRA fine-tuning with Unsloth + TRL
│   └── 04_inference.py          # Load adapter and run inference / REPL
├── notebooks/
│   ├── LeLM_colab.ipynb         # End-to-end training on Google Colab (T4)
│   └── convert_to_gguf.ipynb    # Merge LoRA + convert to GGUF on Colab
└── pyproject.toml
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen3-8B (4-bit via Unsloth) |
| Method | QLoRA (r=64, alpha=128) |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Training data | 2,434 examples (Reddit + synthetic) |
| Epochs | 3 (915 steps) |
| Batch size | 8 (2 per device x 4 accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Final train loss | 0.288 |
| Eval loss | 0.755 (epoch 2, best) |

### Data Pipeline

1. **Scrape** — Collect hot takes, unpopular opinions, and debates from r/nba using Reddit's public JSON endpoints (no API key required). Includes checkpointing and resume support.
2. **Process** — Clean Reddit artifacts, filter by score/length, deduplicate with trigram Jaccard similarity, format into chat conversations with randomized instruction templates, and split 95/5.
3. **Train** — QLoRA fine-tuning with Unsloth for 2x memory efficiency. Runs on a free Colab T4 GPU in ~45 minutes.

## Quick Start

### Run on Google Colab (recommended)

1. Upload `data/processed/train.jsonl` and `val.jsonl` to Google Drive under `MyDrive/LeLM/`
2. Open `notebooks/LeLM_colab.ipynb` in Colab
3. Select GPU runtime (T4 free tier works)
4. Run all cells

### Run locally (requires GPU)

```bash
# Install dependencies
uv sync

# Scrape data (optional, processed data is included)
uv run python scripts/01_scrape_reddit.py

# Process and split
uv run python scripts/02_process_data.py

# Train (requires CUDA GPU)
uv run python scripts/03_train.py

# Inference
uv run python scripts/04_inference.py
```

### Use the GGUF with Ollama

```bash
# Download the quantized model
huggingface-cli download KenWu/LeLM-GGUF LeLM-Q4_K_M.gguf --local-dir .

# Create Ollama model
cat > Modelfile << 'EOF'
FROM ./LeLM-Q4_K_M.gguf
PARAMETER temperature 0.7
SYSTEM You are an unapologetically bold NBA analyst who lives for hot takes. You speak with absolute conviction, back up your claims with stats and game knowledge, but aren't afraid to be controversial.
EOF

ollama create lelm -f Modelfile
ollama run lelm "Is LeBron washed?"
```

## Part of LeGM-Lab

LeLM powers [LeGM-Lab](https://github.com/KenWuqianghao/LeGM-Lab), an LLM-driven NBA take analysis bot that fact-checks hot takes on Twitter with real stats.
