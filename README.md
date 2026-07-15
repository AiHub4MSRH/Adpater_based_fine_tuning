# Adapter-Based Fine-Tuning for Multilingual SRH Models

This repository supports two LoRA/QLoRA fine-tuning pipelines for the same
multilingual sexual and reproductive health (SRH) dataset:

- `medigemma/` for `google/medgemma-4b-it`
- `meditron train scripts/` for `epfl-llm/meditron-7b`

Both pipelines follow the same adapter-target workflow:

1. Mirror multilingual source data from Hugging Face to local disk
2. Train one adapter per deployment target
3. Evaluate adapters on the matching split
4. Compare adapters against a baseline model
5. Push adapters to Hugging Face
6. Run inference from published adapters

The adapter targets are `aka_gha`, `amh_eth`, `eng`, `lug_uga`, and `swa`.
English combines `eng_eth`, `eng_gha`, `eng_ken`, and `eng_uga` into
`adapter_eng`; Swahili combines `swa_ken` and `swa_uga` into `adapter_swa`.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── medigemma/
│   ├── config.py
│   ├── data_utils.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluation.py
│   ├── compare_models.py
│   ├── push_adapters_to_hub.py
│   └── run_inference_from_hub.py
└── meditron train scripts/
    ├── config.py
    ├── data_utils.py
    ├── prepare_data.py
    ├── train.py
    ├── evaluation.py
    ├── compare_models.py
    ├── push_adapters_to_hub.py
    └── run_inference_from_hub.py
```

## Model Families

| Pipeline | Base model | Model type | Folder |
| --- | --- | --- | --- |
| MedGemma | `google/medgemma-4b-it` | image-text-to-text | `medigemma/` |
| Meditron | `epfl-llm/meditron-7b` | text-only causal LM | `meditron train scripts/` |

Use separate adapter output directories and separate Hub repos for each model
family. Adapters trained for one base model are not interchangeable with the
other.

Note: the Meditron folder name contains spaces, so quote that path in shell
commands.

## Supported Dataset Selections

Grouped CLI selections resolve to these adapter targets:

- `aka` -> `aka_gha`
- `amh` -> `amh_eth`
- `eng` -> `eng`
- `lug` -> `lug_uga`
- `swa` -> `swa`

Legacy leaf selections such as `eng_uga`, `eng_ken`, `swa_ken`, and `swa_uga`
are still accepted, but they resolve to the combined `eng` or `swa` adapter.

Current adapter target set in this repo:

- `aka_gha`
- `amh_eth`
- `eng`
- `lug_uga`
- `swa`

Current source leaf set used by the loader:

- `aka_gha`
- `amh_eth`
- `eng_eth`
- `eng_gha`
- `eng_ken`
- `eng_uga`
- `lug_uga`
- `swa_ken`
- `swa_uga`

## Dataset Assumptions

The dataset loader expects:

- `train`, `dev`, and `test` splits
- `input` and `output` columns
- shard files such as `train-*`, `dev-*`, and `test-*`

The loader accepts both local and Hub layouts, including case variants such as:

```text
aka/aka_gha/train-*
Aka/Aka_Gha/train-*
aka_gha/train-*
```

## Environment Setup

Run these commands once from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
```

`HF_TOKEN` is required for private dataset/model repos and may be required for
gated base models. You can also pass `--hf_token` to the scripts instead of
exporting the environment variable.

## Training Modes

Both model families now default to full-precision LoRA:

```bash
--precision full_lora
```

This loads the base model without 4-bit quantization and trains only the LoRA
adapter weights. This is the preferred setting when GPU memory allows it.

Use QLoRA only when memory is tight:

```bash
--precision qlora
```

QLoRA loads the frozen base model in 4-bit, then trains the LoRA adapter
weights. It is cheaper, but it is no longer the default.

## Output Layout

Keep MedGemma and Meditron outputs separate. Adapters are not interchangeable
between base models.

```text
data/
├── aka_gha/
├── amh_eth/
├── eng/
├── lug_uga/
└── swa/

adapters/
├── medigemma/
└── meditron/
```

Local data may also be stored as source leaves such as `data/eng_uga/` and
`data/swa_ken/`; the loader combines those leaves when training `eng` or `swa`.

## Script Map

| Task | MedGemma | Meditron |
| --- | --- | --- |
| Mirror data | `python3 medigemma/prepare_data.py` | `python3 'meditron train scripts/prepare_data.py'` |
| Train or eval | `python3 medigemma/train.py` | `python3 'meditron train scripts/train.py'` |
| Compare baseline vs adapter | `python3 medigemma/compare_models.py` | `python3 'meditron train scripts/compare_models.py'` |
| Push adapters to Hub | `python3 medigemma/push_adapters_to_hub.py` | `python3 'meditron train scripts/push_adapters_to_hub.py'` |
| Run Hub inference | `python3 medigemma/run_inference_from_hub.py` | `python3 'meditron train scripts/run_inference_from_hub.py'` |

## Step 1: Mirror Data Locally

You only need to mirror data once. Both model families can read the same
`./data` directory.

MedGemma mirror command:

```bash
python3 medigemma/prepare_data.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

Meditron mirror command:

```bash
python3 'meditron train scripts/prepare_data.py' \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

Either command produces compatible local data. Running both is harmless, but
usually unnecessary unless you want to test both loaders explicitly.

## Step 2: Train MedGemma

MedGemma uses `google/medgemma-4b-it` and the scripts under `medigemma/`. It
uses the MedGemma processor/chat template path and writes adapters under
`./adapters/medigemma`.

### 2.1 Smoke Test One Adapter

Use a small evaluation sample first to verify authentication, data loading, and
GPU memory.

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages amh \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 50
```

This trains `adapter_amh_eth` and then evaluates it.

### 2.2 Train One Production Adapter

Train English as one combined adapter:

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages eng \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 200
```

This combines `eng_eth`, `eng_gha`, `eng_ken`, and `eng_uga`, then writes
`./adapters/medigemma/adapter_eng`.

Train Swahili as one combined adapter:

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages swa \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 200
```

This combines `swa_ken` and `swa_uga`, then writes
`./adapters/medigemma/adapter_swa`.

### 2.3 Train All MedGemma Adapters

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 200
```

This trains five MedGemma adapters:

- `adapter_aka_gha`
- `adapter_amh_eth`
- `adapter_eng`
- `adapter_lug_uga`
- `adapter_swa`

### 2.4 Train MedGemma Directly From Hugging Face

Use this when you do not want a local `./data` mirror.

```bash
python3 medigemma/train.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 200
```

## Step 3: Train Meditron

Meditron uses `epfl-llm/meditron-7b` and the scripts under
`meditron train scripts/`. It is text-only/LLaMA-style, so always quote the
folder path because it contains spaces. It writes adapters under
`./adapters/meditron`.

### 3.1 Smoke Test One Adapter

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages amh \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 50
```

This trains `adapter_amh_eth` and then evaluates it.

### 3.2 Train One Production Adapter

Train English as one combined adapter:

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages eng \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 200
```

This writes `./adapters/meditron/adapter_eng`.

Train Swahili as one combined adapter:

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages swa \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 200
```

This writes `./adapters/meditron/adapter_swa`.

### 3.3 Train All Meditron Adapters

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 200
```

This trains five Meditron adapters:

- `adapter_aka_gha`
- `adapter_amh_eth`
- `adapter_eng`
- `adapter_lug_uga`
- `adapter_swa`

### 3.4 Train Meditron Directly From Hugging Face

```bash
python3 'meditron train scripts/train.py' \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 200
```

## Step 4: Evaluate Existing Adapters

Use `--eval_only` when adapters already exist and you only want a fresh
evaluation report. Reports are written to `<output_root>/eval_report.json`.

Evaluate MedGemma adapters:

```bash
python3 medigemma/train.py \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 500
```

Evaluate Meditron adapters:

```bash
python3 'meditron train scripts/train.py' \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 500
```

For full test-set evaluation, increase `--max_eval_samples`:

```bash
python3 medigemma/train.py \
  --eval_only \
  --data_root ./data \
  --languages aka_gha amh_eth eng lug_uga swa \
  --output_root ./adapters/medigemma \
  --precision full_lora \
  --max_eval_samples 100000
```

```bash
python3 'meditron train scripts/train.py' \
  --eval_only \
  --data_root ./data \
  --languages aka_gha amh_eth eng lug_uga swa \
  --output_root ./adapters/meditron \
  --precision full_lora \
  --max_eval_samples 100000
```

## Step 5: Compare Baseline vs Adapter

Comparison scripts generate a row-level CSV plus JSON reports. Use `--load_in_4bit`
only if comparison does not fit in GPU memory.

Compare MedGemma:

```bash
python3 medigemma/compare_models.py \
  --data_root ./data \
  --languages eng swa \
  --output_root ./adapters/medigemma \
  --baseline_model google/medgemma-4b-it \
  --max_eval_samples 100
```

Compare Meditron:

```bash
python3 'meditron train scripts/compare_models.py' \
  --data_root ./data \
  --languages eng swa \
  --output_root ./adapters/meditron \
  --baseline_model epfl-llm/meditron-7b \
  --max_eval_samples 100
```

Each comparison writes:

- `adapter_baseline_comparison.csv`
- `baseline_eval_report.json`
- `adapter_comparison_report.json`

## Step 6: Push Adapters to Hugging Face

Use one model repo per base model family. Do not mix MedGemma and Meditron
adapters in the same repo.

Push MedGemma adapters:

```bash
python3 medigemma/push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-medgemma-adapters \
  --output_root ./adapters/medigemma \
  --languages aka_gha amh_eth eng lug_uga swa \
  --private
```

Push Meditron adapters:

```bash
python3 'meditron train scripts/push_adapters_to_hub.py' \
  --repo_id your-org/hashie-srh-meditron-adapters \
  --output_root ./adapters/meditron \
  --languages aka_gha amh_eth eng lug_uga swa \
  --private
```

Published repos use this layout:

```text
README.md
adapters/
├── manifest.json
├── aka_gha/
├── amh_eth/
├── eng/
├── lug_uga/
└── swa/
reports/
└── eval_report.json
```

## Step 7: Run Inference From Published Adapters

Run MedGemma adapter inference:

```bash
python3 medigemma/run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-medgemma-adapters \
  --adapter_name eng \
  --prompt "What are common symptoms of an STI?"
```

Run Meditron adapter inference:

```bash
python3 'meditron train scripts/run_inference_from_hub.py' \
  --adapter_repo your-org/hashie-srh-meditron-adapters \
  --adapter_name eng \
  --prompt "What are common symptoms of an STI?"
```

Run Swahili inference with low-memory 4-bit loading:

```bash
python3 medigemma/run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-medgemma-adapters \
  --adapter_name swa \
  --prompt "Shida ya Ukimwi ni nini?" \
  --load_in_4bit
```

```bash
python3 'meditron train scripts/run_inference_from_hub.py' \
  --adapter_repo your-org/hashie-srh-meditron-adapters \
  --adapter_name swa \
  --prompt "Shida ya Ukimwi ni nini?" \
  --load_in_4bit
```

## Practical Notes

- Train one adapter target at a time if GPU memory or disk space is tight.
- Keep `./adapters/medigemma` and `./adapters/meditron` separate.
- Use `--precision full_lora` for final-quality runs.
- Use `--precision qlora` only when full-precision LoRA does not fit.
- LoRA training updates adapter weights only. The scripts no longer train or
  save `lm_head` / `embed_tokens`, which keeps low-resource adapters from
  overfitting billions of extra parameters.
- Training loss is assistant-only: system and user prompt tokens are masked out
  so the model is optimized for answer generation rather than prompt copying.
- Low-resource augmentation is language-safe. English donor and synthetic
  examples are not mixed into non-English adapters unless a same-language donor
  is available.
- Training uses dev-set early stopping when a `dev` split is available. The
  scripts evaluate and save every 50 steps, track `eval_loss`, reload the best
  checkpoint at the end, and stop after the language-specific patience is
  exhausted.
- Treat `num_epochs` in the config as a ceiling. Low-resource adapters may stop
  much earlier when dev loss starts rising.
- The train scripts run evaluation after training unless `--eval_only` is used.
- Exact Match is strict for generative SRH answers, so review token F1,
  ROUGE-L, and manual samples.
