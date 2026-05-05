# Adapter-Based Fine-Tuning for Multilingual SRH Models

This repository supports two LoRA/QLoRA fine-tuning pipelines for the same
multilingual sexual and reproductive health (SRH) dataset:

- `medigemma/` for `google/medgemma-4b-it`
- `meditron train scripts/` for `epfl-llm/meditron-7b`

Both pipelines follow the same leaf-based workflow:

1. Mirror multilingual dataset leaves from Hugging Face to local disk
2. Train one adapter per dataset leaf
3. Evaluate adapters on the matching split
4. Compare adapters against a baseline model
5. Push adapters to Hugging Face
6. Run inference from published adapters

The training unit is a dataset leaf such as `eng_uga` or `swa_ken`, not just a
base language code like `eng` or `swa`.

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

Grouped CLI selections expand to these dataset leaves:

- `aka` -> `aka_gha`
- `amh` -> `amh_eth`
- `eng` -> `eng_eth eng_gha eng_ken eng_uga`
- `lug` -> `lug_uga`
- `swa` -> `swa_ken swa_uga`

You can also pass explicit leaves such as `eng_uga` or `swa_ken`.

Current full leaf set in this repo:

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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
```

## Script Map

| Task | MedGemma | Meditron |
| --- | --- | --- |
| Mirror data | `python3 medigemma/prepare_data.py` | `python3 'meditron train scripts/prepare_data.py'` |
| Train or eval | `python3 medigemma/train.py` | `python3 'meditron train scripts/train.py'` |
| Compare baseline vs adapter | `python3 medigemma/compare_models.py` | `python3 'meditron train scripts/compare_models.py'` |
| Push adapters to Hub | `python3 medigemma/push_adapters_to_hub.py` | `python3 'meditron train scripts/push_adapters_to_hub.py'` |
| Run Hub inference | `python3 medigemma/run_inference_from_hub.py` | `python3 'meditron train scripts/run_inference_from_hub.py'` |

## Recommended Output Layout

To avoid mixing adapters from different base models, keep outputs separate:

```text
data/
├── aka_gha/
├── amh_eth/
├── eng_eth/
├── eng_gha/
├── eng_ken/
├── eng_uga/
├── lug_uga/
├── swa_ken/
└── swa_uga/

adapters/
├── medigemma/
└── meditron/
```

## Typical Workflow

The commands below use the same dataset cache for both models, but separate
adapter output roots.

### 1. Mirror the Dataset Locally

MedGemma:

```bash
python3 medigemma/prepare_data.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

Meditron:

```bash
python3 'meditron train scripts/prepare_data.py' \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

### 2. Train One Adapter as a Smoke Test

MedGemma:

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages amh \
  --output_root ./adapters/medigemma \
  --max_eval_samples 50
```

Meditron:

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages amh \
  --output_root ./adapters/meditron \
  --max_eval_samples 50
```

### 3. Train a Single Leaf for Production Runs

MedGemma:

```bash
python3 medigemma/train.py \
  --data_root ./data \
  --languages eng_uga \
  --output_root ./adapters/medigemma \
  --max_eval_samples 100
```

Meditron:

```bash
python3 'meditron train scripts/train.py' \
  --data_root ./data \
  --languages eng_uga \
  --output_root ./adapters/meditron \
  --max_eval_samples 100
```

If disk space is limited, train one leaf at a time and remove stale
`checkpoint-*` directories after successful runs:

```bash
find ./adapters -type d -name "checkpoint-*" -prune -exec rm -rf {} +
```

### 4. Evaluate Existing Adapters

If you want to evaluate the exact full leaf set explicitly instead of using
group aliases, use:

```bash
python3 'meditron train scripts/train.py' \
  --eval_only \
  --data_root ./data \
  --languages eng_uga eng_eth eng_gha eng_ken aka_gha amh_eth lug_uga swa_ken swa_uga \
  --output_root ./adapters/meditron \
  --max_eval_samples 100000
```

MedGemma:

```bash
python3 medigemma/train.py \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/medigemma \
  --max_eval_samples 200
```

Meditron:

```bash
python3 'meditron train scripts/train.py' \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters/meditron \
  --max_eval_samples 200
```

### 5. Train Directly From a Hugging Face Dataset Repo

MedGemma:

```bash
python3 medigemma/train.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters/medigemma
```

Meditron:

```bash
python3 'meditron train scripts/train.py' \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters/meditron
```

## Comparing Baselines and Adapters

MedGemma:

```bash
python3 medigemma/compare_models.py \
  --data_root ./data \
  --languages eng_uga swa_ken \
  --output_root ./adapters/medigemma \
  --baseline_model /path/to/your/original-finetuned-medgemma \
  --max_eval_samples 100 \
  --load_in_4bit
```

Meditron:

```bash
python3 'meditron train scripts/compare_models.py' \
  --data_root ./data \
  --languages eng_uga swa_ken \
  --output_root ./adapters/meditron \
  --baseline_model /path/to/your/original-finetuned-meditron \
  --max_eval_samples 100 \
  --load_in_4bit
```

These commands write:

- `adapter_baseline_comparison.csv`
- `baseline_eval_report.json`
- `adapter_comparison_report.json`

## Pushing Adapters to Hugging Face

If you want to push the exact full Meditron leaf set explicitly, use:

```bash
python3 'meditron train scripts/push_adapters_to_hub.py' \
  --repo_id your-org/hashie-srh-meditron-adapters \
  --output_root ./adapters/meditron \
  --languages eng_uga eng_eth eng_gha eng_ken aka_gha amh_eth lug_uga swa_ken swa_uga \
  --private
```

MedGemma:

```bash
python3 medigemma/push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-medgemma-adapters \
  --output_root ./adapters/medigemma \
  --private
```

Meditron:

```bash
python3 'meditron train scripts/push_adapters_to_hub.py' \
  --repo_id your-org/hashie-srh-meditron-adapters \
  --output_root ./adapters/meditron \
  --private
```

Published repos use a layout like:

```text
README.md
adapters/
├── manifest.json
├── aka_gha/
├── amh_eth/
├── eng_eth/
├── eng_gha/
├── eng_ken/
├── eng_uga/
├── lug_uga/
├── swa_ken/
└── swa_uga/
reports/
└── eval_report.json
```

## Running Hub Inference

MedGemma:

```bash
python3 medigemma/run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-medgemma-adapters \
  --adapter_name amh_eth \
  --prompt "What are common symptoms of an STI?"
```

Meditron:

```bash
python3 'meditron train scripts/run_inference_from_hub.py' \
  --adapter_repo your-org/hashie-srh-meditron-adapters \
  --adapter_name amh_eth \
  --prompt "What are common symptoms of an STI?"
```

Optional low-memory loading:

```bash
python3 medigemma/run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-medgemma-adapters \
  --adapter_name swa_ken \
  --prompt "Shida ya Ukimwi ni nini?" \
  --load_in_4bit
```

```bash
python3 'meditron train scripts/run_inference_from_hub.py' \
  --adapter_repo your-org/hashie-srh-meditron-adapters \
  --adapter_name swa_ken \
  --prompt "Shida ya Ukimwi ni nini?" \
  --load_in_4bit
```

## Practical Notes

- Train one dataset leaf at a time on storage-constrained machines.
- Keep MedGemma and Meditron adapters in different output roots.
- Keep separate Hugging Face repos for each base model family.
- The publish and inference scripts support both flat and nested PEFT adapter
  layouts.
- Exact Match is strict for generative SRH answers, so use token F1 and
  ROUGE-L alongside manual review.

## Quick Commands

Mirror data:

```bash
python3 medigemma/prepare_data.py --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET --languages aka amh eng lug swa --output_root ./data
python3 'meditron train scripts/prepare_data.py' --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET --languages aka amh eng lug swa --output_root ./data
```

Train one leaf:

```bash
python3 medigemma/train.py --data_root ./data --languages amh_eth --output_root ./adapters/medigemma --max_eval_samples 100
python3 'meditron train scripts/train.py' --data_root ./data --languages amh_eth --output_root ./adapters/meditron --max_eval_samples 100
```

Evaluate all:

```bash
python3 medigemma/train.py --eval_only --data_root ./data --languages aka amh eng lug swa --output_root ./adapters/medigemma --max_eval_samples 200
python3 'meditron train scripts/train.py' --eval_only --data_root ./data --languages aka amh eng lug swa --output_root ./adapters/meditron --max_eval_samples 200
```

Evaluate the explicit nine-leaf set for Meditron:

```bash
python3 'meditron train scripts/train.py' --eval_only --data_root ./data --languages eng_uga eng_eth eng_gha eng_ken aka_gha amh_eth lug_uga swa_ken swa_uga --output_root ./adapters/meditron --max_eval_samples 100000
```

Push adapters:

```bash
python3 medigemma/push_adapters_to_hub.py --repo_id your-org/hashie-srh-medgemma-adapters --output_root ./adapters/medigemma --private
python3 'meditron train scripts/push_adapters_to_hub.py' --repo_id your-org/hashie-srh-meditron-adapters --output_root ./adapters/meditron --private
```

Push the explicit nine-leaf Meditron set:

```bash
python3 'meditron train scripts/push_adapters_to_hub.py' --repo_id your-org/hashie-srh-meditron-adapters --output_root ./adapters/meditron --languages eng_uga eng_eth eng_gha eng_ken aka_gha amh_eth lug_uga swa_ken swa_uga --private
```
