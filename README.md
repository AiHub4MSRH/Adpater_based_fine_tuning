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
| Evaluate external HF adapter | `python3 medigemma/evaluate_external_adapter.py` | N/A |
| LLM judge comparison CSV | `python3 judge_comparison_with_openai.py` | `python3 judge_comparison_with_openai.py` |
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

### 5.1 Evaluate Darius / Brainiac HF Adapters With Our Pipeline

Use this to evaluate an external PEFT adapter on the same data splits and
automatic metrics we use for our own MedGemma comparisons. The row-level CSV is
compatible with `judge_comparison_with_openai.py`: the external model is written
as `adapter_prediction`.

Start with `gen7454`, because it is the Brainiac adapter trained across all
eight challenge subsets. Use `--load_in_4bit` for the first run unless you
explicitly want full-weight loading.

The Brainiac path uses vLLM. If `python3 -c "import vllm"` fails, use an image
or environment with vLLM installed before running the commands below; this is
the same serving dependency used by Darius' generation scripts.

If your virtual environment was created before this external-adapter evaluator
was added, upgrade the ML stack first:

```bash
pip install -U -r requirements.txt
```

If a previous upgrade installed `torch==2.13.x` inside `.venv` and you see a
`torchvision ... requires torch==2.12.0` conflict, remove the local Torch wheel
so the CUDA-matched image wheel is used again:

```bash
pip uninstall -y torch triton cuda-toolkit
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Smoke test across all supported adapter targets:

```bash
python3 medigemma/evaluate_external_adapter.py \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --adapter_repo DariusTheGeek/mhqa-itu-adapters \
  --adapter_subfolder gen7454 \
  --adapter_base_model google/gemma-4-31B-it \
  --candidate_name darius_gen7454 \
  --inference_backend vllm \
  --prompt_style darius \
  --skip_baseline \
  --load_in_4bit \
  --tensor_parallel_size 1 \
  --max_model_len 2048 \
  --lora_rank 64 \
  --max_eval_samples 20 \
  --output_dir ./reports
```

Full evaluation across the available test split:

```bash
python3 medigemma/evaluate_external_adapter.py \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --adapter_repo DariusTheGeek/mhqa-itu-adapters \
  --adapter_subfolder gen7454 \
  --adapter_base_model google/gemma-4-31B-it \
  --candidate_name darius_gen7454 \
  --inference_backend vllm \
  --prompt_style darius \
  --skip_baseline \
  --load_in_4bit \
  --tensor_parallel_size 1 \
  --max_model_len 2048 \
  --lora_rank 64 \
  --max_eval_samples 100000 \
  --output_dir ./reports
```

This writes:

- `reports/darius_gen7454_test_comparison.csv`
- `reports/darius_gen7454_test_comparison_report.json`

For Darius' Gemma-4 adapter, prefer `--inference_backend vllm`. His published
`gen7454` adapter targets Gemma-4 clippable linear layers that vanilla PEFT does
not inject into correctly, while his own generation code serves the LoRA through
vLLM's `LoRARequest`.

The default `--prompt_style hashie` is the fairest apples-to-apples comparison
against our adapters for ordinary external adapters. For Brainiac, start with
`--prompt_style darius`, because it mirrors his subset-aware inference prompt.
To run source-leaf reporting with his prompt style:

```bash
python3 medigemma/evaluate_external_adapter.py \
  --data_root ./data \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --adapter_repo DariusTheGeek/mhqa-itu-adapters \
  --adapter_subfolder gen7454 \
  --adapter_base_model google/gemma-4-31B-it \
  --candidate_name darius_gen7454_native_prompt \
  --inference_backend vllm \
  --prompt_style darius \
  --use_source_leaves \
  --require_source_leaves \
  --max_eval_samples_per_source 100 \
  --skip_baseline \
  --load_in_4bit \
  --tensor_parallel_size 1 \
  --max_model_len 2048 \
  --lora_rank 64 \
  --output_dir ./reports
```

If your local `./data` directory only contains combined `eng/` and `swa/`
folders, use `--dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET` so the evaluator
can fetch `eng_eth`, `eng_gha`, `eng_ken`, `eng_uga`, `swa_ken`, and `swa_uga`.
Keep `--require_source_leaves` on for Brainiac checks so the run fails loudly
instead of silently falling back to combined `eng` and `swa` data.

Run the OpenAI judge on the external-adapter comparison:

```bash
python3 judge_comparison_with_openai.py \
  --comparison_csv ./reports/darius_gen7454_test_comparison.csv \
  --output_csv ./reports/darius_gen7454_test_llm_judged.csv \
  --report_json ./reports/darius_gen7454_test_llm_judge_report.json \
  --batch_size 8 \
  --concurrency 4
```

If you run with `--skip_baseline`, judge only the external adapter column:

```bash
python3 judge_comparison_with_openai.py \
  --comparison_csv ./reports/darius_gen7454_test_comparison.csv \
  --prediction_columns adapter_prediction \
  --output_csv ./reports/darius_gen7454_test_llm_judged.csv \
  --report_json ./reports/darius_gen7454_test_llm_judge_report.json \
  --batch_size 8 \
  --concurrency 4
```

The other published Brainiac adapters are narrower:

- `mg_2226` uses `google/medgemma-27b-text-it` and was trained for the Ghana
  generation subsets.
- `raft_r1` expects a merged `gen7454` base, so do not use it first unless that
  merged base is available locally.

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
- Exact Match is too strict to use as the headline metric for generative SRH
  answers. Treat it as useful only for categorical or multiple-choice checks.
- For free-form medical answers, review character n-gram F1, token F1,
  ROUGE-L, target-script match, Latin/code-switch leakage, repetition rate,
  length ratio, quality-flag rate, and manual sample predictions.
- Clinical acceptance should use a separate rubric: factual correctness,
  coverage of required clinical facts, absence of harmful advice, appropriate
  referral/triage language, and target-language fluency.

### Optional LLM-as-Judge Review

Use LLM-as-judge after `compare_models.py` has produced a row-level comparison
CSV. The judge scores each baseline and adapter answer using a clinical SRH
rubric:

- `clinical_correctness`
- `completeness`
- `safety`
- `language_quality`
- `helpfulness`
- `overall`
- `critical_error`
- `harmful_advice`

Set the OpenAI token as an environment variable. Do not paste tokens into the
command or commit them to the repo.

```bash
export OPENAI_API_KEY="your-token-here"
```

Run the judge on a comparison CSV:

```bash
python3 judge_comparison_with_openai.py \
  --comparison_csv ./reports/amh_baseline_vs_adapter_v2.csv \
  --output_csv ./reports/amh_baseline_vs_adapter_llm_judged.csv \
  --report_json ./reports/amh_llm_judge_report.json \
  --batch_size 8 \
  --concurrency 4
```

The judge uses rubric versioning and a local cache. When the rubric changes,
new cache keys are used automatically. The current rubric also applies
deterministic metric caps: highly repetitive, wrong-script, code-switched, or
runaway-long generations cannot receive high `language_quality`, `helpfulness`,
or `overall` scores even if the LLM judge finds some correct clinical fragments.

For a small smoke test:

```bash
python3 judge_comparison_with_openai.py \
  --comparison_csv ./reports/amh_baseline_vs_adapter_v2.csv \
  --output_csv ./reports/amh_llm_judge_smoke.csv \
  --report_json ./reports/amh_llm_judge_smoke.json \
  --max_rows 5 \
  --batch_size 5 \
  --concurrency 1
```

LLM judging is a screening layer, not final clinical approval. Use it to compare
model versions and prioritize manual review; final deployment decisions should
still include clinician review of sampled outputs and all high-risk failures.
