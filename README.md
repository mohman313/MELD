# MELD: Measuring Multilingual Capability of Language Models through Latent Distribution Geometry

**MELD (Multilingual Evaluation of Latent Distributions)** is an intrinsic metric for estimating multilingual capability directly from the geometry of language-model hidden states.

MELD is motivated by the observation that multilingual language models often develop a more shared semantic and reasoning space in intermediate layers, while language-specific structure remains stronger near the input and output boundaries. Rather than relying on downstream task evaluation alone, MELD measures how strongly two language-conditioned representation distributions are separated relative to their within-language variation.

## Repository overview

This repository contains the code used for both intrinsic multilingual evaluation and downstream benchmark evaluation.

```text
MELD/
├── README.md
├── meld_evaluation/
│   ├── README.md
│   ├── config.yaml
│   ├── extract_hidden_states.py
│   ├── extract_non_parallel_hidden_states.py
│   ├── calculate_intrinsic_scores.py
│   ├── analyze_correlations.py
│   └── data/
│       ├── flores/
│       │   └── flores_plus_devtest_subset.feather
│       ├── ntrex/
│       │   └── NTREX-128/
│       │       └── ntrex_128_subset.feather
│       └── glue/
│           └── glue_sst2_qqp_mrpc_qnli_mnli_test_rsample.feather
└── benchmark_evaluation/
    ├── README.md
    ├── run_benchmark.sh
    └── lm-evaluation-harness/
```

The two main components are:

- [`meld_evaluation/`](meld_evaluation/README.md): hidden-state extraction and intrinsic scoring for **MELD**, **MEXA**, and **Language Ranker (LR)**, including the non-parallel reference experiment.
- [`benchmark_evaluation/`](benchmark_evaluation/README.md): downstream multilingual benchmark evaluation using `lm-evaluation-harness`.

## MELD in brief

For a reference language \(p\) and target language \(q\), MELD compares their hidden-state distributions at intermediate model layers.

At each layer, the method considers:

- the separation between the two language centroids;
- the within-language dispersion of their representations; and
- a shared variance-normalized geometry.


The default implementation uses a **diagonal Mahalanobis approximation**, which normalizes each hidden dimension by pooled within-language variance and avoids estimating a full high-dimensional covariance matrix.

## Experimental pipeline

The repository supports the following workflow:

```text
Multilingual corpus
        │
        ▼
Hidden-state extraction
        │
        ├── MELD: mean pooling
        ├── MEXA: weighted mean pooling
        └── LR: last-token representation
        │
        ▼
Layer-wise intrinsic scoring
        │
        ▼
Correlation analysis with downstream multilingual benchmarks
```

The same extraction and scoring interface is used for all three intrinsic methods.

## Quick start

### 1. Configure the experiment

Edit:

```text
meld_evaluation/config.yaml
```

to specify dataset locations, output directories, GPU selection, language lists, reference language, and scoring options.

For example:

```yaml
runtime:
  gpu_id: 0

paths:
  cache_dir: null
  hidden_states_dir: ./outputs/hidden_states
  scores_dir: ./outputs/scores

scoring:
  base_language: eng
  variance_mode: diagonal
```

Each script processes one model per invocation and uses one GPU.

### 2. Extract hidden states

From `meld_evaluation/`:

```bash
python extract_hidden_states.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

Supported intrinsic methods are:

```text
meld
mexa
lr
```

Supported multilingual representation corpora are:

```text
ntrex
flores
```

The saved hidden-state tensor has shape:

```text
[languages, examples, layers, hidden_dim]
```

### 3. Calculate intrinsic scores

Run:

```bash
python calculate_intrinsic_scores.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

The scorer computes a layer-wise score for each target language relative to the configured reference language and stores the result as a Feather table.

For complete configuration options and file-naming conventions, see [`meld_evaluation/README.md`](meld_evaluation/README.md).

## Supported intrinsic methods

| Method | Representation | Intrinsic signal |
| --- | --- | --- |
| **MELD** | Mean-pooled hidden states | Distributional separation relative to within-language variation |
| **MEXA** | Position-weighted mean pooling | Bidirectional cross-lingual retrieval |
| **Language Ranker (LR)** | Last-token hidden states | Mean cosine similarity between aligned sentence representations |

MELD differs from pair-based approaches such as MEXA and LR in that its scoring function does not require sentence-level bilingual correspondence.

## Representation corpora

The main experiments use:

- **FLORES+** for multilingual parallel representations;
- **NTREX-128** for multilingual parallel representations; and
- an independently sampled **GLUE-derived English corpus** for the non-parallel reference experiment.

The multilingual input files are expected to contain at least:

| Column | Description |
| --- | --- |
| `iso_639_3` | ISO 639-3 language code |
| `text` | Sentence passed to the language model |

The language order configured in `config.yaml` is important because it determines the language dimension of the saved hidden-state tensor.

## Non-parallel evaluation

MELD can also be evaluated without sentence-level correspondence between the reference and target-language corpora.

First extract the independent reference representations:

```bash
python extract_non_parallel_hidden_states.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --subgroup 5
```

Then enable the non-parallel setting in `config.yaml`:

```yaml
scoring:
  non_parallel:
    enabled: true
    subgroup: 5
```

and run the intrinsic scorer normally:

```bash
python calculate_intrinsic_scores.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

See [`meld_evaluation/README.md`](meld_evaluation/README.md) for details.

## Downstream benchmark evaluation

The paper evaluates multilingual capability using five benchmark families:

- **m-MMLU**
- **m-ARC**
- **m-HellaSwag**
- **OpenAI MMMLU**
- **BELEBELE**

Benchmark evaluation is provided under [`benchmark_evaluation/`](benchmark_evaluation/README.md) and uses the included `lm-evaluation-harness`.

A benchmark can be launched with:

```bash
cd benchmark_evaluation

bash run_benchmark.sh \
  belebele \
  Qwen/Qwen3.5-4B \
  0
```

The experiments use zero-shot evaluation for pretrained models and five-shot evaluation for instruction-tuned and reasoning-oriented models.

## Models

The experiments cover **14 open-weight language models** spanning multiple model families, parameter scales, model depths, and training regimes, including models from:

- Gemma
- Llama
- Ministral
- Phi
- Qwen

The complete model list and downstream evaluation settings are documented in [`benchmark_evaluation/README.md`](benchmark_evaluation/README.md).

## Reproducibility

Experiment-independent settings are stored in `meld_evaluation/config.yaml`, including:

- random seed;
- GPU selection;
- tokenizer settings;
- corpus paths and language order;
- reference language;
- covariance/variance mode;
- PCA ablation settings;
- non-parallel subgroup configuration; and
- hidden-state and score output directories.

Run-specific choices are supplied through the command line:

```text
--model
--method
--corpus
```

A custom configuration file can be supplied with:

```bash
--config /path/to/config.yaml
```

For exact reproduction, keep the configuration used for an experiment together with the generated hidden-state and score files.