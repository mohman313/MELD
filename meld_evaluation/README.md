# MELD: Hidden-State Extraction and Intrinsic Scoring

This folder contains the representation-extraction and intrinsic-scoring utilities used for multilingual evaluation with **MELD**, **MEXA**, and **Language Ranker (LR)**.

It provides four command-line utilities:

1. `extract_hidden_states.py` extracts sentence-level hidden representations from FLORES or NTREX.
2. `extract_non_parallel_hidden_states.py` extracts the independent GLUE reference representations used in the non-parallel robustness experiment.
3. `calculate_intrinsic_scores.py` loads the saved tensors and computes a layer-wise intrinsic score for every configured language relative to the configured reference language.
4. `analyze_correlations.py` combines intrinsic scores with downstream benchmark results and produces general model-level correlation data for parallel and non-parallel experiments.

Each extraction/scoring script processes one model per invocation and uses one GPU. Experiment-independent settings are stored in `config.yaml`; run-specific choices are supplied from the command line.

A CUDA-enabled PyTorch installation is required for hidden-state extraction and intrinsic scoring. The analysis script itself does not perform model inference.

## Configuration

Before running the pipeline, review `config.yaml`. The corpora are stored in the `data/` folder by default, and generated files are saved under `outputs/`. Update these paths if needed.

```yaml
runtime:
  gpu_id: 0

paths:
  cache_dir: null
  hidden_states_dir: ./outputs/hidden_states
  scores_dir: ./outputs/scores
```

Set `runtime.gpu_id` to the physical GPU you want to use.
The `corpora` section specifies each corpus's Feather file and language order. Keep this order consistent between extraction and scoring, since it determines the language indices in the saved hidden-state tensor.

## Input corpora

Each multilingual corpus is expected to be a Feather file containing at least:

| Column | Description |
| --- | --- |
| `iso_639_3` | ISO 639-3 language code |
| `text` | Sentence text passed to the model |

The repository configuration includes language lists for FLORES and NTREX.

## Methods

The method argument determines both the sentence representation used during extraction and the scoring function used in the second stage.

| Method | Extraction pooling | Scoring function |
| --- | --- | --- |
| `meld` | Mean pooling over non-padding tokens | Inter-/intra-language distribution score |
| `mexa` | Position-weighted mean pooling | Bidirectional retrieval score |
| `lr` | Last non-padding token | Mean aligned-pair cosine similarity |

All model hidden states are retained by the extraction script. Layer-wise scores are computed later by `calculate_intrinsic_scores.py`.

## 1. Extract hidden states

Run:

```bash
python extract_hidden_states.py \
  --model MODEL_NAME_OR_PATH \
  --method {meld,mexa,lr} \
  --corpus {ntrex,flores}
```

Example:

```bash
python extract_hidden_states.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

A local model path can be supplied instead of a Hugging Face identifier.

The saved tensor follows the filename convention:

```text
<model_name>_new_langs_<pooling>_<corpus>.pt
```

and has shape:

```text
[languages, examples, layers, hidden_dim]
```

For example:

```text
Qwen3.5-27B_new_langs_mean_pool_ntrex.pt
```

The tensor is saved in `paths.hidden_states_dir`.

## 2. Calculate intrinsic scores

After extraction, run the scorer with the same model, method, and corpus:

```bash
python calculate_intrinsic_scores.py \
  --model MODEL_NAME_OR_PATH \
  --method {meld,mexa,lr} \
  --corpus {ntrex,flores}
```

Example:

```bash
python calculate_intrinsic_scores.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

The scorer infers the expected hidden-state filename from the method configuration and loads it from `paths.hidden_states_dir`.

Scores are written as a Feather table in `paths.scores_dir`. Each column corresponds to a language and each row corresponds to a model layer.

The output filename follows the experiment naming convention:

```text
<model_name>_maha_<variance_mode>_<pooling>_<scoring>_<corpus>_<reference>.feather
```

For the default MELD configuration, an example is:

```text
Qwen3.5-27B_maha_diagonal_mean_pool_i_score_ntrex_eng.feather
```

## Scoring configuration

The default scoring settings are defined in `config.yaml`:

```yaml
scoring:
  base_language: eng
  variance_mode: diagonal
  eps: 1.0e-12
  unbiased: true
  normalize_dim: true
  per_language_weighting: true
  robust: false
```

`base_language` determines the reference language used to form each two-language comparison.

For MELD-style distribution scoring, `variance_mode` supports:

- `diagonal`: pooled per-dimension variance scaling;
- `full`: pooled full covariance with pseudoinverse;
- `none`: unscaled Euclidean geometry.

The default is `diagonal`.

## PCA ablations

Optional PCA reduction can be enabled in the scoring configuration:

```yaml
scoring:
  pca:
    mode: none
    explained_variance: 0.9
    n_components: 256
```

Supported modes are:

- `none`: no PCA reduction;
- `variance`: retain the minimum number of components required to reach `explained_variance`;
- `components`: retain exactly `n_components` components.

The PCA directions are fitted to pooled within-language-centered representations before the two language distributions are projected into the reduced space.

## Non-parallel reference experiments

The non-parallel experiment replaces the reference-language representation set with representations extracted from an independently sampled English corpus. The baseline extractor uses the GLUE-derived Feather file configured under `non_parallel_corpus`:

```yaml
non_parallel_corpus:
  name: glue
  main_dir: ./data/glue
  subset_file: glue_sst2_qqp_mrpc_qnli_mnli_test_rsample.feather
  text_column: text
  group_column: group
```

The file must contain the configured text column and subgroup column. For a selected subgroup, rows are filtered by the subgroup value and then shuffled with the repository seed before hidden-state extraction.

### Extract the non-parallel baseline

Use the same model and method as the multilingual extraction:

```bash
python extract_non_parallel_hidden_states.py \
  --model MODEL_NAME_OR_PATH \
  --method {meld,mexa,lr} \
  --subgroup 5
```

If `--subgroup` is omitted, the value from `scoring.non_parallel.subgroup` is used. The output is saved in `paths.hidden_states_dir` using:

```text
<model_name>_baseline_<pooling>_glue_<subgroup>.pt
```

For example:

```text
Qwen3.5-27B_baseline_mean_pool_glue_5.pt
```

The baseline tensor has shape:

```text
[examples, layers, hidden_dim]
```

### Enable non-parallel scoring

Set the scorer to use the same subgroup:

```yaml
scoring:
  non_parallel:
    enabled: true
    subgroup: 5
```

Then run `calculate_intrinsic_scores.py` normally for FLORES or NTREX. The scorer loads the corresponding baseline tensor and replaces the configured reference-language samples with the non-parallel samples before applying the selected intrinsic metric.

The baseline tensor must contain at least as many samples as the multilingual representation tensor and must have compatible layer and hidden dimensions.

## 3. Analyze intrinsic/downstream correlations

`analyze_correlations.py` produces general observation tables.

Run both experiment settings:

```bash
python analyze_correlations.py --setting all
```

or run them separately:

```bash
python analyze_correlations.py --setting parallel
python analyze_correlations.py --setting non-parallel
```

The analysis configuration is stored under `analysis:` in `config.yaml`. It specifies the model list, corpora, intrinsic methods, downstream benchmarks, non-parallel subgroups, layer aggregation settings, benchmark-result directory, and analysis output directory.

The script recursively searches `analysis.benchmark_results_dir` for `lm-evaluation-harness` JSON result files and aligns benchmark language scores with the intrinsic language scores.

### Parallel output

The parallel experiment produces:

```text
outputs/analysis/
├── parallel_correlations.csv
└── parallel_correlations.feather
```

Each row corresponds to one:

```text
corpus × method × benchmark × model
```

with columns including:

```text
corpus
method
pooling
scoring
benchmark
model
reference_language
n_languages
pearson
spearman
intrinsic_file
benchmark_file
```

### Non-parallel output

The non-parallel experiment produces:

```text
outputs/analysis/
├── non_parallel_correlations.csv
└── non_parallel_correlations.feather
```

It has the same structure, plus:

```text
subgroup
```

Each row therefore corresponds to:

```text
subgroup × corpus × method × benchmark × model
```

### Example downstream aggregation

One can aggregate the observation tables according to their own analysis needs.

For example, macro-averaging the parallel correlations across models:

```python
import pandas as pd

df = pd.read_feather(
    "outputs/analysis/parallel_correlations.feather"
)

macro = (
    df.groupby(["corpus", "method", "benchmark"])[
        ["pearson", "spearman"]
    ]
    .mean()
)
```

For the non-parallel setting, first obtain one macro correlation per subgroup:

```python
df = pd.read_feather(
    "outputs/analysis/non_parallel_correlations.feather"
)

subgroup_macro = (
    df.groupby(
        ["subgroup", "corpus", "method", "benchmark"]
    )[["pearson", "spearman"]]
    .mean()
)
```

and, if desired, summarize those subgroup-level values:

```python
summary = (
    subgroup_macro.reset_index()
    .groupby(["corpus", "method", "benchmark"])[
        ["pearson", "spearman"]
    ]
    .agg(["mean", "std"])
)
```

Use `--allow-missing` to skip unavailable score/result files with warnings rather than stopping the analysis:

```bash
python analyze_correlations.py \
  --setting all \
  --allow-missing
```

## Large hidden-state tensors

By default, hidden-state tensors are loaded directly onto the selected GPU for scoring. Models listed under:

```yaml
runtime:
  load_hidden_states_on_cpu_models:
    - Llama-3.1-70B
```

are instead loaded into CPU memory, and only the current language/layer slice is moved to the GPU. This can reduce GPU-memory pressure when the saved representation tensor is large.

## Model loading

Most models are loaded during extraction using:

```python
AutoModel.from_pretrained(...)
AutoTokenizer.from_pretrained(...)
```

Models whose identifier or local path contains `Ministral-3` use `Mistral3Model` and `MistralCommonBackend`.

If the tokenizer does not define a padding token, the EOS token is used as the padding token.

## Reproducibility

The extraction seed, tokenizer settings, corpus language order, scoring parameters, reference language, PCA settings, analysis settings, and GPU configuration are recorded in `config.yaml`.

For reproducible experiments, keep the configuration file used for each run together with the generated hidden-state, score, and analysis files.

A custom configuration file can be supplied with:

```bash
--config /path/to/config.yaml
```
