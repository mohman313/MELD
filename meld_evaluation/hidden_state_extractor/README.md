# Hidden-State Extraction

This repository provides a utility for extracting multilingual hidden-state representations from Hugging Face language models. The extracted representations are used by the evaluation pipelines for **MELD**, **MEXA**, and **Language Ranker (LR)**.

The extractor accepts three experiment-specific arguments:

- a model name or local model path;
- an extraction method: `meld`, `mexa`, or `lr`;
- a multilingual corpus: `ntrex` or `flores`.

All fixed settings, including dataset paths, language lists, tokenizer settings, cache location, output directory, and GPU ID, are defined in `config.yaml`.

A CUDA-enabled PyTorch installation is required for GPU inference. The selected model must fit on a single GPU.

## Data format

The extractor expects each corpus to be stored as a Feather file. The file must contain at least the following columns:

| Column | Description |
| --- | --- |
| `iso_639_3` | ISO 639-3 language code |
| `text` | Sentence text passed to the model |

The corpus locations and filenames are configured in `config.yaml`.

## Configuration

Edit `config.yaml` before running an experiment.

```yaml
seed: 313

runtime:
  gpu_id: 0

paths:
  cache_dir: /path/to/huggingface/cache
  output_dir: ./hidden_states
```

### GPU

`runtime.gpu_id` specifies the physical GPU that should be visible to the process. The script exposes only that GPU and uses it internally as `cuda:0`.

For example:

```yaml
runtime:
  gpu_id: 2
```

runs the extraction on physical GPU 2.

### Corpora

Each corpus entry defines its directory, Feather filename, and language list:

```yaml
corpora:
  ntrex:
    main_dir: /path/to/NTREX-128/
    subset_file: ntrex_128_subset.feather
    languages:
      - eng
      - fra
      - deu
```

The repository configuration contains the complete language lists used for the experiments.

### Method-specific pooling

The extraction method determines how token-level hidden states are converted into one representation per sentence and layer:

| Method | Pooling | Representation |
| --- | --- | --- |
| `meld` | `mean_pool` | Mean over non-padding token representations |
| `mexa` | `mexa_pool` | Position-weighted mean over non-padding token representations |
| `lr` | `last_token` | Last non-padding token representation |

The extractor stores the returned hidden state for every model layer. Layer selection required by a downstream metric is performed by the corresponding evaluation code.

## Usage

The command-line interface is:

```bash
python extract_hidden_states.py \
  --model MODEL_NAME_OR_PATH \
  --method {meld,mexa,lr} \
  --corpus {ntrex,flores}
```

### MELD

```bash
python extract_hidden_states.py \
  --model Qwen/Qwen3.5-27B \
  --method meld \
  --corpus ntrex
```

A Hugging Face model identifier can also be supplied instead of a local path, provided the model can be loaded by the configured Transformers installation.

To use a different YAML file:

```bash
python extract_hidden_states.py \
  --model MODEL_NAME_OR_PATH \
  --method meld \
  --corpus ntrex \
  --config /path/to/config.yaml
```

## Output

The extractor saves one PyTorch tensor per run using the filename convention:

```text
<model_name>_<pooling>_<corpus>.pt
```

Examples:

```text
Qwen3.5-27B_mean_pool_ntrex.pt
Llama-3.1-8B_mexa_pool_flores.pt
```

The output is saved under `paths.output_dir` from `config.yaml`.

For the configured multilingual corpora, the tensor is organized as:

```text
[languages, examples, layers, hidden_dim]
```

It can be loaded directly with PyTorch:

```python
import torch

hidden_states = torch.load(
    "Qwen3.5-27B_mean_pool_ntrex.pt",
    map_location="cpu",
)

print(hidden_states.shape)
```

## Model loading

Most models are loaded with:

```python
AutoModel.from_pretrained(...)
AutoTokenizer.from_pretrained(...)
```

Models whose path or identifier contains `Ministral-3` are loaded with `Mistral3Model` and `MistralCommonBackend`.

If a tokenizer does not define a padding token, the EOS token is used as the padding token.

## Reproducibility

The random seed is defined in `config.yaml`. Sentences are passed directly to the model without task prompts or templates, and hidden states are extracted with the model in evaluation mode under `torch.inference_mode()`.

The output filename records the model name, pooling strategy, and corpus. Keep the corresponding `config.yaml` with experiment outputs so that the corpus paths, language ordering, tokenizer settings, and seed used for an extraction remain traceable.
