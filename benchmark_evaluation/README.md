# MELD Benchmark Evaluation

This folder includes `run_benchmark.sh`, the downstream benchmark runner used with the MELD experiments. It evaluates one model on one lm-evaluation-harness task per invocation. 

## Requirements

- Bash, Python, PyTorch, and a CUDA-capable environment suitable for the chosen model.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Hugging Face Accelerate](https://github.com/huggingface/accelerate), installed in the active environment.
- The requested benchmark task must be available in the harness installation.
- Access to the selected Hugging Face model, including authentication or license acceptance where required, or a downloaded local model directory.

The script currently writes results to `./lm-evaluation-harness/eval_results`. You could edit those lines in `run_benchmark.sh` for your machine before running it. It uses `dtype=bfloat16`, and automatic batch sizing; ensure your hardware supports the selected model and precision.

## Installation

Keep `run_benchmark.sh` in `benchmark_evaluation/`, alongside the `lm-evaluation-harness/` folder. Before running the script, install the included harness and its Hugging Face dependencies in your active Python environment.

From the MELD repository root, run:

```bash
cd benchmark_evaluation/lm-evaluation-harness
pip install -e .
pip install "lm_eval[hf]"
cd ..
```

The editable installation uses the local `lm-evaluation-harness` source. After these commands, your working directory is `benchmark_evaluation/`, where you can run the examples below.

## Usage

```bash
bash run_benchmark.sh BENCHMARK MODEL_PATH_OR_ID NUM_FEWSHOT
```

| Argument | Meaning | Example |
| --- | --- | --- |
| `BENCHMARK` | Task name registered with lm-evaluation-harness | `belebele` |
| `MODEL_PATH_OR_ID` | Hugging Face repo ID or local model directory | `Qwen/Qwen3.5-4B` |
| `NUM_FEWSHOT` | Nonnegative number of examples in the evaluation prompt | `0` or `5` |

Examples:

```bash
# A Hugging Face model ID
bash run_benchmark.sh belebele Qwen/Qwen3.5-4B 0

# An instruction-tuned model
bash run_benchmark.sh belebele mistralai/Ministral-3-8B-Instruct-2512-BF16 5
```

The experiments described in the paper used **zero-shot** for pretrained models and **five-shot** for instruction-tuned and reasoning models. Pass the corresponding value explicitly for each run. The paper evaluates m-MMLU, m-ARC, and m-HellaSwag from Okapi, OpenAI MMMLU, and BELEBELE; task identifiers in the original runner include `m_mmlu`, `arc_multilingual`, `hellaswag_multilingual`, `mmmlu`, and `belebele`. Confirm the names with your installed harness before a long run.

## Models in the paper

The paper evaluates 14 models across six families.

| Family | Model directory / experiment identifier | Parameters | Regime | Shots in paper |
| --- | --- | ---: | --- | ---: |
| Gemma-3 | `google/gemma-3-27b-pt` | 27B | Pretrained | 0 |
| Gemma-3 | `google/gemma-3-12b-pt` | 12B | Pretrained | 0 |
| Gemma-4 | `google/gemma-4-31B` | 31B | Pretrained | 0 |
| Gemma-4 | `google/gemma-4-12B` | 12B | Pretrained | 0 |
| Llama-3.1 | `meta-llama/Llama-3.1-70B` | 70B | Pretrained | 0 |
| Llama-3.1 | `meta-llama/Llama-3.1-8B` | 8B | Pretrained | 0 |
| Ministral-3 | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 8B | Instruct | 5 |
| Ministral-3 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3B | Instruct | 5 |
| Qwen3.5 | `Qwen/Qwen3.5-27B` | 27B | Pretrained | 0 |
| Qwen3.5 | `Qwen/Qwen3.5-9B` | 9B | Pretrained | 0 |
| Qwen3.5 | `Qwen/Qwen3.5-4B` | 4B | Pretrained | 0 |
| Phi-4 | `microsoft/Phi-4-reasoning` | 15B | Reasoning | 5 |
| Phi-4-mini | `microsoft/Phi-4-mini-reasoning` | 4B | Reasoning | 5 |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 4B | Instruct | 5 |
