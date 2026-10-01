#!/usr/bin/env python3
"""Extract hidden states for the non-parallel reference corpus on one GPU.

The script builds the baseline hidden-state tensor used by
``calculate_intrinsic_scores.py`` when non-parallel evaluation is enabled.
It supports the same MELD, MEXA, and Language Ranker (LR) pooling settings as
``extract_hidden_states.py`` and preserves the baseline tensor format expected
by the scoring code.
"""

import argparse
import gc
import os
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden states for the non-parallel GLUE reference corpus "
            "on a single GPU."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str.lower,
        choices=["meld", "mexa", "lr"],
        help="Extraction method: meld, mexa, or lr.",
    )
    parser.add_argument(
        "--subgroup",
        type=int,
        default=None,
        help=(
            "Non-parallel corpus subgroup to extract. If omitted, the subgroup "
            "configured under scoring.non_parallel.subgroup is used."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Expose exactly one physical GPU before importing torch / initializing CUDA.
    gpu_id = str(config["runtime"]["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        Mistral3Model,
        MistralCommonBackend,
    )

    seed = int(config["seed"])
    cache_dir = config["paths"]["cache_dir"]
    output_dir = config["paths"]["hidden_states_dir"]
    tokenizer_configs = config["tokenizer"]

    method_cfg = config["methods"][args.method]
    pooling = method_cfg["pooling"]

    non_parallel_data_cfg = config["non_parallel_corpus"]
    data_name = non_parallel_data_cfg.get("name", "glue")
    data_main_dir = non_parallel_data_cfg["main_dir"]
    data_sub_set = non_parallel_data_cfg["subset_file"]
    text_column = non_parallel_data_cfg.get("text_column", "text")
    group_column = non_parallel_data_cfg.get("group_column", "group")

    if args.subgroup is None:
        subgroup = int(config["scoring"]["non_parallel"]["subgroup"])
    else:
        subgroup = int(args.subgroup)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Pooling: {pooling}")
    print(f"Non-parallel corpus: {data_name}")
    print(f"Subgroup: {subgroup}")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    # Because CUDA_VISIBLE_DEVICES exposes one GPU, it is cuda:0 internally.
    device = torch.device("cuda:0")

    pl.seed_everything(seed, workers=True)

    # ------------------------------------------------------------
    # Load non-parallel reference data.
    # ------------------------------------------------------------
    data_path = os.path.join(data_main_dir, data_sub_set)
    data_df = pd.read_feather(data_path)
    data_df = data_df[data_df[group_column] == subgroup].copy().reset_index(drop=True)

    texts = data_df.sample(frac=1.0, random_state=seed)[text_column].values

    # ------------------------------------------------------------
    # Load model and tokenizer.
    # ------------------------------------------------------------
    print("Loading model/tokenizer...")

    if "Ministral-3" in args.model:
        print("Mistral3 model detected.")
        model = Mistral3Model.from_pretrained(
            args.model,
            cache_dir=cache_dir,
        )
        tokenizer = MistralCommonBackend.from_pretrained(
            args.model,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModel.from_pretrained(
            args.model,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=cache_dir,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    mdl = model

    print("Model loaded.")

    # ------------------------------------------------------------
    # Extract hidden states.
    # ------------------------------------------------------------
    lang_lyr_mean = []
    en_hs = []

    for i, input_text in enumerate(texts):
        if i % 300 == 0:
            print(f"{i}/{len(texts)}")

        inputs = tokenizer(input_text, **tokenizer_configs)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = mdl(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        mask = inputs["attention_mask"].squeeze(0).bool()

        if pooling == "mean_pool":
            hidden_states = [
                hs.squeeze(0)[mask].mean(dim=0, keepdim=True).cpu()
                for hs in outputs.hidden_states
            ]

        elif pooling == "last_token":
            hidden_states = [
                hs.squeeze(0)[mask][-1].cpu() for hs in outputs.hidden_states
            ]

        elif pooling == "mexa_pool":
            num_tokens = mask.sum().item()
            weights = torch.arange(
                1,
                num_tokens + 1,
                device=outputs.hidden_states[0].device,
                dtype=outputs.hidden_states[0].dtype,
            ).unsqueeze(-1)

            weight_sum = weights.sum()

            hidden_states = [
                (hs.squeeze(0)[mask] * weights)
                .sum(dim=0, keepdim=True)
                .div(weight_sum)
                .cpu()
                for hs in outputs.hidden_states
            ]

        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        en_hs.append(hidden_states)

    # [examples, layers, hidden_dim], after the final squeeze below.
    lang_tensor = torch.stack(
        [torch.stack(group, dim=0) for group in en_hs],
        dim=0,
    )
    lang_lyr_mean.append(lang_tensor)

    del en_hs
    gc.collect()
    torch.cuda.empty_cache()

    # The baseline tensor construction expected by the scorer.
    lang_lyr_hs = torch.stack(lang_lyr_mean, dim=0).float().squeeze()

    print(f"Final shape: {lang_lyr_hs.shape}")

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(args.model.rstrip("/"))
    output_name = f"{model_name}_baseline_{pooling}_{data_name}_{subgroup}.pt"
    output_path = os.path.join(output_dir, output_name)

    torch.save(lang_lyr_hs, output_path)
    print(f"Saved: {output_path}")

    # ------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------
    del model
    del tokenizer
    del lang_lyr_mean
    del lang_lyr_hs

    gc.collect()
    torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
