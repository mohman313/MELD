#!/usr/bin/env python3
"""Extract multilingual hidden-state representations on a single GPU.

The script supports MELD, MEXA, and Language Ranker (LR) extraction settings
and stores one PyTorch tensor containing sentence-level representations for
all configured languages and model layers.
"""

import argparse
import gc
import os
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hidden states on one GPU for MELD, MEXA, or LR."
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
        "--corpus",
        required=True,
        type=str.lower,
        choices=["ntrex", "flores"],
        help="Corpus: ntrex or flores.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------
    # Single-GPU setup.
    # This must happen before importing torch / initializing CUDA.
    # ------------------------------------------------------------
    gpu_id = str(config["runtime"]["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import numpy as np
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

    corpus_cfg = config["corpora"][args.corpus]
    data_main_dir = corpus_cfg["main_dir"]
    data_sub_set = corpus_cfg["subset_file"]
    lang_pairs = corpus_cfg["languages"]

    # Select the sentence-level pooling strategy for the requested method.
    pooling = config["methods"][args.method]["pooling"]

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Model:  {args.model}")
    print(f"Method: {args.method}")
    print(f"Corpus: {args.corpus}")
    print(f"Pooling: {pooling}")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    # Because CUDA_VISIBLE_DEVICES exposes one GPU, it is cuda:0 internally.
    device = torch.device("cuda:0")

    # Reproducible inference setup.
    pl.seed_everything(seed, workers=True)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    data_path = os.path.join(data_main_dir, data_sub_set)
    data_df = pd.read_feather(data_path)
    data_df = (
        data_df[data_df["iso_639_3"].isin(lang_pairs)].copy().reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Load model and tokenizer
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

    print("Model loaded.")

    # Model module used for hidden-state extraction.
    mdl = model

    # ------------------------------------------------------------
    # Extract hidden states
    # ------------------------------------------------------------
    lang_lyr_mean = []

    for lang_idx, lang in enumerate(lang_pairs):
        print(f"{lang} ({lang_idx + 1}/{len(lang_pairs)})")

        texts = data_df[data_df["iso_639_3"] == lang]["text"].values

        # FLORES devtest contains 1,012 examples per language.
        if args.corpus == "flores":
            texts = texts[:1012]

        en_hs = []

        for input_text in texts:
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
                # Position-weighted mean pooling used for MEXA.
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

        # Stack sentence representations for the current language.
        lang_tensor = torch.stack(
            [torch.stack(group, dim=0) for group in en_hs],
            dim=0,
        )

        lang_lyr_mean.append(lang_tensor)

        del en_hs
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final tensor
    # [languages, examples, layers, hidden_dim]
    # ------------------------------------------------------------
    lang_lyr_hs = torch.stack(lang_lyr_mean, dim=0).float().squeeze()

    print(f"Final shape: {lang_lyr_hs.shape}")

    # ------------------------------------------------------------
    # Save the complete hidden-state tensor for this run.
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(args.model.rstrip("/"))
    output_name = f"{model_name}_new_langs_{pooling}_{args.corpus}.pt"
    output_path = os.path.join(output_dir, output_name)

    torch.save(lang_lyr_hs, output_path)
    print(f"Saved: {output_path}")

    # ------------------------------------------------------------
    # Cleanup
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
