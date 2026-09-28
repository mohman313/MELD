#!/usr/bin/env python3
"""Calculate layer-wise multilingual intrinsic scores from saved hidden states.

The script consumes hidden-state tensors produced by ``extract_hidden_states.py``
and computes one of three intrinsic metrics:

- MELD (``--method meld``)
- MEXA (``--method mexa``)
- Language Ranker (``--method lr``)

One model is processed per invocation on a single configured GPU. Fixed
experiment settings are read from ``config.yaml``.
"""

import argparse
import os
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate layer-wise MELD, MEXA, or Language Ranker scores."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name or path used during hidden-state extraction. Only the "
            "final path component is used to construct the hidden-state filename."
        ),
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str.lower,
        choices=["meld", "mexa", "lr"],
        help="Intrinsic scoring method: meld, mexa, or lr.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=str.lower,
        choices=["ntrex", "flores"],
        help="Corpus used to extract the hidden states: ntrex or flores.",
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


def model_basename(model_name_or_path):
    return os.path.basename(model_name_or_path.rstrip("/"))


def non_parallel_lang(H, BH, reference_idx=0):
    """Replace the reference-language samples with non-parallel baseline samples."""
    H[reference_idx] = BH[: H.size(1)]
    return H


def pca_reduce_components(H, n_components):
    """Project two language distributions onto a fixed number of PCA components."""
    import torch

    if H.ndim != 3:
        raise ValueError("H must have shape (2, N, d).")

    n_languages, N, d = H.shape

    if n_languages != 2:
        raise ValueError("Expected exactly two languages.")

    max_components = min(2 * N - 2, d)

    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components must be between 1 and {max_components}, got {n_components}."
        )

    # Fit PCA to pooled within-language variation. Each language is centered
    # around its own centroid before estimating the PCA directions.
    language_means = H.mean(dim=1, keepdim=True)
    H_within = H - language_means
    X_fit = H_within.reshape(2 * N, d)

    _, _, Vh = torch.linalg.svd(X_fit, full_matrices=False)
    components = Vh[:n_components].T

    # Project from a common origin so the between-language centroid difference
    # is preserved.
    pooled_mean = H.reshape(2 * N, d).mean(dim=0, keepdim=True)
    X = H.reshape(2 * N, d) - pooled_mean
    X_pca = X @ components

    return X_pca.reshape(2, N, n_components)


def pca_reduce_variance(H, explained_variance):
    """Project two language distributions while retaining target PCA variance."""
    import torch

    if H.ndim != 3:
        raise ValueError("H must have shape (2, N, d).")

    L, N, d = H.shape

    if L != 2:
        raise ValueError("Expected exactly two languages.")

    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    # Fit PCA using pooled within-language variation.
    language_means = H.mean(dim=1, keepdim=True)
    H_within = H - language_means
    X_fit = H_within.reshape(2 * N, d)

    _, S, Vh = torch.linalg.svd(X_fit, full_matrices=False)

    # The covariance denominator cancels in explained-variance ratios.
    variance = S.square()
    explained_ratio = variance / variance.sum()
    cumulative = torch.cumsum(explained_ratio, dim=0)

    target = torch.as_tensor(
        explained_variance,
        device=cumulative.device,
        dtype=cumulative.dtype,
    )

    k = int(torch.searchsorted(cumulative, target).item()) + 1
    k = min(k, cumulative.numel())
    components = Vh[:k].T

    # Use a common origin to preserve between-language centroid differences.
    pooled_mean = H.reshape(2 * N, d).mean(dim=0, keepdim=True)
    X = H.reshape(2 * N, d) - pooled_mean
    X_pca = X @ components

    return X_pca.reshape(2, N, k)


def inter_intra_ratio_maha(
    H,
    eps=1e-12,
    unbiased=True,
    normalize_dim=True,
    per_language_weighting=True,
    robust=False,
    variance_mode="none",
    pca_mode="none",
    pca_ratio=0.9,
    pca_components=256,
):
    """Compute the inter-language share of inter + intra dispersion.

    ``H`` must have shape ``(L, N, d)``. The supported variance modes are
    ``none``, ``diagonal``, and ``full``. Optional PCA reduction is applied
    before the distance calculation and is intended for covariance ablations.
    """
    import torch

    if pca_mode == "variance":
        H = pca_reduce_variance(H, explained_variance=pca_ratio)
    elif pca_mode == "components":
        H = pca_reduce_components(H, n_components=pca_components)
    elif pca_mode != "none":
        raise ValueError(
            "pca_mode must be one of: none, variance, components."
        )

    L, N, d = H.shape

    if L < 2:
        raise ValueError(f"Expected at least two languages, got L={L}.")

    # Centroids.
    if robust:
        centroids = H.median(dim=1).values
    else:
        centroids = H.mean(dim=1)

    diffs = H - centroids[:, None, :]

    # Within-language dispersion.
    if variance_mode == "diagonal":
        diffs_flat = diffs.reshape(L * N, d)
        denom = (L * N - L) if unbiased else (L * N)
        var = (diffs_flat**2).sum(dim=0) / max(denom, 1)
        inv_var = 1.0 / (var + eps)

        def scaled_sq(vecs):
            return (vecs**2 * inv_var).sum(dim=-1)

        per_sample = scaled_sq(diffs)

        if per_language_weighting:
            intra = per_sample.mean(dim=1).mean()
        else:
            intra = per_sample.mean()

    elif variance_mode == "full":
        diffs_flat = diffs.reshape(L * N, d)
        denom = (L * N - L) if unbiased else (L * N)
        cov = (diffs_flat.T @ diffs_flat) / max(denom, 1)
        cov_inv = torch.linalg.pinv(
            cov + eps * torch.eye(d, device=H.device, dtype=H.dtype)
        )

        def maha_sq(vecs):
            return torch.einsum("...i,ij,...j->...", vecs, cov_inv, vecs)

        per_sample = maha_sq(diffs)

        if per_language_weighting:
            intra = per_sample.mean(dim=1).mean()
        else:
            intra = per_sample.mean()

    elif variance_mode == "none":
        if robust:
            per_sample = diffs.abs().sum(dim=-1)
        else:
            per_sample = (diffs**2).sum(dim=-1)

        if unbiased and N > 1 and not robust:
            per_sample = per_sample * (N / (N - 1))

        if per_language_weighting:
            intra = per_sample.mean(dim=1).mean()
        else:
            intra = per_sample.mean()

    else:
        raise ValueError(
            "variance_mode must be one of: none, diagonal, full."
        )

    # Between-language dispersion.
    if variance_mode == "diagonal":
        c_diffs = centroids[:, None, :] - centroids[None, :, :]
        dist2 = (c_diffs**2 * inv_var).sum(dim=-1)
    elif variance_mode == "full":
        c_diffs = centroids[:, None, :] - centroids[None, :, :]
        dist2 = torch.einsum("ijk,kl,ijl->ij", c_diffs, cov_inv, c_diffs)
    else:
        dist2 = torch.cdist(centroids, centroids, p=2).pow(2)

    inter = dist2.triu(diagonal=1).sum() * 2.0 / (L * (L - 1))

    if normalize_dim and variance_mode == "none":
        inter = inter / d
        intra = intra / d

    ratio = inter / (intra + inter + eps)
    return ratio


def mexa_score(hidden_states):
    """Compute the bidirectional MEXA retrieval score for one layer."""
    import torch
    import torch.nn.functional as F

    assert hidden_states.ndim == 3, (
        f"Expected [2, N, d], got {hidden_states.shape}"
    )
    assert hidden_states.shape[0] == 2, (
        f"MEXA expects 2 languages, got {hidden_states.shape[0]}"
    )

    lang1 = F.normalize(hidden_states[0], p=2, dim=-1)
    lang2 = F.normalize(hidden_states[1], p=2, dim=-1)

    similarity = lang1 @ lang2.T
    diagonal = similarity.diagonal()

    N = similarity.shape[0]
    mask = torch.eye(N, device=similarity.device, dtype=torch.bool)

    row_max = similarity.masked_fill(mask, -torch.inf).max(dim=1).values
    col_max = similarity.masked_fill(mask, -torch.inf).max(dim=0).values

    correct = (diagonal > row_max) & (diagonal > col_max)
    return correct.float().mean()


def language_ranker_score(hidden_states):
    """Compute mean aligned-pair cosine similarity for one layer."""
    import torch.nn.functional as F

    assert hidden_states.ndim == 3, (
        f"Expected [2, N, d], got {hidden_states.shape}"
    )
    assert hidden_states.shape[0] == 2, (
        f"Language Ranker expects 2 languages, got {hidden_states.shape[0]}"
    )

    lang1 = hidden_states[0]
    lang2 = hidden_states[1]

    assert lang1.shape == lang2.shape, (
        "The two languages must contain aligned sentence pairs."
    )

    lang1 = F.normalize(lang1, p=2, dim=-1)
    lang2 = F.normalize(lang2, p=2, dim=-1)

    similarities = (lang1 * lang2).sum(dim=-1)
    return similarities.mean()


def build_note(scoring_cfg):
    """Build the score-file suffix used by the experiment pipeline."""
    base_language = scoring_cfg["base_language"]
    note = f"_{base_language}"

    non_parallel_cfg = scoring_cfg["non_parallel"]
    if non_parallel_cfg["enabled"]:
        subgroup = int(non_parallel_cfg["subgroup"])
        note = f"{note}_non_parallel_{subgroup}"

    pca_cfg = scoring_cfg["pca"]
    pca_mode = pca_cfg["mode"]

    if pca_mode == "variance":
        note = f"{note}_pca{pca_cfg['explained_variance']}"
    elif pca_mode == "components":
        note = f"{note}_pca{pca_cfg['n_components']}"

    return note


def main():
    args = parse_args()
    config = load_config(args.config)

    # Expose exactly one physical GPU before importing torch / initializing CUDA.
    gpu_id = str(config["runtime"]["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    device = torch.device("cuda:0")

    method_cfg = config["methods"][args.method]
    pooling = method_cfg["pooling"]
    scoring = method_cfg["scoring"]

    corpus_cfg = config["corpora"][args.corpus]
    lang_pairs = corpus_cfg["languages"]

    scoring_cfg = config["scoring"]
    base_lang = scoring_cfg["base_language"]
    variance_mode = scoring_cfg["variance_mode"]
    non_parallel_cfg = scoring_cfg["non_parallel"]
    pca_cfg = scoring_cfg["pca"]

    if base_lang not in lang_pairs:
        raise ValueError(
            f"Base language '{base_lang}' is not configured for corpus '{args.corpus}'."
        )

    model_name = model_basename(args.model)
    hidden_states_dir = Path(config["paths"]["hidden_states_dir"])
    scores_dir = Path(config["paths"]["scores_dir"])
    scores_dir.mkdir(parents=True, exist_ok=True)

    input_path = hidden_states_dir / (
        f"{model_name}_new_langs_{pooling}_{args.corpus}.pt"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Hidden-state file not found: {input_path}\n"
            "Run extract_hidden_states.py with the same model, method, and corpus first."
        )

    # Preserve the original memory strategy: selected very large hidden-state
    # tensors can remain on CPU while each language/layer slice is moved to GPU.
    cpu_models = set(config["runtime"].get("load_hidden_states_on_cpu_models", []))
    load_location = "cpu" if model_name in cpu_models else device

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Model: {model_name}")
    print(f"Method: {args.method}")
    print(f"Corpus: {args.corpus}")
    print(f"Pooling: {pooling}")
    print(f"Scoring: {scoring}")
    print(f"Loading hidden states from: {input_path}")
    print(f"Hidden-state load location: {load_location}")

    lang_lyr_hs = torch.load(input_path, map_location=load_location)

    baseline_lang_lyr_hs = None
    baseline_data_name = None

    if non_parallel_cfg["enabled"]:
        subgroup = int(non_parallel_cfg["subgroup"])
        baseline_data_name = f"glue_{subgroup}"
        baseline_path = hidden_states_dir / (
            f"{model_name}_baseline_{pooling}_{baseline_data_name}.pt"
        )

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Non-parallel baseline file not found: {baseline_path}"
            )

        print(f"Using non-parallel reference samples from: {baseline_path}")
        baseline_lang_lyr_hs = torch.load(
            baseline_path,
            map_location=load_location,
        )

    base_lang_idx = lang_pairs.index(base_lang)
    score_lang_lyr_dict = {}

    for lang_idx, lang in enumerate(lang_pairs):
        print(f"{model_name}: {lang}")
        score_lyr = []

        for lyr in range(lang_lyr_hs.shape[-2]):
            H = lang_lyr_hs[[base_lang_idx, lang_idx], :, lyr, :].to(
                device=device
            )

            if non_parallel_cfg["enabled"]:
                BH = baseline_lang_lyr_hs[:, lyr, :]
                H = non_parallel_lang(H, BH)

            if scoring == "i_score":
                score = inter_intra_ratio_maha(
                    H,
                    eps=float(scoring_cfg["eps"]),
                    unbiased=bool(scoring_cfg["unbiased"]),
                    normalize_dim=bool(scoring_cfg["normalize_dim"]),
                    per_language_weighting=bool(
                        scoring_cfg["per_language_weighting"]
                    ),
                    robust=bool(scoring_cfg["robust"]),
                    variance_mode=variance_mode,
                    pca_mode=pca_cfg["mode"],
                    pca_ratio=float(pca_cfg["explained_variance"]),
                    pca_components=int(pca_cfg["n_components"]),
                )
            elif scoring == "mexa_score":
                score = mexa_score(H)
            elif scoring == "lr_score":
                score = language_ranker_score(H)
            else:
                raise ValueError(f"Unsupported scoring function: {scoring}")

            score_lyr.append(score.item())

        score_lang_lyr_dict[lang] = score_lyr

    df = pd.DataFrame(score_lang_lyr_dict)

    note = build_note(scoring_cfg)
    output_name = (
        f"{model_name}_maha_{variance_mode}_{pooling}_{scoring}_"
        f"{args.corpus}{note}.feather"
    )
    output_path = scores_dir / output_name

    df.to_feather(output_path)
    print(f"Saved: {output_path}")

    del lang_lyr_hs
    if baseline_lang_lyr_hs is not None:
        del baseline_lang_lyr_hs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
