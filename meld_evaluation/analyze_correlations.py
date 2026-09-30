#!/usr/bin/env python3
"""Analyze intrinsic/downstream correlations for parallel and non-parallel experiments.

This script converts layer-wise intrinsic score files and lm-evaluation-harness
JSON results into two general observation tables:

- parallel_correlations.{csv,feather}
- non_parallel_correlations.{csv,feather}

No paper-table-specific aggregation is performed. Each output row contains the
correlation for one corpus, intrinsic method, downstream benchmark, and model.
The non-parallel output additionally contains the independent-reference subgroup.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


LOGGER = logging.getLogger("meld.analysis")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create model-level intrinsic/downstream correlation data for "
            "parallel and/or non-parallel MELD experiments."
        )
    )
    parser.add_argument(
        "--setting",
        choices=["parallel", "non-parallel", "all"],
        default="all",
        help="Experiment setting to analyze. Default: all.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip missing intrinsic-score or benchmark-result files with a warning. "
            "By default, missing required files raise an error."
        ),
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def middle_percent(indices, pct, min_layers=2):
    """Select a centered fraction of layer indices."""
    n = len(indices)
    if n == 0:
        return []

    k = int(round(n * pct))
    k = max(min_layers, k)
    k = min(k, n)

    start = (n - k) // 2
    return indices[start : start + k]


def auc_score(layer_scores):
    """Normalized middle-layer AUC with the MELD score direction used in the paper."""
    values = np.asarray(layer_scores)
    K = len(values)
    if K < 2:
        raise ValueError("MELD AUC requires at least two selected layers.")

    relative_depth = torch.arange(K, dtype=torch.float32) / (K - 1)
    score = torch.trapz(
        torch.as_tensor(values, dtype=torch.float32),
        relative_depth,
    ).item()
    return 1.0 - score


def layer_steps(indices, step=5):
    """Return every ``step``-th layer, starting at ``step``."""
    return list(range(step, len(indices), step))


def get_lang_map(task):
    """Map lm-evaluation-harness subtask names to ISO 639-3 codes."""
    if task == "mmmlu":
        return {
            "mmmlu_ar_xy": "arb",
            "mmmlu_bn_bd": "ben",
            "mmmlu_de_de": "deu",
            "mmmlu_es_la": "spa",
            "mmmlu_fr_fr": "fra",
            "mmmlu_hi_in": "hin",
            "mmmlu_id_id": "ind",
            "mmmlu_it_it": "ita",
            "mmmlu_ja_jp": "jpn",
            "mmmlu_ko_kr": "kor",
            "mmmlu_pt_br": "por",
            "mmmlu_sw_ke": "swh",
            "mmmlu_yo_ng": "yor",
            "mmmlu_zh_cn": "cmn",
        }

    if task == "belebele":
        return {
            "belebele_afr_Latn": "afr",
            "belebele_arb_Arab": "arb",
            "belebele_ben_Beng": "ben",
            "belebele_cat_Latn": "cat",
            "belebele_ces_Latn": "ces",
            "belebele_zho_Hans": "cmn",
            "belebele_dan_Latn": "dan",
            "belebele_deu_Latn": "deu",
            "belebele_eus_Latn": "eus",
            "belebele_fra_Latn": "fra",
            "belebele_guj_Gujr": "guj",
            "belebele_hin_Deva": "hin",
            "belebele_hrv_Latn": "hrv",
            "belebele_hun_Latn": "hun",
            "belebele_hye_Armn": "hye",
            "belebele_ind_Latn": "ind",
            "belebele_ita_Latn": "ita",
            "belebele_jpn_Jpan": "jpn",
            "belebele_kan_Knda": "kan",
            "belebele_kor_Hang": "kor",
            "belebele_mal_Mlym": "mal",
            "belebele_mar_Deva": "mar",
            "belebele_nld_Latn": "nld",
            "belebele_npi_Deva": "npi",
            "belebele_por_Latn": "por",
            "belebele_ron_Latn": "ron",
            "belebele_rus_Cyrl": "rus",
            "belebele_slk_Latn": "slk",
            "belebele_spa_Latn": "spa",
            "belebele_swe_Latn": "swe",
            "belebele_swh_Latn": "swh",
            "belebele_tam_Taml": "tam",
            "belebele_tel_Telu": "tel",
            "belebele_tha_Thai": "tha",
            "belebele_ukr_Cyrl": "ukr",
            "belebele_urd_Arab": "urd",
            "belebele_vie_Latn": "vie",
            "belebele_wol_Latn": "wol",
            "belebele_yor_Latn": "yor",
            "belebele_zul_Latn": "zul",
        }

    if task in {"arc_multilingual", "hellaswag_multilingual", "m_mmlu"}:
        lm = {
            "ar": "arb",
            "bn": "ben",
            "ca": "cat",
            "da": "dan",
            "de": "deu",
            "es": "spa",
            "eu": "eus",
            "fr": "fra",
            "gu": "guj",
            "hi": "hin",
            "hr": "hrv",
            "hu": "hun",
            "hy": "hye",
            "id": "ind",
            "it": "ita",
            "kn": "kan",
            "ml": "mal",
            "mr": "mar",
            "ne": "npi",
            "nl": "nld",
            "pt": "por",
            "ro": "ron",
            "ru": "rus",
            "sk": "slk",
            "sr": "srp",
            "sv": "swe",
            "ta": "tam",
            "te": "tel",
            "uk": "ukr",
            "vi": "vie",
            "zh": "cmn",
        }

        if task == "arc_multilingual":
            return {f"arc_{k}": v for k, v in lm.items()}

        if task == "hellaswag_multilingual":
            # The evaluated HellaSwag task set does not contain Chinese.
            lm = {k: v for k, v in lm.items() if k != "zh"}
            return {f"hellaswag_{k}": v for k, v in lm.items()}

        return {f"m_mmlu_{k}": v for k, v in lm.items()}

    raise ValueError(f"Unsupported benchmark: {task}")


def get_sub_task_metric(task):
    """Return benchmark subtasks and the metric field used in the experiments."""
    if task == "mmmlu":
        sub_tasks = [
            "mmmlu_ar_xy",
            "mmmlu_bn_bd",
            "mmmlu_de_de",
            "mmmlu_es_la",
            "mmmlu_fr_fr",
            "mmmlu_hi_in",
            "mmmlu_id_id",
            "mmmlu_it_it",
            "mmmlu_ja_jp",
            "mmmlu_ko_kr",
            "mmmlu_pt_br",
            "mmmlu_sw_ke",
            "mmmlu_yo_ng",
            "mmmlu_zh_cn",
        ]
        return sub_tasks, "acc_norm,none"

    if task == "belebele":
        return list(get_lang_map(task).keys()), "acc_norm,none"

    if task == "arc_multilingual":
        return list(get_lang_map(task).keys()), "acc_norm,none"

    if task == "hellaswag_multilingual":
        return list(get_lang_map(task).keys()), "acc_norm,none"

    if task == "m_mmlu":
        return list(get_lang_map(task).keys()), "acc,none"

    raise ValueError(f"Unsupported benchmark: {task}")


def build_score_note(scoring_cfg, subgroup=None):
    """Build the suffix used by calculate_intrinsic_scores.py."""
    note = f"_{scoring_cfg['base_language']}"

    if subgroup is not None:
        note += f"_non_parallel_{int(subgroup)}"

    pca_cfg = scoring_cfg.get("pca", {})
    pca_mode = pca_cfg.get("mode", "none")

    if pca_mode == "variance":
        note += f"_pca{pca_cfg['explained_variance']}"
    elif pca_mode == "components":
        note += f"_pca{pca_cfg['n_components']}"
    elif pca_mode != "none":
        raise ValueError(
            "scoring.pca.mode must be one of: none, variance, components."
        )

    return note


def score_file_path(config, model, method, corpus, subgroup=None):
    method_cfg = config["methods"][method]
    scoring_cfg = config["scoring"]
    scores_dir = Path(config["paths"]["scores_dir"])

    filename = (
        f"{model}_maha_{scoring_cfg['variance_mode']}_"
        f"{method_cfg['pooling']}_{method_cfg['scoring']}_"
        f"{corpus}{build_score_note(scoring_cfg, subgroup=subgroup)}.feather"
    )
    return scores_dir / filename


def aggregate_intrinsic_scores(df, method, analysis_cfg):
    """Collapse a layer-wise score table to one intrinsic score per language."""
    if bool(analysis_cfg.get("drop_first_layer", True)):
        df = df.drop(index=0, errors="ignore")

    if len(df) == 0:
        raise ValueError("Intrinsic score table contains no layers after filtering.")

    if method == "meld":
        idx = middle_percent(
            list(range(len(df))),
            float(analysis_cfg["meld_middle_fraction"]),
            min_layers=int(analysis_cfg["meld_min_layers"]),
        )
        return df.apply(lambda col: auc_score(col.values[idx]))

    if method == "mexa":
        return df.apply(lambda col: col.mean())

    if method == "lr":
        idx = layer_steps(
            list(range(len(df))),
            step=int(analysis_cfg["lr_layer_step"]),
        )
        if not idx:
            raise ValueError(
                "No Language Ranker layers were selected. "
                "Reduce analysis.lr_layer_step for this model."
            )
        return df.apply(lambda col: np.mean(col.values[idx]))

    raise ValueError(f"Unsupported method: {method}")


def infer_model_name(path, data, configured_models):
    """Infer the configured model identifier represented by an evaluation JSON."""
    stem = path.stem

    # Preserve compatibility with the earlier explicit output convention:
    # <model>__<task>_...json
    prefix = stem.split("__", 1)[0]
    if prefix in configured_models:
        return prefix

    stem_matches = [model for model in configured_models if model in stem]
    if len(stem_matches) == 1:
        return stem_matches[0]

    # Current lm-evaluation-harness output paths can encode model information in
    # the JSON rather than the filename. Search likely metadata sections.
    metadata_parts = []
    for key in ("model", "model_name", "model_args", "config", "model_configs"):
        if key in data:
            metadata_parts.append(json.dumps(data[key], default=str))

    metadata = " ".join(metadata_parts)
    metadata_matches = [model for model in configured_models if model in metadata]

    if len(metadata_matches) == 1:
        return metadata_matches[0]

    return None


def load_benchmark_index(results_dir, benchmarks, models):
    """Index benchmark JSON files by (benchmark, model).

    If multiple files match the same benchmark/model pair, the most recently
    modified file is used and a warning is emitted.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Benchmark results directory does not exist: {results_dir}"
        )

    candidates = {}

    for path in results_dir.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable JSON %s: %s", path, exc)
            continue

        results = data.get("results")
        if not isinstance(results, dict):
            continue

        model = infer_model_name(path, data, models)
        if model is None:
            continue

        result_keys = set(results)

        for benchmark in benchmarks:
            sub_tasks, _ = get_sub_task_metric(benchmark)
            if set(sub_tasks).issubset(result_keys):
                candidates.setdefault((benchmark, model), []).append(
                    (path, data)
                )

    index = {}
    for key, matches in candidates.items():
        if len(matches) > 1:
            matches.sort(key=lambda item: item[0].stat().st_mtime)
            LOGGER.warning(
                "Multiple benchmark files found for %s/%s; using latest: %s",
                key[0],
                key[1],
                matches[-1][0],
            )
        index[key] = matches[-1]

    return index


def benchmark_language_scores(benchmark, data):
    """Return a Series indexed by ISO 639-3 code for one benchmark result."""
    sub_tasks, metric = get_sub_task_metric(benchmark)
    lang_map = get_lang_map(benchmark)
    results = data["results"]

    scores = {}
    for task_name in sub_tasks:
        scores[lang_map[task_name]] = results[task_name][metric]

    return pd.Series(scores, name="benchmark_score", dtype=float)


def handle_missing(message, allow_missing):
    if allow_missing:
        LOGGER.warning(message)
        return True
    raise FileNotFoundError(message)


def analyze(config, setting, allow_missing=False):
    """Create one observation table for a parallel or non-parallel setting."""
    analysis_cfg = config["analysis"]
    models = list(analysis_cfg["models"])
    methods = list(analysis_cfg["methods"])
    corpora = list(analysis_cfg["corpora"])
    benchmarks = list(analysis_cfg["benchmarks"])

    benchmark_index = load_benchmark_index(
        analysis_cfg["benchmark_results_dir"],
        benchmarks,
        models,
    )

    if setting == "parallel":
        subgroup_values = [None]
    elif setting == "non-parallel":
        subgroup_values = [
            int(x) for x in analysis_cfg["non_parallel_subgroups"]
        ]
    else:
        raise ValueError(f"Unsupported setting: {setting}")

    records = []

    # Cache one-language intrinsic scores because the same scores are reused for
    # all five downstream benchmarks.
    intrinsic_cache = {}

    for subgroup in subgroup_values:
        for method in methods:
            method_cfg = config["methods"][method]

            for corpus in corpora:
                for model in models:
                    cache_key = (subgroup, method, corpus, model)

                    score_path = score_file_path(
                        config,
                        model=model,
                        method=method,
                        corpus=corpus,
                        subgroup=subgroup,
                    )

                    if not score_path.exists():
                        if handle_missing(
                            f"Intrinsic score file not found: {score_path}",
                            allow_missing,
                        ):
                            continue

                    score_df = pd.read_feather(score_path)
                    intrinsic_cache[cache_key] = aggregate_intrinsic_scores(
                        score_df,
                        method,
                        analysis_cfg,
                    )

                    for benchmark in benchmarks:
                        benchmark_key = (benchmark, model)
                        if benchmark_key not in benchmark_index:
                            if handle_missing(
                                "Benchmark result not found for "
                                f"benchmark='{benchmark}', model='{model}'.",
                                allow_missing,
                            ):
                                continue

                        benchmark_path, benchmark_json = benchmark_index[
                            benchmark_key
                        ]
                        benchmark_scores = benchmark_language_scores(
                            benchmark,
                            benchmark_json,
                        )

                        intrinsic_scores = intrinsic_cache[cache_key].rename(
                            "intrinsic_score"
                        )

                        paired = pd.concat(
                            [benchmark_scores, intrinsic_scores],
                            axis=1,
                            join="inner",
                        ).dropna()

                        n_languages = len(paired)
                        if n_languages < 2:
                            pearson = np.nan
                            spearman = np.nan
                        else:
                            pearson = paired["benchmark_score"].corr(
                                paired["intrinsic_score"],
                                method="pearson",
                            )
                            spearman = paired["benchmark_score"].corr(
                                paired["intrinsic_score"],
                                method="spearman",
                            )

                        row = {
                            "corpus": corpus,
                            "method": method,
                            "pooling": method_cfg["pooling"],
                            "scoring": method_cfg["scoring"],
                            "benchmark": benchmark,
                            "model": model,
                            "reference_language": config["scoring"][
                                "base_language"
                            ],
                            "n_languages": n_languages,
                            "pearson": pearson,
                            "spearman": spearman,
                            "intrinsic_file": str(score_path),
                            "benchmark_file": str(benchmark_path),
                        }

                        if subgroup is not None:
                            row["subgroup"] = subgroup

                        records.append(row)

    df = pd.DataFrame(records)

    sort_cols = ["corpus", "method", "benchmark", "model"]
    if setting == "non-parallel":
        sort_cols = ["subgroup"] + sort_cols

    if not df.empty:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def save_table(df, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    feather_path = output_dir / f"{stem}.feather"

    df.to_csv(csv_path, index=False)
    df.to_feather(feather_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {feather_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = config["analysis"]["output_dir"]

    if args.setting in {"parallel", "all"}:
        parallel_df = analyze(
            config,
            setting="parallel",
            allow_missing=args.allow_missing,
        )
        save_table(
            parallel_df,
            output_dir,
            stem="parallel_correlations",
        )

    if args.setting in {"non-parallel", "all"}:
        non_parallel_df = analyze(
            config,
            setting="non-parallel",
            allow_missing=args.allow_missing,
        )
        save_table(
            non_parallel_df,
            output_dir,
            stem="non_parallel_correlations",
        )


if __name__ == "__main__":
    main()
