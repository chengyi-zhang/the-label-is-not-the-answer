#!/usr/bin/env python3
"""Independent integrity and paired-statistics audit for the InternVL3 experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.stats import binomtest


LABELS = "ABCDE"
ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_DIR = ROOT / "outputs/external_validation_internvl3_9b_strict"
ANALYSIS_DIR = ROOT / "analysis/external_validation_internvl3_9b_strict"
MANIFEST = ROOT / "data/experiments/counterfactual_200/manifest.json"
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260812


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_response(text: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return json.loads(stripped)


def chi_square_statistic(table: np.ndarray) -> float:
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(np.sum((table - expected) ** 2 / expected))


def main() -> None:
    selected_ids = load_json(MANIFEST)["selected_ids"][:200]
    assert len(selected_ids) == 200 and len(set(selected_ids)) == 200

    predictions = np.empty((len(selected_ids), len(LABELS)), dtype=np.int8)
    files = {}
    parse_modes = Counter()
    elapsed = []
    response_lengths = []
    tile_counts = Counter()

    for cue_index, anchor in enumerate(LABELS):
        raw_path = OUTPUT_DIR / f"anchor_{anchor}" / "raw.jsonl"
        result_path = OUTPUT_DIR / f"anchor_{anchor}" / "results.json"
        raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        result_rows = load_json(result_path)["answers"]

        assert len(raw_rows) == len(result_rows) == 200
        assert [row["case_id"] for row in raw_rows] == selected_ids
        assert [row["case_id"] for row in result_rows] == selected_ids
        assert len({row["case_id"] for row in raw_rows}) == 200

        for case_index, (raw_row, result_row) in enumerate(zip(raw_rows, result_rows)):
            assert raw_row["anchor"] == anchor
            assert raw_row["answer"] in LABELS
            assert result_row["answer"] == raw_row["answer"]
            parsed = parse_response(raw_row["raw_response"])
            assert parsed["answer"] == raw_row["answer"]
            assert isinstance(parsed.get("reasoning_trace"), str) and parsed["reasoning_trace"].strip()
            parse_modes[raw_row["parse_mode"]] += 1
            elapsed.append(float(raw_row["elapsed_seconds"]))
            response_lengths.append(len(raw_row["raw_response"]))
            tile_counts[int(raw_row["image_tiles"])] += 1
            predictions[case_index, cue_index] = LABELS.index(raw_row["answer"])

        files[str(raw_path.relative_to(ROOT))] = sha256(raw_path)
        files[str(result_path.relative_to(ROOT))] = sha256(result_path)

    assert parse_modes == {"json_direct": 1000}

    observed = np.array(
        [[np.sum(predictions[:, cue] == response) for response in range(5)] for cue in range(5)],
        dtype=np.int64,
    )
    observed_statistic = chi_square_statistic(observed)

    rng = np.random.default_rng(RANDOM_SEED)
    exceedances = 0
    for _ in range(N_PERMUTATIONS):
        order = np.argsort(rng.random(predictions.shape), axis=1)
        permuted = np.take_along_axis(predictions, order, axis=1)
        table = np.array(
            [[np.sum(permuted[:, cue] == response) for response in range(5)] for cue in range(5)],
            dtype=np.int64,
        )
        exceedances += chi_square_statistic(table) >= observed_statistic - 1e-12
    paired_permutation_p = (exceedances + 1) / (N_PERMUTATIONS + 1)

    mp.mp.dps = 100
    df = 16
    asymptotic_p = mp.gammainc(mp.mpf(df) / 2, mp.mpf(observed_statistic) / 2, mp.inf) / mp.gamma(
        mp.mpf(df) / 2
    )

    follow_counts = [int(observed[index, index]) for index in range(5)]
    binomial_p = [float(binomtest(count, 200, 0.2, alternative="greater").pvalue) for count in follow_counts]
    unique_per_case = np.array([len(set(row.tolist())) for row in predictions])

    audit = {
        "status": "PASS",
        "model": "OpenGVLab/InternVL3-9B",
        "prespecified_cases": 200,
        "total_records": 1000,
        "parse_modes": dict(parse_modes),
        "contingency_cue_by_response": observed.tolist(),
        "observed_chi_square": observed_statistic,
        "asymptotic_p_high_precision": mp.nstr(asymptotic_p, 12),
        "paired_casewise_permutation": {
            "seed": RANDOM_SEED,
            "permutations": N_PERMUTATIONS,
            "exceedances": exceedances,
            "p_value": paired_permutation_p,
        },
        "follow_counts": dict(zip(LABELS, follow_counts)),
        "one_sided_binomial_p_vs_20_percent": dict(zip(LABELS, binomial_p)),
        "cases_changed_by_cue": int(np.sum(unique_per_case > 1)),
        "cases_with_all_five_labels": int(np.sum(unique_per_case == 5)),
        "unique_prediction_count_distribution": {
            str(value): int(np.sum(unique_per_case == value)) for value in range(1, 6)
        },
        "mean_seconds_per_inference": float(np.mean(elapsed)),
        "median_seconds_per_inference": float(np.median(elapsed)),
        "response_length_characters": {
            "min": int(np.min(response_lengths)),
            "median": float(np.median(response_lengths)),
            "max": int(np.max(response_lengths)),
        },
        "image_tile_counts": {str(key): value for key, value in sorted(tile_counts.items())},
        "file_sha256": files,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# InternVL3-9B External-Validation Audit",
        "",
        "**Status: PASS**",
        "",
        "- Prespecified IDs/order/uniqueness: PASS (200 cases per cue).",
        "- Raw/results answer agreement: PASS (1000/1000).",
        "- Strict raw-response JSON agreement: PASS (1000/1000).",
        "- Parse mode: `json_direct` for 1000/1000; no fallback.",
        f"- Observed chi-square: {observed_statistic:.3f}; high-precision asymptotic p = {mp.nstr(asymptotic_p, 6)}.",
        f"- Case-wise paired permutation: {exceedances}/{N_PERMUTATIONS} null statistics as large; p = {paired_permutation_p:.6f}.",
        f"- Cases changed by cue: {audit['cases_changed_by_cue']}/200; all five labels observed: {audit['cases_with_all_five_labels']}/200.",
        f"- Mean/median inference time: {audit['mean_seconds_per_inference']:.2f}/{audit['median_seconds_per_inference']:.2f} s.",
        "- SHA-256 hashes for every raw and result file are recorded in `audit.json`.",
    ]
    (ANALYSIS_DIR / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
