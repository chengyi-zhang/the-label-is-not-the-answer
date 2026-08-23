from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LABELS = "ABCDE"
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260813


def chi_square_statistic(table: np.ndarray) -> float:
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(np.sum((table - expected) ** 2 / expected))


def load_predictions(root: Path) -> tuple[list[str], np.ndarray, dict[str, dict[str, dict]]]:
    runs: dict[str, dict[str, dict]] = {}
    for cue in LABELS:
        payload = json.loads((root / f"anchor_{cue}" / "results.json").read_text(encoding="utf-8"))
        runs[cue] = {row["case_id"]: row for row in payload["answers"]}

    case_ids = sorted(set.intersection(*(set(run) for run in runs.values())))
    predictions = np.empty((len(case_ids), len(LABELS)), dtype=np.int64)
    for case_index, case_id in enumerate(case_ids):
        for cue_index, cue in enumerate(LABELS):
            predictions[case_index, cue_index] = LABELS.index(runs[cue][case_id]["answer"])
    return case_ids, predictions, runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-example-case",
        metavar="CASE_ID",
        help="Optionally include one case-level example in the report. Omit for public releases.",
    )
    args = parser.parse_args()

    case_ids, predictions, runs = load_predictions(args.root)
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

    unique_counts = np.array([len(set(row.tolist())) for row in predictions])
    report = {
        "cases": len(case_ids),
        "contingency_table": observed.tolist(),
        "chi_square": observed_statistic,
        "paired_casewise_permutation": {
            "seed": RANDOM_SEED,
            "permutations": N_PERMUTATIONS,
            "exceedances": exceedances,
            "p_value": (exceedances + 1) / (N_PERMUTATIONS + 1),
        },
        "unique_reported_labels_per_case": {
            "mean": float(unique_counts.mean()),
            "distribution": {str(k): v for k, v in sorted(Counter(unique_counts.tolist()).items())},
        },
    }
    if args.include_example_case:
        case_id = args.include_example_case
        if case_id not in runs[LABELS[0]]:
            raise KeyError(f"unknown case ID: {case_id}")
        report["illustrative_case"] = {
            "case_id": case_id,
            "outputs": {
                cue: {
                    "answer": runs[cue][case_id]["answer"],
                    "reasoning_trace": runs[cue][case_id]["reasoning_trace"],
                }
                for cue in LABELS
            },
        }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
