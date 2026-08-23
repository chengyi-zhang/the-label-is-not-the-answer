#!/usr/bin/env python3
"""Validate and summarize the InternVL3-9B cross-family anchor experiment."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

from scipy.stats import chi2_contingency


LABELS = "ABCDE"
ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_DIR = ROOT / "outputs/external_validation_internvl3_9b_strict"
ANALYSIS_DIR = ROOT / "analysis/external_validation_internvl3_9b_strict"
MANIFEST = ROOT / "data/experiments/counterfactual_200/manifest.json"
GOLD_PATH = ROOT / "data/experiments/eval_600/gold.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - half), 100 * (center + half)]


def main() -> None:
    selected_ids = load(MANIFEST)["selected_ids"][:200]
    if len(selected_ids) != 200 or len(set(selected_ids)) != 200:
        raise ValueError("Expected 200 unique prespecified case IDs")
    gold = load(GOLD_PATH)
    predictions = {}
    report = {
        "model": "OpenGVLab/InternVL3-9B",
        "language_backbone": "InternLM3-8B-Instruct",
        "prespecified_cases": len(selected_ids),
        "total_inferences": len(selected_ids) * len(LABELS),
        "anchors": {},
    }
    contingency = []

    for anchor in LABELS:
        payload = load(OUTPUT_DIR / f"anchor_{anchor}" / "results.json")
        rows = payload["answers"]
        ids = [row["case_id"] for row in rows]
        if ids != selected_ids:
            raise ValueError(f"Anchor {anchor} case IDs or order do not match the prespecified list")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Anchor {anchor} contains duplicate case IDs")
        answers = {row["case_id"]: row["answer"] for row in rows}
        if any(answer not in LABELS for answer in answers.values()):
            raise ValueError(f"Anchor {anchor} contains a noncanonical answer")
        predictions[anchor] = answers
        counts = Counter(answers.values())
        contingency.append([counts[label] for label in LABELS])
        correct = sum(answers[case_id] == gold[case_id] for case_id in selected_ids)
        follow = counts[anchor]
        parse_modes = Counter(row.get("parse_mode", "missing") for row in rows)
        elapsed = [float(row.get("elapsed_seconds", 0)) for row in rows]
        report["anchors"][anchor] = {
            "counts": {label: counts[label] for label in LABELS},
            "accuracy_percent": 100 * correct / len(selected_ids),
            "accuracy_wilson_95": wilson(correct, len(selected_ids)),
            "anchor_follow_count": follow,
            "anchor_follow_percent": 100 * follow / len(selected_ids),
            "anchor_follow_wilson_95": wilson(follow, len(selected_ids)),
            "parse_modes": dict(parse_modes),
            "fallback_count": parse_modes["unparsed"],
            "mean_seconds_per_case": sum(elapsed) / len(elapsed),
        }

    statistic, p_value, degrees, expected = chi2_contingency(contingency)
    n = sum(sum(row) for row in contingency)
    cramers_v = math.sqrt(statistic / (n * min(len(contingency) - 1, len(contingency[0]) - 1)))
    unique_counts = [
        len({predictions[anchor][case_id] for anchor in LABELS}) for case_id in selected_ids
    ]
    report["aggregate"] = {
        "contingency_cue_by_response": contingency,
        "chi_square": statistic,
        "degrees_of_freedom": degrees,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "cases_changed_by_anchor": sum(value > 1 for value in unique_counts),
        "cases_unchanged_across_all_anchors": sum(value == 1 for value in unique_counts),
        "mean_unique_predictions_per_case": sum(unique_counts) / len(unique_counts),
        "minimum_expected_cell_count": float(expected.min()),
        "total_parser_fallbacks": sum(
            report["anchors"][anchor]["fallback_count"] for anchor in LABELS
        ),
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# InternVL3-9B Cross-Family Anchor Validation",
        "",
        "Prespecified sample: 200 MedReason MCQ cases; five schema-example labels per case.",
        "",
        "| Cue | Accuracy (95% CI) | Follow-cue rate (95% CI) | A/B/C/D/E | Parser fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    for anchor in LABELS:
        row = report["anchors"][anchor]
        acc_ci = row["accuracy_wilson_95"]
        follow_ci = row["anchor_follow_wilson_95"]
        counts = "/".join(str(row["counts"][label]) for label in LABELS)
        lines.append(
            f"| {anchor} | {row['accuracy_percent']:.2f}% "
            f"[{acc_ci[0]:.2f}, {acc_ci[1]:.2f}] | "
            f"{row['anchor_follow_percent']:.2f}% "
            f"[{follow_ci[0]:.2f}, {follow_ci[1]:.2f}] | {counts} | "
            f"{row['fallback_count']} |"
        )
    aggregate = report["aggregate"]
    lines += [
        "",
        f"- Chi-square({aggregate['degrees_of_freedom']}) = {aggregate['chi_square']:.3f}, "
        f"p = {aggregate['p_value']:.6g}; Cramer's V = {aggregate['cramers_v']:.3f}.",
        f"- Cases changed by at least one cue: {aggregate['cases_changed_by_anchor']}/200.",
        f"- Cases unchanged across all cues: {aggregate['cases_unchanged_across_all_anchors']}/200.",
        f"- Total parser fallbacks: {aggregate['total_parser_fallbacks']}/1000.",
        f"- Minimum expected contingency-cell count: {aggregate['minimum_expected_cell_count']:.2f}.",
    ]
    (ANALYSIS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
