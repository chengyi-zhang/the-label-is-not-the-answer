#!/usr/bin/env python3
import collections
import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
SPLIT = ROOT / "data/experiments/eval_600"
OUTPUTS = ROOT / "outputs/eval_600"
ANALYSIS = ROOT / "analysis/eval_600"
ANALYSIS.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("submitted", "Submitted"),
    ("factor_anchor_clean", "Anchored + clean options"),
    ("factor_no_anchor_raw", "No anchor + raw options"),
    ("factor_no_anchor_clean", "No anchor + clean options"),
    ("comparative", "Comparative"),
    ("cue_free", "Cue-free"),
    ("permutation_ensemble", "5-permutation ensemble"),
]
LABELS = tuple("ABCDE")


def load_predictions(variant):
    payload = json.loads((OUTPUTS / variant / "results.json").read_text(encoding="utf-8"))
    return {str(row["case_id"]): str(row["answer"]).strip().upper() for row in payload["answers"]}


def wilson_interval(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin


def exact_mcnemar_p(improved, worsened):
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(improved, worsened) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bias_metrics(predictions):
    counts = collections.Counter(predictions.values())
    n = sum(counts.values())
    probabilities = [counts[label] / n for label in LABELS]
    tv_uniform = 0.5 * sum(abs(value - 0.2) for value in probabilities)
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0) / math.log(len(LABELS))
    chi_square = sum((counts[label] - n / 5) ** 2 / (n / 5) for label in LABELS)
    return {
        "answer_counts": {label: counts[label] for label in LABELS},
        "tv_distance_from_uniform": tv_uniform,
        "normalized_answer_entropy": entropy,
        "chi_square_vs_uniform": chi_square,
    }


gold = json.loads((SPLIT / "gold.json").read_text(encoding="utf-8"))
metadata = json.loads((SPLIT / "metadata.json").read_text(encoding="utf-8"))
predictions = {variant: load_predictions(variant) for variant, _ in VARIANTS}

summary = {}
for variant, display_name in VARIANTS:
    pred = predictions[variant]
    correct = sum(pred[case_id] == answer for case_id, answer in gold.items())
    ci_low, ci_high = wilson_interval(correct, len(gold))
    summary[variant] = {
        "display_name": display_name,
        "n": len(gold),
        "correct": correct,
        "accuracy_percent": 100 * correct / len(gold),
        "ci95_low_percent": 100 * ci_low,
        "ci95_high_percent": 100 * ci_high,
        **bias_metrics(pred),
    }

comparisons = {}
comparison_pairs = [
    ("submitted", "factor_anchor_clean"),
    ("submitted", "factor_no_anchor_raw"),
    ("factor_no_anchor_raw", "factor_no_anchor_clean"),
    ("factor_anchor_clean", "factor_no_anchor_clean"),
    ("factor_no_anchor_clean", "cue_free"),
    ("submitted", "comparative"),
    ("submitted", "cue_free"),
    ("comparative", "cue_free"),
    ("cue_free", "permutation_ensemble"),
]
for left, right in comparison_pairs:
    improved = sum(
        predictions[left][case_id] != gold[case_id] and predictions[right][case_id] == gold[case_id]
        for case_id in gold
    )
    worsened = sum(
        predictions[left][case_id] == gold[case_id] and predictions[right][case_id] != gold[case_id]
        for case_id in gold
    )
    changed = sum(predictions[left][case_id] != predictions[right][case_id] for case_id in gold)
    comparisons[f"{left}__to__{right}"] = {
        "left": left,
        "right": right,
        "changed_predictions": changed,
        "improved_cases": improved,
        "worsened_cases": worsened,
        "net_correct_change": improved - worsened,
        "accuracy_delta_points": summary[right]["accuracy_percent"] - summary[left]["accuracy_percent"],
        "exact_mcnemar_p": exact_mcnemar_p(improved, worsened),
    }

subgroups = {}
for field in ("task", "modality", "organ_system"):
    values = sorted({row[field] for row in metadata.values()})
    subgroups[field] = {}
    for value in values:
        ids = [case_id for case_id in gold if metadata[case_id][field] == value]
        row = {"n": len(ids)}
        for variant, _ in VARIANTS:
            correct = sum(predictions[variant][case_id] == gold[case_id] for case_id in ids)
            row[variant] = 100 * correct / len(ids)
        subgroups[field][value] = row

report = {"summary": summary, "comparisons": comparisons, "subgroups": subgroups}
(ANALYSIS / "statistical_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
)

with (ANALYSIS / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["variant", "accuracy_percent", "ci95_low", "ci95_high", "correct", "tv_uniform", "answer_entropy", *LABELS]
    )
    for variant, _ in VARIANTS:
        row = summary[variant]
        writer.writerow(
            [
                variant,
                f"{row['accuracy_percent']:.3f}",
                f"{row['ci95_low_percent']:.3f}",
                f"{row['ci95_high_percent']:.3f}",
                row["correct"],
                f"{row['tv_distance_from_uniform']:.6f}",
                f"{row['normalized_answer_entropy']:.6f}",
                *(row["answer_counts"][label] for label in LABELS),
            ]
        )

lines = [
    "# MedReason Eval-600 Statistical Report",
    "",
    "The evaluation set contains 600 MCQ cases, balanced at 120 gold answers per label (A-E).",
    "",
    "| Variant | Correct | Accuracy | 95% CI | TV bias | Answer entropy | A/B/C/D/E |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for variant, display_name in VARIANTS:
    row = summary[variant]
    counts = "/".join(str(row["answer_counts"][label]) for label in LABELS)
    lines.append(
        f"| {display_name} | {row['correct']}/600 | {row['accuracy_percent']:.2f}% | "
        f"{row['ci95_low_percent']:.2f}-{row['ci95_high_percent']:.2f}% | "
        f"{row['tv_distance_from_uniform']:.3f} | {row['normalized_answer_entropy']:.3f} | {counts} |"
    )

lines.extend(["", "## Paired comparisons", "", "| Comparison | Improved | Worsened | Delta | Exact McNemar p |", "|---|---:|---:|---:|---:|"])
for comparison in comparisons.values():
    lines.append(
        f"| {comparison['left']} -> {comparison['right']} | {comparison['improved_cases']} | "
        f"{comparison['worsened_cases']} | {comparison['accuracy_delta_points']:+.2f} pp | "
        f"{comparison['exact_mcnemar_p']:.6g} |"
    )

(ANALYSIS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
