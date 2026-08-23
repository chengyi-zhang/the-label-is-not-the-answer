#!/usr/bin/env python3
import collections
import json
import math
import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
ANALYSIS = ROOT / "analysis/validation_extensions"
ANALYSIS.mkdir(parents=True, exist_ok=True)
LABELS = tuple("ABCDE")

STUDIES = {
    "3b_original": {
        "split": ROOT / "data/experiments/eval_600",
        "variants": {
            "submitted": ROOT / "outputs/eval_600/submitted/results.json",
            "comparative": ROOT / "outputs/eval_600/comparative/results.json",
            "cue_free": ROOT / "outputs/eval_600/cue_free/results.json",
        },
    },
    "3b_replication": {
        "split": ROOT / "data/experiments/eval_600_replication",
        "variants": {
            "submitted": ROOT / "outputs/eval_600_replication_3b/submitted/results.json",
            "comparative": ROOT / "outputs/eval_600_replication_3b/comparative/results.json",
            "cue_free": ROOT / "outputs/eval_600_replication_3b/cue_free/results.json",
        },
    },
    "7b_original": {
        "split": ROOT / "data/experiments/eval_600",
        "variants": {
            "submitted": ROOT / "outputs/eval_600_7b/submitted/results.json",
            "comparative": ROOT / "outputs/eval_600_7b/comparative/results.json",
            "cue_free": ROOT / "outputs/eval_600_7b/cue_free/results.json",
        },
    },
}


def load_predictions(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_id"]): str(row["answer"]).strip().upper() for row in payload["answers"]}


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin


def exact_mcnemar(improved, worsened):
    n = improved + worsened
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(improved, worsened) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize(gold, predictions):
    correct = sum(predictions[case_id] == answer for case_id, answer in gold.items())
    low, high = wilson(correct, len(gold))
    counts = collections.Counter(predictions.values())
    return {
        "n": len(gold),
        "correct": correct,
        "accuracy_percent": 100 * correct / len(gold),
        "ci95_percent": [100 * low, 100 * high],
        "answer_counts": {label: counts[label] for label in LABELS},
    }


def compare(gold, left, right):
    improved = sum(left[c] != gold[c] and right[c] == gold[c] for c in gold)
    worsened = sum(left[c] == gold[c] and right[c] != gold[c] for c in gold)
    return {
        "improved": improved,
        "worsened": worsened,
        "delta_points": 100 * (improved - worsened) / len(gold),
        "exact_mcnemar_p": exact_mcnemar(improved, worsened),
    }


report = {"studies": {}}
loaded = {}
for study_name, config in STUDIES.items():
    gold = json.loads((config["split"] / "gold.json").read_text(encoding="utf-8"))
    predictions = {name: load_predictions(path) for name, path in config["variants"].items()}
    if any(set(pred) != set(gold) for pred in predictions.values()):
        raise ValueError(f"ID mismatch in {study_name}")
    report["studies"][study_name] = {
        "summary": {name: summarize(gold, pred) for name, pred in predictions.items()},
        "comparisons": {
            "submitted_to_comparative": compare(gold, predictions["submitted"], predictions["comparative"]),
            "comparative_to_cue_free": compare(gold, predictions["comparative"], predictions["cue_free"]),
            "submitted_to_cue_free": compare(gold, predictions["submitted"], predictions["cue_free"]),
        },
    }
    loaded[study_name] = (gold, predictions)

combined_gold = {}
combined_predictions = {"submitted": {}, "comparative": {}, "cue_free": {}}
for study_name in ("3b_original", "3b_replication"):
    gold, predictions = loaded[study_name]
    overlap = set(combined_gold) & set(gold)
    if overlap:
        raise ValueError(f"Unexpected overlap across 3B splits: {len(overlap)}")
    combined_gold.update(gold)
    for variant in combined_predictions:
        combined_predictions[variant].update(predictions[variant])

report["combined_3b_1200"] = {
    "summary": {name: summarize(combined_gold, pred) for name, pred in combined_predictions.items()},
    "comparisons": {
        "submitted_to_comparative": compare(
            combined_gold, combined_predictions["submitted"], combined_predictions["comparative"]
        ),
        "comparative_to_cue_free": compare(
            combined_gold, combined_predictions["comparative"], combined_predictions["cue_free"]
        ),
        "submitted_to_cue_free": compare(
            combined_gold, combined_predictions["submitted"], combined_predictions["cue_free"]
        ),
    },
}

(ANALYSIS / "validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
)

lines = [
    "# MedReason Validation Extensions",
    "",
    "| Study | Variant | Correct | Accuracy | 95% CI | A/B/C/D/E |",
    "|---|---|---:|---:|---:|---:|",
]
for study_name, study in report["studies"].items():
    for variant, row in study["summary"].items():
        counts = "/".join(str(row["answer_counts"][label]) for label in LABELS)
        lines.append(
            f"| {study_name} | {variant} | {row['correct']}/{row['n']} | {row['accuracy_percent']:.2f}% | "
            f"{row['ci95_percent'][0]:.2f}-{row['ci95_percent'][1]:.2f}% | {counts} |"
        )

lines.extend(["", "## Paired comparisons", ""])
for study_name, study in report["studies"].items():
    for name, row in study["comparisons"].items():
        lines.append(
            f"- {study_name} {name}: {row['delta_points']:+.2f} pp; "
            f"improved={row['improved']}, worsened={row['worsened']}, p={row['exact_mcnemar_p']:.6g}."
        )

combined = report["combined_3b_1200"]
lines.extend(["", "## Combined 3B evidence (1,200 non-overlapping cases)", ""])
for variant, row in combined["summary"].items():
    lines.append(f"- {variant}: {row['correct']}/{row['n']} ({row['accuracy_percent']:.2f}%).")
for name, row in combined["comparisons"].items():
    lines.append(f"- {name}: {row['delta_points']:+.2f} pp, p={row['exact_mcnemar_p']:.6g}.")

(ANALYSIS / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
