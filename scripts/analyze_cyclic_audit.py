#!/usr/bin/env python3
"""Comprehensive non-plotting analysis for all MedReason prompt studies."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "analysis/overnight_extensions"
LABELS = "ABCDE"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def predictions(path: Path):
    rows = load(path)["answers"]
    result = {row["case_id"]: row["answer"] for row in rows}
    assert len(result) == len(rows)
    assert all(value in LABELS for value in result.values())
    return result


def wilson(correct: int, n: int, z: float = 1.959963984540054):
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - half), 100 * (center + half)]


def score(gold, pred, ids):
    correct = sum(pred[cid] == gold[cid] for cid in ids)
    return {
        "n": len(ids),
        "correct": correct,
        "accuracy": 100 * correct / len(ids),
        "wilson_95": wilson(correct, len(ids)),
    }


def exact_mcnemar(gold, first, second, ids):
    improved = sum(first[cid] != gold[cid] and second[cid] == gold[cid] for cid in ids)
    worsened = sum(first[cid] == gold[cid] and second[cid] != gold[cid] for cid in ids)
    n = improved + worsened
    if n == 0:
        p = 1.0
    else:
        k = min(improved, worsened)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"improved": improved, "worsened": worsened, "p": p}


def distribution_stats(pred, ids):
    counts = Counter(pred[cid] for cid in ids)
    probs = [counts[label] / len(ids) for label in LABELS]
    entropy = -sum(p * math.log(p) for p in probs if p > 0) / math.log(5)
    return {
        "counts": {label: counts[label] for label in LABELS},
        "max_label_share": 100 * max(probs),
        "normalized_entropy": entropy,
    }


def chi_square_independence(table):
    try:
        from scipy.stats import chi2_contingency
        statistic, p, dof, _ = chi2_contingency(table)
        n = sum(sum(row) for row in table)
        r, c = len(table), len(table[0])
        v = math.sqrt(statistic / (n * min(r - 1, c - 1)))
        return {"chi2": statistic, "p": p, "dof": dof, "cramers_v": v}
    except Exception as error:
        return {"error": repr(error)}


def counterfactual_analysis(split_name: str, output_name: str, original_200_ids):
    split = ROOT / f"data/experiments/{split_name}"
    gold = load(split / "gold.json")
    manifest = load(split / "manifest.json")
    records = {row["case_id"]: row for row in manifest["records"]}
    ids = list(gold)
    variants = ("submitted", "comparative", "cue_free")
    preds = {
        variant: predictions(ROOT / f"outputs/{output_name}/{variant}/results.json")
        for variant in variants
    }
    for pred in preds.values():
        assert set(pred) == set(ids)

    subset_ids = {
        "full_600": ids,
        "original_200": [cid for cid in ids if records[cid]["original_case_id"] in original_200_ids],
        "additional_400": [cid for cid in ids if records[cid]["original_case_id"] not in original_200_ids],
    }
    result = {"subsets": {}, "comparisons": {}, "qualitative_case_ids": {}}
    for subset_name, current_ids in subset_ids.items():
        result["subsets"][subset_name] = {}
        for variant in variants:
            pred = preds[variant]
            by_original = defaultdict(dict)
            for cid in current_ids:
                row = records[cid]
                by_original[row["original_case_id"]][row["shift"]] = row["label_to_original"][pred[cid]]
            consistent = sum(len(set(values.values())) == 1 for values in by_original.values())
            pair_matches = 0
            pair_total = 0
            cases_with_any_flip = 0
            for values in by_original.values():
                ordered = [values[shift] for shift in sorted(values)]
                cases_with_any_flip += len(set(ordered)) > 1
                for left, right in combinations(ordered, 2):
                    pair_matches += left == right
                    pair_total += 1
            per_shift = {}
            for shift in range(5):
                shifted_ids = [cid for cid in current_ids if records[cid]["shift"] == shift]
                per_shift[str(shift)] = {
                    **score(gold, pred, shifted_ids),
                    "distribution": distribution_stats(pred, shifted_ids),
                }
            result["subsets"][subset_name][variant] = {
                "performance": score(gold, pred, current_ids),
                "distribution": distribution_stats(pred, current_ids),
                "all_five_semantic_invariance": 100 * consistent / len(by_original),
                "mean_pairwise_semantic_agreement": 100 * pair_matches / pair_total,
                "cases_with_any_semantic_flip": cases_with_any_flip,
                "original_case_count": len(by_original),
                "by_shift": per_shift,
            }
        for first, second in (
            ("submitted", "comparative"),
            ("comparative", "cue_free"),
            ("submitted", "cue_free"),
        ):
            result["comparisons"][f"{subset_name}:{first}_to_{second}"] = exact_mcnemar(
                gold, preds[first], preds[second], current_ids
            )

    shift_zero = [cid for cid in ids if records[cid]["shift"] == 0]
    submitted, cue_free = preds["submitted"], preds["cue_free"]
    result["qualitative_case_ids"] = {
        "cue_free_corrects_submitted": [
            cid for cid in shift_zero if submitted[cid] != gold[cid] and cue_free[cid] == gold[cid]
        ][:25],
        "cue_free_worsens_submitted": [
            cid for cid in shift_zero if submitted[cid] == gold[cid] and cue_free[cid] != gold[cid]
        ][:25],
        "both_wrong": [
            cid for cid in shift_zero if submitted[cid] != gold[cid] and cue_free[cid] != gold[cid]
        ][:25],
    }
    return result


def main():
    report = {"anchor_control_600": {}, "counterfactual_600": {}, "previous_experiments": {}}
    split = ROOT / "data/experiments/eval_600"
    gold = load(split / "gold.json")
    ids = list(gold)

    anchor_preds = {}
    contingency = []
    for anchor in LABELS:
        pred = predictions(ROOT / f"outputs/anchor_control_600/anchor_{anchor}/results.json")
        assert set(pred) == set(ids)
        anchor_preds[anchor] = pred
        counts = Counter(pred.values())
        contingency.append([counts[label] for label in LABELS])
        report["anchor_control_600"][anchor] = {
            "performance": score(gold, pred, ids),
            "distribution": distribution_stats(pred, ids),
            "anchor_follow_count": counts[anchor],
            "anchor_follow_rate": 100 * counts[anchor] / len(ids),
        }

    unique_counts = [len({anchor_preds[a][cid] for a in LABELS}) for cid in ids]
    report["anchor_control_600"]["aggregate"] = {
        "contingency_anchor_by_prediction": contingency,
        "independence_test": chi_square_independence(contingency),
        "cases_changed_by_anchor": sum(value > 1 for value in unique_counts),
        "cases_unchanged_across_all_anchors": sum(value == 1 for value in unique_counts),
        "mean_unique_predictions_per_case": sum(unique_counts) / len(unique_counts),
    }

    original_200 = set(load(ROOT / "data/experiments/counterfactual_200/manifest.json")["selected_ids"])
    report["counterfactual_600"] = counterfactual_analysis(
        "counterfactual_600", "counterfactual_600", original_200
    )

    previous = ROOT / "analysis/strengthening_experiments/report.json"
    if previous.exists():
        report["previous_experiments"] = load(previous)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MedReason Overnight Extension Analysis",
        "",
        "## Anchor-letter intervention (600 cases per anchor)",
        "",
        "| Prompt example | Accuracy (95% Wilson CI) | Follow-anchor rate | A/B/C/D/E | Entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for anchor in LABELS:
        row = report["anchor_control_600"][anchor]
        perf, dist = row["performance"], row["distribution"]
        ci = perf["wilson_95"]
        counts = "/".join(str(dist["counts"][label]) for label in LABELS)
        lines.append(
            f"| {anchor} | {perf['accuracy']:.2f}% [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{row['anchor_follow_rate']:.2f}% | {counts} | {dist['normalized_entropy']:.3f} |"
        )
    aggregate = report["anchor_control_600"]["aggregate"]
    lines += [
        "",
        f"- Cases whose prediction changed under at least one anchor: {aggregate['cases_changed_by_anchor']}/600.",
        f"- Cases unchanged across all five anchors: {aggregate['cases_unchanged_across_all_anchors']}/600.",
        f"- Anchor/prediction independence test: {aggregate['independence_test']}.",
        "",
        "## Full cyclic relabeling (600 cases x 5 mappings)",
        "",
        "| Subset | Method | Accuracy | All-five invariance | Pairwise agreement | Max label share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for subset_name in ("full_600", "original_200", "additional_400"):
        for variant in ("submitted", "comparative", "cue_free"):
            row = report["counterfactual_600"]["subsets"][subset_name][variant]
            lines.append(
                f"| {subset_name} | {variant} | {row['performance']['accuracy']:.2f}% | "
                f"{row['all_five_semantic_invariance']:.2f}% | "
                f"{row['mean_pairwise_semantic_agreement']:.2f}% | "
                f"{row['distribution']['max_label_share']:.2f}% |"
            )
    lines += ["", "## Paired comparisons", ""]
    for name, row in report["counterfactual_600"]["comparisons"].items():
        lines.append(f"- {name}: improved={row['improved']}, worsened={row['worsened']}, p={row['p']:.6g}.")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Treat the anchor-letter intervention as causal evidence only if output distributions move with the manipulated example label.",
        "- Distinguish reduced positional bias from true semantic permutation invariance; they are not equivalent.",
        "- Report the additional-400 subset separately so the enlarged study is not presented as wholly independent of the original 200 cases.",
        "- Retain nonsignificant degradation findings as robustness boundaries, not as evidence of improvement.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
