import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
LABELS = "ABCDE"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_predictions(path):
    payload = load_json(path)
    return {row["case_id"]: row["answer"] for row in payload["answers"]}


def exact_mcnemar(correct_a, correct_b):
    improved = sum((not a) and b for a, b in zip(correct_a, correct_b))
    worsened = sum(a and (not b) for a, b in zip(correct_a, correct_b))
    n = improved + worsened
    if n == 0:
        return improved, worsened, 1.0
    tail = sum(math.comb(n, k) for k in range(min(improved, worsened) + 1)) / 2**n
    return improved, worsened, min(1.0, 2 * tail)


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def exact_cluster_signflip(differences):
    magnitudes = [abs(int(value)) for value in differences if value]
    observed = abs(sum(int(value) for value in differences))
    distribution = {0: 1}
    for magnitude in magnitudes:
        updated = defaultdict(int)
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = updated
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    return extreme / (2 ** len(magnitudes)) if magnitudes else 1.0


def cluster_bootstrap_ci(differences, seed=20260807, draws=200000):
    values, counts = np.unique(np.asarray(differences, dtype=int), return_counts=True)
    rng = np.random.default_rng(seed)
    sampled_counts = rng.multinomial(len(differences), counts / counts.sum(), size=draws)
    deltas = (sampled_counts @ values) / (len(differences) * 5) * 100
    return tuple(np.quantile(deltas, [0.025, 0.975]))


def anchor_test():
    table = np.asarray(
        [
            [597, 1, 1, 0, 1],
            [1, 598, 0, 0, 1],
            [0, 1, 595, 0, 4],
            [2, 2, 2, 588, 6],
            [2, 0, 3, 6, 589],
        ],
        dtype=float,
    )
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi_square = float(((table - expected) ** 2 / expected).sum())
    cramers_v = math.sqrt(chi_square / (table.sum() * 4))
    print("ANCHOR")
    print(f"chi_square={chi_square:.8f} df=16 cramers_v={cramers_v:.8f}")


def cyclic_tests():
    base = ROOT / "outputs/counterfactual_600"
    gold = load_json(ROOT / "data/experiments/counterfactual_600/gold.json")
    manifest = load_json(ROOT / "data/experiments/counterfactual_600/manifest.json")
    methods = {
        "Submitted": load_predictions(base / "submitted/results.json"),
        "Comparative": load_predictions(base / "comparative/results.json"),
        "Cue-free": load_predictions(base / "cue_free/results.json"),
    }
    records_by_case = defaultdict(list)
    for row in manifest["records"]:
        records_by_case[row["original_case_id"]].append(row["case_id"])
    correct_counts = {}
    for name, predictions in methods.items():
        correct_counts[name] = [
            sum(predictions[case_id] == gold[case_id] for case_id in records_by_case[original])
            for original in manifest["selected_ids"]
        ]
    comparisons = [
        ("Submitted", "Comparative"),
        ("Comparative", "Cue-free"),
        ("Submitted", "Cue-free"),
    ]
    rows = []
    for left, right in comparisons:
        differences = np.asarray(correct_counts[right]) - np.asarray(correct_counts[left])
        delta = differences.mean() / 5 * 100
        p_value = exact_cluster_signflip(differences)
        low, high = cluster_bootstrap_ci(differences)
        rows.append((left, right, delta, low, high, p_value, Counter(differences)))
    adjusted = holm_adjust([row[5] for row in rows])
    print("CYCLIC_CLUSTERED")
    for row, p_adjusted in zip(rows, adjusted):
        left, right, delta, low, high, p_value, counts = row
        print(
            f"{left}->{right} delta_pp={delta:.4f} ci95=[{low:.4f},{high:.4f}] "
            f"exact_cluster_p={p_value:.10g} holm_p={p_adjusted:.10g} "
            f"case_diff_counts={dict(sorted(counts.items()))}"
        )


def canonical_tests():
    datasets = {
        "3B primary": {
            "gold": ROOT / "data/experiments/eval_600/gold.json",
            "Submitted": ROOT / "outputs/eval_600/submitted/results.json",
            "Comparative": ROOT / "outputs/eval_600/comparative/results.json",
            "Cue-free": ROOT / "outputs/eval_600/cue_free/results.json",
        },
        "3B replication": {
            "gold": ROOT / "data/experiments/eval_600_replication/gold.json",
            "Submitted": ROOT / "outputs/eval_600_replication_3b/submitted/results.json",
            "Comparative": ROOT / "outputs/eval_600_replication_3b/comparative/results.json",
            "Cue-free": ROOT / "outputs/eval_600_replication_3b/cue_free/results.json",
        },
        "7B primary": {
            "gold": ROOT / "data/experiments/eval_600/gold.json",
            "Submitted": ROOT / "outputs/eval_600_7b/submitted/results.json",
            "Comparative": ROOT / "outputs/eval_600_7b/comparative/results.json",
            "Cue-free": ROOT / "outputs/eval_600_7b/cue_free/results.json",
        },
    }
    comparisons = [
        ("Submitted", "Comparative"),
        ("Comparative", "Cue-free"),
        ("Submitted", "Cue-free"),
    ]
    print("CANONICAL")
    for dataset, paths in datasets.items():
        gold = load_json(paths["gold"])
        ids = list(gold)
        predictions = {name: load_predictions(path) for name, path in paths.items() if name != "gold"}
        correct = {
            name: [values[case_id] == gold[case_id] for case_id in ids]
            for name, values in predictions.items()
        }
        print(dataset)
        for name in ("Submitted", "Comparative", "Cue-free"):
            count = sum(correct[name])
            print(f"  {name}: {count}/600 ({count / 6:.2f}%)")
        tests = []
        for left, right in comparisons:
            improved, worsened, p_value = exact_mcnemar(correct[left], correct[right])
            delta = (sum(correct[right]) - sum(correct[left])) / 6
            tests.append((left, right, delta, improved, worsened, p_value))
        adjusted = holm_adjust([row[5] for row in tests])
        for row, p_adjusted in zip(tests, adjusted):
            left, right, delta, improved, worsened, p_value = row
            print(
                f"  {left}->{right}: delta_pp={delta:.2f} improved={improved} "
                f"worsened={worsened} exact_p={p_value:.10g} holm_p={p_adjusted:.10g}"
            )


if __name__ == "__main__":
    anchor_test()
    cyclic_tests()
    canonical_tests()
