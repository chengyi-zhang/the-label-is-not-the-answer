from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

COLORS = {
    "Submitted": "#8A959A",
    "Comparative": "#AE5A45",
    "Cue-free": "#315F82",
}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (center - half), 100 * (center + half)


def write_csv(name: str, rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_b() -> None:
    counts = [
        [597, 1, 1, 0, 1],
        [1, 598, 0, 0, 1],
        [0, 1, 595, 0, 4],
        [2, 2, 2, 588, 6],
        [2, 0, 3, 6, 589],
    ]
    labels = list("ABCDE")
    rows = []
    for row_index, cue in enumerate(labels):
        row = {"schema_cue": cue}
        for col_index, reported in enumerate(labels):
            row[reported] = counts[row_index][col_index] / 6.0
        rows.append(row)
    write_csv("panel_b_cue_response_matrix.csv", rows)


def panel_c() -> None:
    rows: list[dict] = []
    # These are pooled accuracies across 600 cases x five cyclic relabelings.
    # The five predictions from one case are correlated, so the paired,
    # case-cluster bootstrap below carries the inferential uncertainty. We do
    # not attach an independence-based Wilson interval to these point estimates.
    accuracy = [("Submitted", 599), ("Comparative", 699), ("Cue-free", 788)]
    for order, (label, correct) in enumerate(accuracy, start=1):
        estimate = 100 * correct / 3000
        rows.append(
            {
                "section": "Accuracy across cyclic relabelings (%)",
                "display_order": order,
                "label": label,
                "estimate": f"{estimate:.6f}",
                "ci_low": "",
                "ci_high": "",
                "n": 3000,
                "ci_method": "Point estimate; case-clustered inference reported as paired gains",
                "p_label": "",
                "color_hex": COLORS[label],
            }
        )

    gains = [
        ("Submitted to Comparative", 3.3333, 0.9333, 5.8000, "p=.0079", "Comparative"),
        ("Comparative to Cue-free", 2.9667, 1.2667, 4.6675, "p=.0016", "Cue-free"),
        ("Submitted to Cue-free", 6.3000, 3.7667, 8.8667, "p<.0001", "Cue-free"),
    ]
    for order, (label, estimate, low, high, p_label, color_key) in enumerate(gains, start=1):
        rows.append(
            {
                "section": "Paired accuracy gain (pp; 95% CI)",
                "display_order": order,
                "label": label,
                "estimate": f"{estimate:.6f}",
                "ci_low": f"{low:.6f}",
                "ci_high": f"{high:.6f}",
                "n": 600,
                "ci_method": "Case-cluster bootstrap 95% CI",
                "p_label": p_label,
                "color_hex": COLORS[color_key],
            }
        )

    invariance = [("Submitted", 0), ("Comparative", 86), ("Cue-free", 90)]
    for order, (label, correct) in enumerate(invariance, start=1):
        estimate = 100 * correct / 600
        low, high = wilson(correct, 600)
        rows.append(
            {
                "section": "All-five semantic invariance (%)",
                "display_order": order,
                "label": label,
                "estimate": f"{estimate:.6f}",
                "ci_low": f"{low:.6f}",
                "ci_high": f"{high:.6f}",
                "n": 600,
                "ci_method": "Wilson 95% CI",
                "p_label": "",
                "color_hex": COLORS[label],
            }
        )
    write_csv("panel_c_forest.csv", rows)


def panel_d() -> None:
    studies = [
        ("3B primary", [121, 148, 171], "+8.33"),
        ("3B replication", [126, 145, 176], "+8.33"),
        ("7B primary", [153, 188, 211], "+9.67"),
    ]
    methods = ["Submitted", "Comparative", "Cue-free"]
    rows = []
    for study_order, (study, correct_values, gain) in enumerate(studies, start=1):
        for method_order, (method, correct) in enumerate(zip(methods, correct_values), start=1):
            estimate = 100 * correct / 600
            low, high = wilson(correct, 600)
            rows.append(
                {
                    "study_order": study_order,
                    "study": study,
                    "method_order": method_order,
                    "method": method,
                    "estimate": f"{estimate:.6f}",
                    "ci_low": f"{low:.6f}",
                    "ci_high": f"{high:.6f}",
                    "n": 600,
                    "ci_method": "Wilson 95% CI",
                    "submitted_to_cue_free_gain_pp": gain if method == "Cue-free" else "",
                    "color_hex": COLORS[method],
                }
            )
    write_csv("panel_d_replication_scale.csv", rows)


def manifest() -> None:
    payload = {
        "source": "make_figure_v12.py",
        "figure_dimensions_mm": [183, 112],
        "panel_target_dimensions_mm": {
            "b": [56, 43],
            "c": [109, 50],
            "d": [56, 47],
        },
        "palette": COLORS,
        "integrity_notes": [
            "Panel B contains row percentages computed from 600 cases per schema cue.",
            "Panel C cyclic accuracies are pooled point estimates over 600 cases x five relabelings; no independence-based CI is attached.",
            "Panel C paired gains use case-cluster bootstrap intervals over 600 case clusters.",
            "Panel C all-five invariance uses Wilson intervals over 600 cases.",
            "Panel D uses Wilson intervals for canonical accuracy.",
            "All CI columns store absolute lower and upper endpoints, not symmetric error widths.",
        ],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "source_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    panel_b()
    panel_c()
    panel_d()
    manifest()
    print(DATA_DIR)
