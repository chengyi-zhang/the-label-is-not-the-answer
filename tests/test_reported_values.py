from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 1e-5) -> None:
    assert math.isclose(actual, expected, abs_tol=tolerance), (actual, expected)


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "19.97 / 23.30 / 26.27%" in readme
    assert "21.00 / 24.17 / 29.33%" in readme

    with (ROOT / "results/canonical/primary_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        primary = {row["condition"]: row for row in csv.DictReader(handle)}
    close(float(primary["Submitted"]["accuracy_percent"]), 20.166667)
    close(float(primary["Comparative"]["accuracy_percent"]), 24.666667)
    close(float(primary["Cue-free"]["accuracy_percent"]), 28.5)

    replication = json.loads(
        (ROOT / "results/canonical/replication_and_scale.json").read_text(encoding="utf-8")
    )
    studies = {study["name"]: study for study in replication["studies"]}
    close(
        studies["Qwen2.5-VL-3B same-source holdout"]["conditions"]["Cue-free"]["accuracy_percent"],
        29.333333,
    )
    close(
        studies["Qwen2.5-VL-7B primary"]["conditions"]["Cue-free"]["accuracy_percent"],
        35.166667,
    )

    cyclic = json.loads(
        (ROOT / "results/cyclic/full_audit.json").read_text(encoding="utf-8")
    )["cyclic_label_reassignment"]["subsets"]["full_600"]
    close(cyclic["Submitted"]["performance"]["accuracy"], 19.966667)
    close(cyclic["Comparative"]["all_five_semantic_invariance"], 14.333333)
    close(cyclic["Cue-free"]["all_five_semantic_invariance"], 15.0)

    qwen_anchor = json.loads(
        (ROOT / "results/anchor_intervention/qwen2_5_vl_3b.json").read_text(encoding="utf-8")
    )
    close(qwen_anchor["cramers_v"], 0.986286571136002)
    assert qwen_anchor["cases_changed_by_at_least_one_cue"] == 600

    internvl = json.loads(
        (ROOT / "results/cross_family/internvl_anchor_intervention.json").read_text(encoding="utf-8")
    )
    close(internvl["aggregate"]["cramers_v"], 0.9528466187146513)
    assert internvl["aggregate"]["cases_changed_by_anchor"] == 200

    print("Reported-value checks passed.")


if __name__ == "__main__":
    main()
