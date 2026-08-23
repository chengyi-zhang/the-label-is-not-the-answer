#!/usr/bin/env python3
"""Integrity audit for overnight anchor and 600-case counterfactual studies."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
LABELS = set("ABCDE")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_result(name: str, result: Path, gold: dict, expected_n: int):
    rows = load(result)["answers"]
    pred = {row["case_id"]: row["answer"] for row in rows}
    assert len(rows) == len(pred) == expected_n, f"{name}: count or duplicate-ID failure"
    assert set(pred) == set(gold), f"{name}: ID mismatch"
    assert all(value in LABELS for value in pred.values()), f"{name}: invalid label"
    print(f"PASS {name}: {expected_n} complete valid predictions")


def main():
    source_gold = load(ROOT / "data/experiments/eval_600/gold.json")
    for anchor in "ABCDE":
        audit_result(
            f"anchor_{anchor}",
            ROOT / f"outputs/anchor_control_600/anchor_{anchor}/results.json",
            source_gold,
            600,
        )

    split = ROOT / "data/experiments/counterfactual_600"
    gold = load(split / "gold.json")
    manifest = load(split / "manifest.json")
    metadata = load(split / "metadata.json")
    assert len(gold) == len(metadata) == len(manifest["records"]) == 3000
    assert len(manifest["selected_ids"]) == 600
    for variant in ("submitted", "comparative", "cue_free"):
        audit_result(
            f"counterfactual_600/{variant}",
            ROOT / f"outputs/counterfactual_600/{variant}/results.json",
            gold,
            3000,
        )

    report_json = ROOT / "analysis/overnight_extensions/report.json"
    report_md = ROOT / "analysis/overnight_extensions/report.md"
    assert report_json.stat().st_size > 5000
    assert report_md.stat().st_size > 1000
    load(report_json)
    print("PASS comprehensive reports are present and parseable")
    print("ALL OVERNIGHT EXTENSION AUDITS PASSED")


if __name__ == "__main__":
    main()
