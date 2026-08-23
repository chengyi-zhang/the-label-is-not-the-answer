#!/usr/bin/env python3
"""Prepare the full 600-case cyclic relabeling study deterministically."""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "data/experiments/eval_600"
TARGET = ROOT / "data/experiments/counterfactual_600"
LABELS = "ABCDE"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_option(text: str) -> str:
    return re.sub(r"^\s*[A-E]\s*[\)\].:\-]\s*", "", str(text), count=1)


def main():
    payload = load(SOURCE / "cases.json")
    cases = payload["cases"]
    gold = load(SOURCE / "gold.json")
    metadata = load(SOURCE / "metadata.json")
    assert len(cases) == len(gold) == len(metadata) == 600
    assert Counter(gold.values()) == Counter({label: 120 for label in LABELS})

    TARGET.mkdir(parents=True, exist_ok=True)
    image_link = TARGET / "imgs"
    if not image_link.exists():
        image_link.symlink_to(SOURCE / "imgs", target_is_directory=True)

    transformed_cases = []
    transformed_gold = {}
    transformed_metadata = {}
    records = []
    selected_ids = []

    for case in cases:
        original_id = case["case_id"]
        selected_ids.append(original_id)
        for shift in range(5):
            row = copy.deepcopy(case)
            new_id = f"{original_id}__perm{shift}"
            row["case_id"] = new_id
            new_options = {}
            label_to_original = {}
            for old_index, old_label in enumerate(LABELS):
                new_label = LABELS[(old_index + shift) % 5]
                new_options[new_label] = clean_option(case["options"][old_label])
                label_to_original[new_label] = old_label
            row["options"] = {label: new_options[label] for label in LABELS}
            new_gold = LABELS[(LABELS.index(gold[original_id]) + shift) % 5]
            transformed_cases.append(row)
            transformed_gold[new_id] = new_gold
            transformed_metadata[new_id] = {
                **metadata[original_id],
                "answer": new_gold,
                "original_case_id": original_id,
                "permutation_shift": shift,
            }
            records.append({
                "case_id": new_id,
                "original_case_id": original_id,
                "shift": shift,
                "gold": new_gold,
                "original_gold": gold[original_id],
                "label_to_original": label_to_original,
            })

    output_payload = {key: value for key, value in payload.items() if key != "cases"}
    output_payload["cases"] = transformed_cases
    dump(TARGET / "cases.json", output_payload)
    dump(TARGET / "gold.json", transformed_gold)
    dump(TARGET / "metadata.json", transformed_metadata)
    dump(TARGET / "manifest.json", {"selected_ids": selected_ids, "records": records})

    assert len(transformed_cases) == len(transformed_gold) == len(records) == 3000
    assert Counter(transformed_gold.values()) == Counter({label: 600 for label in LABELS})
    print("PREPARED counterfactual_600: 600 cases x 5 shifts = 3000 inputs")
    print("gold_distribution", dict(Counter(transformed_gold.values())))


if __name__ == "__main__":
    main()
