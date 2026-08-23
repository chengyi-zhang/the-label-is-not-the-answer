#!/usr/bin/env python3
import collections
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = ROOT / "data/train"
SOURCE = DATA_ROOT / "medreason_train_selection.json"
REFERENCE = ROOT / "data/experiments/eval_600/gold.json"
OUTPUT_DIR = ROOT / "data/experiments/eval_600_replication"
MANIFEST = json.loads((ROOT / "configs/experiment_manifest.json").read_text(encoding="utf-8"))
SELECTION_KEY = bytes.fromhex(MANIFEST["selection_keys_hex"]["holdout"])
LABELS = tuple("ABCDE")
PER_LABEL = 120


def stable_key(record):
    payload = SELECTION_KEY + b":" + str(record["case_id"]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_image_path(raw_path):
    path = str(raw_path).replace("\\", "/")
    return path[len("dataset/") :] if path.startswith("dataset/") else path


payload = json.loads(SOURCE.read_text(encoding="utf-8"))
excluded = set(json.loads(REFERENCE.read_text(encoding="utf-8")))
records = [
    record
    for record in payload["cases"]
    if str(record.get("question type", "")).strip().lower() == "mcq"
    and str(record["case_id"]) not in excluded
]

selected = []
for label in LABELS:
    candidates = sorted((record for record in records if record["answer"] == label), key=stable_key)
    if len(candidates) < PER_LABEL:
        raise ValueError(f"Not enough non-overlapping records for {label}")
    selected.extend(candidates[:PER_LABEL])
selected.sort(key=stable_key)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
image_link = OUTPUT_DIR / "imgs"
if not image_link.exists():
    image_link.symlink_to(DATA_ROOT / "imgs", target_is_directory=True)

cases = []
gold = {}
metadata = {}
for record in selected:
    case_id = str(record["case_id"])
    image_path = normalized_image_path(record["image_path"])
    cases.append(
        {
            "case_id": case_id,
            "task_type": "mcq",
            "question": str(record["question"]),
            "image_path": image_path,
            "options": {label: str(record[label]) for label in LABELS},
        }
    )
    gold[case_id] = str(record["answer"])
    metadata[case_id] = {
        "answer": str(record["answer"]),
        "task": str(record["task"]),
        "modality": str(record["primary_modality"]),
        "organ_system": str(record["organ system"]),
        "subtype": str(record["subtype"]),
        "source_pool": str(record["source_pool"]),
        "image_path": image_path,
    }

(OUTPUT_DIR / "cases.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
(OUTPUT_DIR / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

missing = [case["image_path"] for case in cases if not (DATA_ROOT / case["image_path"]).is_file()]
overlap = set(gold) & excluded
if missing or overlap:
    raise RuntimeError(f"missing={len(missing)} overlap={len(overlap)}")

print(f"cases={len(cases)} overlap_with_eval600={len(overlap)} missing_images={len(missing)}")
print("answers=" + json.dumps(collections.Counter(gold.values()), sort_keys=True))
for field in ("task", "modality", "organ_system"):
    print(field + "=" + json.dumps(collections.Counter(row[field] for row in metadata.values()), sort_keys=True))
