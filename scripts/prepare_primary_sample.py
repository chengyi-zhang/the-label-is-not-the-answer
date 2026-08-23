#!/usr/bin/env python3
import collections
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = ROOT / "data" / "train"
SOURCE = DATA_ROOT / "medreason_train_selection.json"
EXPERIMENT_ROOT = ROOT / "data" / "experiments"
MANIFEST = json.loads((ROOT / "configs/experiment_manifest.json").read_text(encoding="utf-8"))
SELECTION_KEY = bytes.fromhex(MANIFEST["selection_keys_hex"]["primary"])
SIZES = {"pilot_60": 60, "eval_600": 600}
LABELS = tuple("ABCDE")


def stable_key(record):
    text = SELECTION_KEY + b":" + str(record["case_id"]).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def normalized_image_path(raw_path):
    path = str(raw_path).replace("\\", "/")
    if path.startswith("dataset/"):
        path = path[len("dataset/") :]
    return path


def select_balanced(records, total):
    if total % len(LABELS):
        raise ValueError("Sample size must be divisible by five")
    per_label = total // len(LABELS)
    selected = []
    for label in LABELS:
        candidates = sorted(
            (record for record in records if record["answer"] == label),
            key=stable_key,
        )
        if len(candidates) < per_label:
            raise ValueError(f"Not enough records for answer label {label}")
        selected.extend(candidates[:per_label])
    return sorted(selected, key=stable_key)


def write_split(name, selected):
    output_dir = EXPERIMENT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    image_link = output_dir / "imgs"
    if image_link.is_symlink() and image_link.resolve() != (DATA_ROOT / "imgs").resolve():
        image_link.unlink()
    if not image_link.exists():
        image_link.symlink_to(DATA_ROOT / "imgs", target_is_directory=True)

    cases = []
    gold = {}
    metadata = {}
    for record in selected:
        case_id = str(record["case_id"])
        options = {label: str(record[label]) for label in LABELS}
        image_path = normalized_image_path(record["image_path"])
        cases.append(
            {
                "case_id": case_id,
                "task_type": "mcq",
                "question": str(record["question"]),
                "image_path": image_path,
                "options": options,
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

    (output_dir / "cases.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    missing = [case["image_path"] for case in cases if not (DATA_ROOT / case["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images; first: {missing[0]}")

    print(f"{name}: {len(cases)} cases")
    for field in ("answer", "task", "modality", "organ_system"):
        counts = collections.Counter(item[field] for item in metadata.values())
        print(f"  {field}: {json.dumps(counts, ensure_ascii=True, sort_keys=True)}")


with SOURCE.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

mcq_records = [
    record
    for record in payload["cases"]
    if str(record.get("question type", "")).strip().lower() == "mcq"
]
print(f"source_mcq_records={len(mcq_records)}")
print("source_answers=" + json.dumps(collections.Counter(r["answer"] for r in mcq_records), sort_keys=True))

for split_name, split_size in SIZES.items():
    write_split(split_name, select_balanced(mcq_records, split_size))
