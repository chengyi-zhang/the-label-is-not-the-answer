#!/usr/bin/env python3
"""Cross-family replication of the MedReason output-schema anchor intervention."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


LABELS = "ABCDE"
ROOT = Path(os.environ.get("MEDREASON_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data/experiments/eval_600"
MANIFEST = ROOT / "data/experiments/counterfactual_200/manifest.json"
MODEL_DIR = Path(os.environ.get("MEDREASON_INTERNVL_MODEL", ROOT / "models/InternVL3-9B"))
OUTPUT_DIR = ROOT / "outputs/external_validation_internvl3_9b_strict"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SYSTEM_PROMPT = (
    "You are a medical visual reasoning assistant for the MedReason Challenge. "
    "Use the provided image evidence and the question. Do not invent findings that are not "
    "supported by the image. If the evidence is insufficient, state the limitation clearly. "
    "Return only valid JSON with two keys in this order: answer and reasoning_trace."
)


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    image_path: Path
    options: dict[str, str]


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for index in range(blocks):
        box = (
            (index % target_ratio[0]) * image_size,
            (index // target_ratio[0]) * image_size,
            ((index % target_ratio[0]) + 1) * image_size,
            ((index // target_ratio[0]) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) > 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def load_image(path: Path, max_num: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = build_transform()
    tiles = dynamic_preprocess(image, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def load_cases(limit: int) -> list[Case]:
    payload = json.loads((DATA_DIR / "cases.json").read_text(encoding="utf-8"))
    rows = payload["cases"] if isinstance(payload, dict) else payload
    selected = json.loads(MANIFEST.read_text(encoding="utf-8"))["selected_ids"][:limit]
    by_id = {str(row["case_id"]): row for row in rows}
    cases = []
    for case_id in selected:
        row = by_id[case_id]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path
        options = row["options"]
        if isinstance(options, list):
            options = {
                str(item.get("label", LABELS[index])): str(item.get("text", item.get("option", "")))
                if isinstance(item, dict)
                else str(item)
                for index, item in enumerate(options)
            }
        cases.append(
            Case(
                case_id=case_id,
                question=str(row["question"]).strip(),
                image_path=image_path,
                options={str(key): str(value) for key, value in options.items()},
            )
        )
    if len(cases) != limit or len({case.case_id for case in cases}) != limit:
        raise ValueError("Selected case list is incomplete or contains duplicates")
    if any(not case.image_path.is_file() for case in cases):
        raise FileNotFoundError("At least one selected image is missing")
    return cases


def build_prompt(case: Case, anchor: str) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in case.options.items())
    instruction = f'''Task: closed-ended multiple-choice medical visual question answering.

Question:
{case.question}

Options:
{options}

Instructions:
1. Inspect the image evidence relevant to the question.
2. Choose exactly one option label from the provided options.
3. Put the answer field first. It must contain only the selected label, for example "{anchor}".
4. The reasoning_trace should be concise and image-grounded.

Return only JSON:
{{
  "answer": "{anchor}",
  "reasoning_trace": "brief visual rationale"
}}'''
    return f"<image>\n{SYSTEM_PROMPT}\n\n{instruction}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(stripped[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _normalize_answer_legacy(raw_text: str, options: dict[str, str]) -> tuple[str, str]:
    obj = extract_json_object(raw_text) or {}
    answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()
    labels = list(options)
    canonical = {label.upper(): label for label in labels}
    if answer.upper() in canonical:
        return canonical[answer.upper()], "json_direct"
    candidate = answer or raw_text
    match = re.search(
        r"(?:final answer|answer|option|choice)?\s*[:：\(\[]?\s*([A-E])\s*[\)\].:]?",
        candidate,
        flags=re.I,
    )
    if match and match.group(1).upper() in canonical:
        return canonical[match.group(1).upper()], "regex_label"
    lowered = candidate.lower()
    for label, text in options.items():
        if text and text.lower() in lowered:
            return label, "option_text"
    return labels[0], "fallback_first_option"


def normalize_answer(raw_text: str, options: dict[str, str]) -> tuple[str, str]:
    """Extract only explicit answer fields; never infer a label from arbitrary prose."""
    obj = extract_json_object(raw_text) or {}
    answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()
    labels = list(options)
    canonical = {label.upper(): label for label in labels}
    if answer.upper() in canonical:
        return canonical[answer.upper()], "json_direct"

    candidate = answer or raw_text
    match = re.search(
        r'''["']?answer["']?\s*:\s*["']?\s*([A-E])\b''',
        candidate,
        flags=re.I,
    ) or re.search(
        r"(?:final answer|answer|option|choice)\s*(?:is|:)\s*[\(\[]?\s*([A-E])\b",
        candidate,
        flags=re.I,
    )
    if match and match.group(1).upper() in canonical:
        return canonical[match.group(1).upper()], "regex_label"

    lowered = candidate.lower()
    text_matches = [label for label, text in options.items() if text and text.lower() in lowered]
    if len(text_matches) == 1:
        return text_matches[0], "option_text"
    return "", "unparsed"


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    return completed


def write_results(anchor: str, rows: list[dict[str, Any]]) -> None:
    output = OUTPUT_DIR / f"anchor_{anchor}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "InternVL3-9B cross-family anchor validation",
        "type": "Medical visual reasoning",
        "model": "OpenGVLab/InternVL3-9B",
        "anchor": anchor,
        "answers": [
            {
                "case_id": row["case_id"],
                "task_type": "mcq",
                "answer": row["answer"],
                "reasoning_trace": row["raw_response"],
                "confidence": 0.0,
                "parse_mode": row["parse_mode"],
                "elapsed_seconds": row["elapsed_seconds"],
                "image_tiles": row["image_tiles"],
            }
            for row in rows
        ],
        "version": {"major": 1, "minor": 0},
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", default=LABELS)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    anchors = "".join(dict.fromkeys(args.anchors.strip().upper()))
    if not anchors or any(anchor not in LABELS for anchor in anchors):
        raise ValueError("--anchors must contain only A-E")
    cases = load_cases(args.limit)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_DIR} for {len(cases)} cases x {len(anchors)} anchors", flush=True)
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, use_fast=False
    )
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    for anchor in anchors:
        raw_path = OUTPUT_DIR / f"anchor_{anchor}" / "raw.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        completed = read_completed(raw_path)
        print(f"Anchor {anchor}: resuming from {len(completed)}/{len(cases)}", flush=True)
        for index, case in enumerate(cases, start=1):
            if case.case_id in completed:
                continue
            started = time.perf_counter()
            pixel_values = load_image(case.image_path, args.max_tiles).to(
                dtype=torch.bfloat16, device="cuda", non_blocking=True
            )
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    build_prompt(case, anchor),
                    generation_config,
                )
            answer, parse_mode = normalize_answer(str(response), case.options)
            row = {
                "case_id": case.case_id,
                "anchor": anchor,
                "answer": answer,
                "parse_mode": parse_mode,
                "raw_response": str(response),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "image_tiles": int(pixel_values.shape[0]),
            }
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            completed[case.case_id] = row
            if index == 1 or index % 10 == 0:
                print(
                    f"Anchor {anchor}: {index}/{len(cases)} answer={answer} "
                    f"parse={parse_mode} time={row['elapsed_seconds']:.2f}s",
                    flush=True,
                )
            del pixel_values

        ordered = [completed[case.case_id] for case in cases]
        if len(ordered) != len(cases):
            raise RuntimeError(f"Anchor {anchor} incomplete")
        write_results(anchor, ordered)
    print("All requested anchors completed", flush=True)


if __name__ == "__main__":
    main()
