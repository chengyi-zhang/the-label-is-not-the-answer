#!/usr/bin/env python3
"""
MedReason Qwen2.5-VL baseline Docker entry point.

This file intentionally follows the style of the VLM3D `reportgen_example_docker`:

* no required runtime CLI flags;
* fixed container paths under /input, /output, and /opt/app/models;
* the main baseline logic is visible in process.py;
* a self-contained model checkpoint is expected to be baked into the Docker image;
* the container writes a single JSON file to /output/results.json.

The baseline is a simple single-model MLLM system:

    images + question/options
        -> MedReason task-specific prompt
        -> Qwen2.5-VL generation
        -> JSON/text parsing
        -> MCQ answer normalization
        -> /output/results.json

Participants may replace this implementation with a stronger system, including
agentic workflows, multi-model ensembles, retrieval modules, or self-verification,
as long as the same input/output contract is preserved.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# HARD-WIRED CONTAINER PATHS & PARAMETERS — EDIT HERE IF NEEDED
# ---------------------------------------------------------------------------
INPUT_DIR = Path(os.environ.get("MEDREASON_INPUT_DIR", "/input"))
CASES_FILE = INPUT_DIR / os.environ.get("MEDREASON_CASES_FILE", "cases.json")
OUT_FILE = Path(os.environ.get("MEDREASON_OUTPUT_FILE", "/output/results.json"))
MODEL_PATH = Path(os.environ.get("MEDREASON_MODEL_PATH", "/opt/app/models/Qwen2.5-VL"))

DEVICE = os.environ.get("MEDREASON_DEVICE", "cuda")
DTYPE = os.environ.get("MEDREASON_DTYPE", "auto")
MAX_NEW_TOKENS = int(os.environ.get("MEDREASON_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("MEDREASON_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("MEDREASON_TOP_P", "1.0"))
SUBMISSION_NAME = os.environ.get("MEDREASON_SUBMISSION_NAME", "MedReason Comparative-prompt predictions")
SMOKE_TEST = os.environ.get("MEDREASON_SMOKE_TEST", "0").lower() in {"1", "true", "yes"}

VALID_TASK_TYPES = {"mcq", "open", "open_ended", "closed_ended"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# PROMPT BANK
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical visual reasoning assistant for the MedReason Challenge.
Use the provided image evidence and the question. Do not invent findings that are not supported by the image. If the evidence is insufficient, state the limitation clearly.
Return only valid JSON with two keys: reasoning_trace and answer."""

MCQ_INSTRUCTION = """Task: closed-ended multiple-choice medical visual question answering.

Question:
{question}

Options:
{options}

Instructions:
1. Inspect the image evidence relevant to the question before choosing.
2. Compare all option labels one by one. Do not choose the first option by default.
3. Pay close attention to negation, location, extent, laterality, layer/depth, temporal wording, and therapy-related wording.
4. Choose exactly one option label from the provided options.
5. The answer field must contain only the selected option label, such as B or D.
6. The reasoning_trace should briefly state why the chosen option matches the image better than the closest alternatives.
7. If uncertain, still choose the best-supported option after comparing all options; do not fall back to A.

Return only valid JSON:
{{
  "reasoning_trace": "brief visual rationale comparing the selected option against close alternatives",
  "answer": "selected option label only"
}}"""

OPEN_INSTRUCTION = """Task: open-ended medical visual reasoning.

Question:
{question}

Instructions:
1. Describe the visual evidence that supports the final answer.
2. Mention uncertainty or evidence limitations when relevant.
3. Keep the final answer concise and clinically focused.
4. The final answer alone is not sufficient; reasoning_trace is required.

Return only JSON:
{{
  "reasoning_trace": "image-grounded evidence and reasoning",
  "answer": "concise final answer"
}}"""

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Option:
    label: str
    text: str


@dataclass(frozen=True)
class Case:
    case_id: str
    task_type: str
    question: str
    image_paths: Tuple[Path, ...]
    options: Tuple[Option, ...] = ()


@dataclass(frozen=True)
class Prediction:
    case_id: str
    task_type: str
    answer: str
    reasoning_trace: str
    confidence: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": canonical_task_type(self.task_type),
            "answer": self.answer,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# INPUT / OUTPUT HELPERS
# ---------------------------------------------------------------------------
def canonical_task_type(task_type: str) -> str:
    lowered = str(task_type).strip().lower().replace("-", "_")
    if lowered in {"closed_ended", "closed", "multiple_choice", "mcq"}:
        return "mcq"
    if lowered in {"open", "open_ended", "free_response", "vqa"}:
        return "open"
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _as_options(raw_options: Any) -> Tuple[Option, ...]:
    if raw_options is None:
        return ()
    options: List[Option] = []
    if isinstance(raw_options, dict):
        for label, text in raw_options.items():
            options.append(Option(label=str(label).strip(), text=str(text).strip()))
    elif isinstance(raw_options, list):
        for i, item in enumerate(raw_options):
            if isinstance(item, dict):
                label = str(item.get("label", chr(ord("A") + i))).strip()
                text = str(item.get("text", item.get("option", ""))).strip()
            else:
                label = chr(ord("A") + i)
                text = str(item).strip()
            options.append(Option(label=label, text=text))
    else:
        raise ValueError("options must be a list or dict")
    if len({o.label for o in options}) != len(options):
        raise ValueError("MCQ option labels must be unique")
    return tuple(options)


def _resolve_image_paths(raw_case: Dict[str, Any]) -> Tuple[Path, ...]:
    raw_paths = raw_case.get("image_paths", None)
    if raw_paths is None:
        raw_paths = raw_case.get("image_path", None)
    if raw_paths is None:
        raise ValueError(f"case {raw_case.get('case_id')} is missing image_path/image_paths")
    if isinstance(raw_paths, (str, os.PathLike)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"case {raw_case.get('case_id')} image_paths must be a non-empty list")

    paths: List[Path] = []
    for raw_path in raw_paths:
        p = Path(str(raw_path))
        if not p.is_absolute():
            p = INPUT_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            warnings.warn(f"Image extension may not be supported by the baseline: {p.name}")
        paths.append(p)
    return tuple(paths)


def load_cases(path: Path) -> List[Case]:
    if not path.exists():
        raise FileNotFoundError(f"cases.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload if isinstance(payload, list) else None)
    if not isinstance(raw_cases, list):
        raise ValueError("cases.json must be either a list of cases or an object with a 'cases' list")

    cases: List[Case] = []
    seen = set()
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"case at index {idx} must be an object")
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"case at index {idx} is missing case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)

        task_type = canonical_task_type(str(raw.get("task_type", "")))
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"case {case_id} is missing question")
        options = _as_options(raw.get("options"))
        if task_type == "mcq" and not options:
            raise ValueError(f"MCQ case {case_id} must include options")
        cases.append(
            Case(
                case_id=case_id,
                task_type=task_type,
                question=question,
                image_paths=_resolve_image_paths(raw),
                options=options,
            )
        )
    return cases


def write_results(predictions: Sequence[Prediction], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": SUBMISSION_NAME,
        "type": "Medical visual reasoning",
        "answers": [p.to_json() for p in predictions],
        "version": {"major": 1, "minor": 0},
    }
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PROMPTING / PARSING
# ---------------------------------------------------------------------------
def _stable_rotation_offset(case_id: str, n: int) -> int:
    """Return a deterministic rotation offset to reduce first-option positional bias."""
    if n <= 0:
        return 0
    return sum(ord(ch) for ch in str(case_id)) % n


def rotated_options_for_prompt(case: Case) -> Tuple[Option, ...]:
    """Rotate display order while preserving original labels, e.g. D, E, A, B, C."""
    options = tuple(case.options)
    if case.task_type != "mcq" or len(options) <= 1:
        return options
    offset = _stable_rotation_offset(case.case_id, len(options))
    return options[offset:] + options[:offset]


def format_options(options: Sequence[Option]) -> str:
    return "\n".join(f"{opt.label}. {opt.text}" for opt in options)


def build_text_prompt(case: Case) -> str:
    if case.task_type == "mcq":
        # Keep the canonical A-E order while retaining the comparative instruction.
        return MCQ_INSTRUCTION.format(question=case.question, options=format_options(case.options))
    return OPEN_INSTRUCTION.format(question=case.question)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    # Remove common code fences if present.
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first balanced-looking JSON object.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidate = stripped[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def normalize_mcq_answer(raw_answer: str, options: Sequence[Option]) -> str:
    labels = [o.label for o in options]
    label_set = {x.upper(): x for x in labels}
    text = str(raw_answer or "").strip()

    # Direct label match, e.g. "B".
    if text.upper() in label_set:
        return label_set[text.upper()]

    # Common patterns, e.g. "Answer: B", "Option (C)".
    match = re.search(r"(?:answer|option|choice)?\s*[:：\(\[]?\s*([A-Z])\s*[\)\].:]?", text, flags=re.IGNORECASE)
    if match and match.group(1).upper() in label_set:
        return label_set[match.group(1).upper()]

    # Match option text.
    lowered = text.lower()
    for opt in options:
        if opt.text and opt.text.lower() in lowered:
            return opt.label

    # Retain the submitted parser fallback while using the comparative prompt.
    return labels[0]


def parse_model_output(case: Case, raw_text: str) -> Prediction:
    obj = extract_json_object(raw_text) or {}
    reasoning = str(obj.get("reasoning_trace", obj.get("rationale", ""))).strip()
    answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()

    if not answer:
        # Text fallback for non-JSON outputs.
        match = re.search(r"(?:final answer|answer)\s*[:：]\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL)
        answer = match.group(1).strip().splitlines()[0] if match else raw_text.strip()
    if not reasoning:
        reasoning = raw_text.strip() or "No reasoning trace was generated."

    if case.task_type == "mcq":
        answer = normalize_mcq_answer(answer, case.options)
    else:
        # Open-ended answer must be a non-empty concise string.
        if not answer:
            answer = "Unable to determine from the available visual evidence."

    return Prediction(case_id=case.case_id, task_type=case.task_type, answer=answer, reasoning_trace=reasoning)


# ---------------------------------------------------------------------------
# MODEL LOADING AND INFERENCE
# ---------------------------------------------------------------------------
def load_qwen25vl(model_path: Path):
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"Qwen2.5-VL checkpoint not found at {model_path}. "
            "Place the model files in /opt/app/models/Qwen2.5-VL or set MEDREASON_MODEL_PATH."
        )

    try:
        import torch
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("Real model mode requires torch and transformers to be installed.") from exc

    # Transformers has used different auto classes across versions/model families.
    model = None
    load_errors: List[str] = []
    for class_name in ["Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]:
        try:
            transformers_mod = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(transformers_mod, class_name)
            model = model_cls.from_pretrained(
                str(model_path),
                torch_dtype="auto" if DTYPE == "auto" else getattr(torch, DTYPE),
                device_map="auto",
                trust_remote_code=True,
            )
            break
        except Exception as exc:  # noqa: BLE001 - collect fallback errors.
            load_errors.append(f"{class_name}: {exc}")
    if model is None:
        raise RuntimeError("Could not load Qwen2.5-VL model. Tried:\n" + "\n".join(load_errors))

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    return processor, model


def make_qwen_messages(case: Case) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for p in case.image_paths:
        content.append({"type": "image", "image": str(p)})
    content.append({"type": "text", "text": build_text_prompt(case)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def generate_with_qwen(case: Case, processor: Any, model: Any) -> str:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError("Real Qwen2.5-VL mode requires qwen-vl-utils and torch.") from exc

    messages = make_qwen_messages(case)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    do_sample = TEMPERATURE > 0
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": TEMPERATURE, "top_p": TOP_P})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # Remove prompt tokens before decoding.
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


# ---------------------------------------------------------------------------
# BASELINE SYSTEMS
# ---------------------------------------------------------------------------
def smoke_predict(case: Case) -> Prediction:
    if case.task_type == "mcq":
        return Prediction(
            case_id=case.case_id,
            task_type=case.task_type,
            answer=case.options[0].label,
            reasoning_trace="Smoke-test baseline: selected the first available option without model inference.",
            confidence=0.0,
        )
    return Prediction(
        case_id=case.case_id,
        task_type=case.task_type,
        answer="Smoke-test answer. Replace this mode with a real MedReason system for submission.",
        reasoning_trace="Smoke-test baseline: no medical image inference was performed.",
        confidence=0.0,
    )


def run_baseline(cases: Sequence[Case]) -> List[Prediction]:
    if SMOKE_TEST:
        print("[MedReason] running smoke-test baseline; no model will be loaded", flush=True)
        return [smoke_predict(case) for case in cases]

    print(f"[MedReason] loading Qwen2.5-VL baseline from {MODEL_PATH}", flush=True)
    processor, model = load_qwen25vl(MODEL_PATH)
    predictions: List[Prediction] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[MedReason] case {idx}/{len(cases)}: {case.case_id}", flush=True)
        raw = generate_with_qwen(case, processor=processor, model=model)
        pred = parse_model_output(case, raw)
        predictions.append(pred)
    return predictions


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
def validate_predictions(predictions: Sequence[Prediction], cases: Sequence[Case]) -> None:
    if len(predictions) != len(cases):
        raise ValueError(f"Expected {len(cases)} predictions, got {len(predictions)}")
    case_by_id = {case.case_id: case for case in cases}
    seen = set()
    for pred in predictions:
        if pred.case_id in seen:
            raise ValueError(f"duplicate prediction for case_id={pred.case_id}")
        seen.add(pred.case_id)
        if pred.case_id not in case_by_id:
            raise ValueError(f"prediction contains unknown case_id={pred.case_id}")
        case = case_by_id[pred.case_id]
        if canonical_task_type(pred.task_type) != case.task_type:
            raise ValueError(f"task_type mismatch for {pred.case_id}")
        if not str(pred.answer).strip():
            raise ValueError(f"empty answer for {pred.case_id}")
        if case.task_type == "open" and not str(pred.reasoning_trace).strip():
            raise ValueError(f"open-ended case {pred.case_id} requires reasoning_trace")
        if case.task_type == "mcq":
            labels = {o.label for o in case.options}
            if pred.answer not in labels:
                raise ValueError(f"MCQ answer for {pred.case_id} must be one of {sorted(labels)}, got {pred.answer!r}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[MedReason] input_dir={INPUT_DIR}", flush=True)
    print(f"[MedReason] cases_file={CASES_FILE}", flush=True)
    print(f"[MedReason] output_file={OUT_FILE}", flush=True)
    print(f"[MedReason] smoke_test={SMOKE_TEST}", flush=True)

    cases = load_cases(CASES_FILE)
    print(f"[MedReason] loaded {len(cases)} cases", flush=True)

    predictions = run_baseline(cases)
    validate_predictions(predictions, cases)
    write_results(predictions, OUT_FILE)
    print(f"[MedReason] saved results -> {OUT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - fail loudly in container logs.
        print(f"[MedReason][ERROR] {exc}", file=sys.stderr, flush=True)
        raise
