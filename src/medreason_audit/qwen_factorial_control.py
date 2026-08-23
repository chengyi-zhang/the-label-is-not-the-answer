#!/usr/bin/env python3
from __future__ import annotations
"""Run one cell of the MCQ schema-cue by option-prefix factorial control.

The unchanged submitted prompt is the cue-present/original-prefix cell. This
wrapper selects another cell with ``MEDREASON_ABLATION_MODE`` while keeping
model loading, decoding, parsing, and non-MCQ behavior fixed.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path


BASE_PATH = Path(__file__).with_name("qwen_submitted.py")
spec = importlib.util.spec_from_file_location("medreason_submitted_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)


MODE = os.environ.get("MEDREASON_ABLATION_MODE", "no_anchor_raw").strip().lower()
VALID_MODES = {"no_anchor_raw", "anchor_clean", "no_anchor_clean"}
if MODE not in VALID_MODES:
    raise ValueError(f"MEDREASON_ABLATION_MODE must be one of {sorted(VALID_MODES)}, got {MODE!r}")


MCQ_NO_ANCHOR = """Task: closed-ended multiple-choice medical visual question answering.

Question:
{question}

Options:
{options}

Instructions:
1. Inspect the image evidence relevant to the question.
2. Choose exactly one option label from the provided options.
3. The answer field must contain only the selected label.
4. The reasoning_trace should be concise and image-grounded.

Return only JSON:
{{
  "reasoning_trace": "brief visual rationale",
  "answer": "selected option label"
}}"""


def clean_option_text(text: str) -> str:
    return re.sub(r"^\s*[A-Z]\s*[\)\].:]\s*", "", str(text), count=1).strip()


def options_for_prompt(case):
    if "clean" not in MODE:
        return case.options
    return tuple(base.Option(label=option.label, text=clean_option_text(option.text)) for option in case.options)


def build_text_prompt(case):
    if case.task_type != "mcq":
        return base.OPEN_INSTRUCTION.format(question=case.question)
    instruction = base.MCQ_INSTRUCTION if MODE == "anchor_clean" else MCQ_NO_ANCHOR
    return instruction.format(question=case.question, options=base.format_options(options_for_prompt(case)))


base.build_text_prompt = build_text_prompt


if __name__ == "__main__":
    raise SystemExit(base.main())
