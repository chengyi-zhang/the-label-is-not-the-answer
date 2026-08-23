#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


BASE_PATH = Path(__file__).with_name("qwen_comparative.py")
spec = importlib.util.spec_from_file_location("medreason_prompt_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

NEUTRAL_MCQ_INSTRUCTION = """Task: closed-ended medical visual question answering.

Question:
{question}

Candidate answers:
{options}

Inspect the image, compare the candidate answers, and select the single candidate best supported by the visual evidence. Pay attention to negation, location, extent, laterality, depth, and temporal wording.

Return only valid JSON using this schema:
{{
  "reasoning_trace": "brief image-grounded rationale",
  "answer": "selected candidate label"
}}"""


def clean_option_text(text: str) -> str:
    return re.sub(r"^\s*[A-E]\s*[\)\].:\-]\s*", "", str(text), count=1)


def clean_options(options):
    return tuple(base.Option(label=option.label, text=clean_option_text(option.text)) for option in options)


def build_text_prompt(case):
    if case.task_type == "mcq":
        return NEUTRAL_MCQ_INSTRUCTION.format(
            question=case.question,
            options=base.format_options(clean_options(case.options)),
        )
    return base.OPEN_INSTRUCTION.format(question=case.question)


base.build_text_prompt = build_text_prompt
base.SUBMISSION_NAME = "MedReason Cue-free-prompt predictions"

if __name__ == "__main__":
    raise SystemExit(base.main())
