#!/usr/bin/env python3
"""Vary only the concrete answer label in the submitted prompt example."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


BASE_PATH = Path(__file__).with_name("qwen_submitted.py")
spec = importlib.util.spec_from_file_location("medreason_anchor_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

anchor = os.environ.get("MEDREASON_ANCHOR_LABEL", "A").strip().upper()
if anchor not in "ABCDE":
    raise ValueError(f"MEDREASON_ANCHOR_LABEL must be A-E, got {anchor!r}")

base.MCQ_INSTRUCTION = f'''Task: closed-ended multiple-choice medical visual question answering.

Question:
{{question}}

Options:
{{options}}

Instructions:
1. Inspect the image evidence relevant to the question.
2. Choose exactly one option label from the provided options.
3. The answer field must contain only the selected label, for example "{anchor}".
4. The reasoning_trace should be concise and image-grounded.

Return only JSON:
{{{{
  "reasoning_trace": "brief visual rationale",
  "answer": "{anchor}"
}}}}'''


if __name__ == "__main__":
    raise SystemExit(base.main())
