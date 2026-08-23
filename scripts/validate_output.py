#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"[validate_output][ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    answers = payload.get("answers")
    if not isinstance(answers, list):
        fail("results.json must contain an answers list")
    seen = set()
    for i, item in enumerate(answers):
        if not isinstance(item, dict):
            fail(f"answers[{i}] must be an object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            fail(f"answers[{i}].case_id must be a non-empty string")
        if case_id in seen:
            fail(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        task_type = item.get("task_type")
        if task_type not in {"mcq", "open"}:
            fail(f"answers[{i}].task_type must be mcq or open")
        if not isinstance(item.get("answer"), str) or not item.get("answer").strip():
            fail(f"answers[{i}].answer must be a non-empty string")
        if task_type == "open" and (not isinstance(item.get("reasoning_trace"), str) or not item.get("reasoning_trace").strip()):
            fail(f"answers[{i}].reasoning_trace is required for open-ended cases")
    print(f"Validated {len(answers)} predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
