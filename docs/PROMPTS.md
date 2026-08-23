# Prompt conditions

The exact executable prompt strings are the source of truth:

| Paper name | Implementation |
|---|---|
| Submitted | `src/medreason_audit/qwen_submitted.py` |
| Comparative | `src/medreason_audit/qwen_comparative.py` |
| Cue-free | `src/medreason_audit/qwen_cue_free.py` |
| Completed-example intervention | `src/medreason_audit/qwen_anchor_intervention.py` |
| Cue/prefix factorial control | `src/medreason_audit/qwen_factorial_control.py` |
| InternVL completed-example intervention | `src/medreason_audit/internvl_anchor_intervention.py` |

All conditions use deterministic greedy decoding. The Qwen pipeline permits at most 512 new tokens. The InternVL cross-family check uses its official dynamic 448-pixel preprocessing, at most six tiles plus a thumbnail, bfloat16 inference, and at most 192 new tokens.

The parser first attempts direct JSON extraction, then normalises a reported MCQ label against the labels present in that case. Parser behaviour is held fixed within each controlled comparison.
