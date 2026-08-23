# Data access and local layout

This repository does not redistribute the MedReason 2026 participant data. Request access through the official challenge and comply with its data-use terms.

After extraction, set `MEDREASON_ROOT` to the repository root and place the participant package in `data/medreason2026_train/`. The preparation scripts expect the released JSON records and referenced images to remain together. They write derived manifests beneath `data/derived/` and case-level predictions beneath `outputs/`; both locations are excluded by `.gitignore`.

The paper used only participant-accessible records. Public-leaderboard cases were excluded before sampling, and no hidden challenge cases were used. The primary and holdout samples are disjoint and are reproduced by the fixed selection keys in `configs/experiment_manifest.json`.

Do not commit participant records, images, gold labels, raw generations, or derived case identifiers to a public fork.
