# Reproducibility workflow

## 1. Verify the environment

Install `requirements.txt`, install the package in editable mode, and expose the official model directories through `MEDREASON_MODEL_PATH` and `MEDREASON_INTERNVL_MODEL`.

## 2. Prepare samples

Run `prepare_primary_sample.py` to construct the label-balanced 600-case sample. The script selects 120 five-option MCQs from each reference-label stratum by the SHA-256 digest of the published selection key and case ID. Run `prepare_replication_sample.py` for the disjoint same-source holdout.

The selection keys are public, non-secret byte strings encoded in hexadecimal so their exact bytes are interpreted identically across platforms.

## 3. Generate predictions

Set `MEDREASON_INPUT_DIR` to a prepared condition directory and `MEDREASON_OUTPUT_FILE` to an untracked destination. Execute one of the modules under `src/medreason_audit/`. Repeat with the same checkpoint and decoding configuration for every condition being compared.

## 4. Validate outputs

Run `validate_output.py` on every result and `validate_cyclic_audit.py` on each complete five-assignment audit. Validation checks case coverage, duplicate IDs, output labels, task types, and the expected reassignment structure.

## 5. Analyse

The analysis scripts calculate accuracy after mapping labels back to option content, output-agreement measures, completed-example contingency tables, paired differences, and replication summaries. `verify_statistics.py` recomputes the exact case-level sign-flip tests and 200,000-resample case-cluster bootstrap intervals used in the paper.

## 6. Compare aggregate records

Expected aggregate outputs are stored under `results/`. Case-level predictions are intentionally absent. Exact floating-point formatting may vary by platform, but counts, percentages, confidence intervals, and test decisions should agree.

## Statistical procedures

- Paired accuracy contrasts: exact case-level sign-flip test under sign exchangeability.
- Confidence intervals: 200,000 case-cluster bootstrap resamples.
- Paired binary comparisons: exact McNemar tests with Holm adjustment where reported.
- Proportions: Wilson intervals.
- Cue-response association: Cramer's V with case-wise permutation of cue assignments.
