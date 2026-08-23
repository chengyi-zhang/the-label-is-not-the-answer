# Contributing

Bug reports and reproducibility improvements are welcome. Please do not attach MedReason participant records, medical images, model checkpoints, case-level outputs, credentials, or private infrastructure details to issues or pull requests.

Before submitting a change, run:

```bash
python -m compileall -q src scripts figures tests
python tests/test_release_integrity.py
```

Changes to prompts, parsers, sampling, or statistical procedures should state whether they reproduce the published configuration or define a new experiment.
