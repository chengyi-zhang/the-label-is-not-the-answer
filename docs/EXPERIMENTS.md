# Experiment index

## Completed-example intervention

`qwen_anchor_intervention.py` changes only the concrete answer label in the completed JSON example from A through E. Image, question, options, decoding, model weights, and parser are held fixed. `qwen_factorial_control.py` crosses cue presence with original or cleaned option prefixes.

The same intervention is implemented for InternVL3-9B in `internvl_anchor_intervention.py`. This is a cross-family check of the completed-example effect, not a repetition of the full cyclic audit.

## Cyclic label reassignment

`prepare_cyclic_audit.py` rotates A-E identifiers over fixed option texts. Predictions are mapped back to option content before accuracy and agreement are computed. The audit reports canonical accuracy, all-five semantic invariance, pairwise agreement, and the number of distinct selected options.

## Prompt conditions

- **Submitted**: the challenge prompt with a completed JSON example containing answer label A.
- **Comparative**: removes that completed example and asks the model to compare all candidates.
- **Cue-free**: also removes concrete examples and duplicated labels from option prefixes.

Comparative and Cue-free differ in more than one prompt component. Their accuracy difference is therefore an empirical comparison, not a single-factor mechanistic ablation. The separate factorial experiment isolates completed-example presence from prefix cleaning.

## Replication

The canonical comparison is repeated on a non-overlapping, same-source 600-case holdout. A Qwen2.5-VL-7B checkpoint tests model scale. InternVL3-9B tests the completed-example intervention only. These checks support the observed failure mode in the evaluated pipelines; they do not estimate its prevalence across medical benchmarks.
