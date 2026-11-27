# Archived experiments

Scripts from earlier rounds whose ideas were tested and either dropped (no CV gain beyond noise) or replaced.
They are kept so every row of `EXPERIMENTS.md` can be traced to code. Run them as
`python -m experiments.archive.<name>` from the repo root; some read outputs written by other scripts.

| script | what it tested | outcome (see EXPERIMENTS.md) |
|---|---|---|
| variance_models.py | regularized / bagged LightGBM and XGBoost to cut model variance | superseded by shallow trees |
| te_tuning.py | target-encoding smoothing and column choice | settings folded into src/fraud/features.py |
| digit_keys.py | first screen of digit keys (last digit, decimals) | 12 keys kept (eivd), later 42 keys |
| original_data.py | adding the original 20k-row dataset to training folds | no gain |
| feature_ideas_b.py | derived features: vel_ex, mr_ex, binned amount TE, category x amount TE | vel_ex kept (+0.0003, the "v" in eivd); others within noise |
| shallow_trees.py | LightGBM num_leaves 3/4/15, XGBoost depth 2/3 | num_leaves 3 / depth 2 kept |
| round4_trees.py | max_bin 1023, row weights, scale_pos_weight, specialists on hard rows | max_bin 1023 kept (gain within noise); rest dropped |
| round6_trees.py | min_child_samples, learning rate for num_leaves 3 | within noise |
| models_v2_extra.py | CatBoost depth 6, RealMLP (pytabkit) | CatBoost +0.00002 in blend; RealMLP too slow (~9 h) and weaker in a smoke test |
| round4_nn.py | MLP with piecewise-linear inputs, MLP trained on hard rows only | worse (-0.002 to -0.007) |
| round4_eval.py | blend evaluation for round 4 | replaced by experiments/round7_stack.py |
| make_submission.py, make_blend_submission.py | submissions 1 and 2 | replaced by make_blend_submission_v3.py / make_stack_submission.py |
