#!/usr/bin/env bash
# Rebuild the final submission (stack v5) from the raw competition data.
#   ./reproduce.sh                      all 7 fold seeds into outputs/ (many hours on a 10-core laptop)
#   FRAUD_OUTPUTS=outputs_repro ./reproduce.sh 42 repro_seed42
#                                       one fold seed into another folder (reproducibility check, ~2 h)
# Arguments: [comma-separated fold seeds] [submission name]. Needs data/raw/{train,test,sample_submission}.csv.
# Scripts refuse to overwrite existing predictions (pass --force to a script to replace them).
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="${1:-42,7,2026,11,123,31337,777}"
NAME="${2:-stack_v5_x3_7seeds_zone1}"
PY=.venv/bin/python
run() { echo "+ $*"; "$PY" -m "$@"; }

# 8 stack members, each on the same 5-fold splits per fold seed
run experiments.round4_gam ebm --seeds="$SEEDS"                                    # EBM (GAM with 10 pairs)
run experiments.round5_trees --only=lgb_l3_b1023_x3,xgb_d2_x3 --seeds="$SEEDS"     # shallow LightGBM / XGBoost
run experiments.nn_mlp --seeds="$SEEDS"                                            # MLP
run experiments.nn_mlp --digits --seeds="$SEEDS"                                   # MLP + 12 digit keys
run experiments.round7_glm glm_plain --seeds="$SEEDS"                              # logistic regression + 42 keys
run experiments.round7_trees --only=lgb_l3_k42_x3,cat_d2_k42 --seeds="$SEEDS"      # trees + 42 keys
# logistic-regression stacker + zone 1 -> submission/$NAME.csv (+ .manifest.json)
run experiments.make_stack_submission v4_x3 "$NAME" --seeds="$SEEDS"
