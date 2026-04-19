#!/usr/bin/env bash
set -eo pipefail

# ------------------------------------------------------------
# Submit static architecture-grid search as a Slurm array job
# ------------------------------------------------------------

JOB_SCRIPT="${JOB_SCRIPT:-standardized_static_deephit_arch_grid_search.job}"
CONFIG_JSONL_PATH="${CONFIG_JSONL_PATH:-model_configs/deephit_configs_param_200k_500k_v1.jsonl}"
# MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

# Optional overrides passed through to sbatch as exported env vars
NOTEBOOK_PATH="${NOTEBOOK_PATH:-}"
WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-}"
NUM_EPOCHS_OVERRIDE="${NUM_EPOCHS_OVERRIDE:-}"
LEARNING_RATE_OVERRIDE="${LEARNING_RATE_OVERRIDE:-}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Job script not found: $JOB_SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_JSONL_PATH" ]]; then
  echo "Config JSONL not found: $CONFIG_JSONL_PATH" >&2
  exit 2
fi

NUM_CONFIGS=$(python - <<PY
import json
count = 0
with open("$CONFIG_JSONL_PATH", "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        json.loads(line)
        count += 1
print(count)
PY
)

if [[ "$NUM_CONFIGS" -le 0 ]]; then
  echo "No configs found in $CONFIG_JSONL_PATH" >&2
  exit 2
fi

TICKERS=("AAPL" "INTC" "LCID")
SEEDS=(4718 4819)

NUM_TICKERS=${#TICKERS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((NUM_CONFIGS * NUM_TICKERS * NUM_SEEDS))

ARRAY_MAX=$((TOTAL_TASKS - 1))
ARRAY_SPEC="0-${ARRAY_MAX}" #%${MAX_CONCURRENT}"

echo "Config file:      $CONFIG_JSONL_PATH"
echo "Configs found:    $NUM_CONFIGS"
echo "Array spec:       $ARRAY_SPEC"
echo "Job script:       $JOB_SCRIPT"
# echo "Max concurrent:   $MAX_CONCURRENT"

EXPORTS=(
  "ALL"
  "CONFIG_JSONL_PATH=$CONFIG_JSONL_PATH"
)

if [[ -n "$NOTEBOOK_PATH" ]]; then
  EXPORTS+=("NOTEBOOK_PATH=$NOTEBOOK_PATH")
fi
if [[ -n "$WANDB_MODE_OVERRIDE" ]]; then
  EXPORTS+=("WANDB_MODE_OVERRIDE=$WANDB_MODE_OVERRIDE")
fi
if [[ -n "$NUM_EPOCHS_OVERRIDE" ]]; then
  EXPORTS+=("NUM_EPOCHS_OVERRIDE=$NUM_EPOCHS_OVERRIDE")
fi
if [[ -n "$LEARNING_RATE_OVERRIDE" ]]; then
  EXPORTS+=("LEARNING_RATE_OVERRIDE=$LEARNING_RATE_OVERRIDE")
fi

EXPORT_ARG=$(IFS=,; echo "${EXPORTS[*]}")

SUBMIT_OUTPUT=$(sbatch \
  --array="$ARRAY_SPEC" \
  --export="$EXPORT_ARG" \
  "$JOB_SCRIPT"
)

echo "$SUBMIT_OUTPUT"

JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')
if [[ -n "${JOB_ID:-}" ]]; then
  echo "Submitted Slurm array job: $JOB_ID"
fi
