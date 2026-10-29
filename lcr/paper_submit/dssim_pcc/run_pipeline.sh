#!/bin/bash

set -euo pipefail

workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
paper_submit_dir="$(cd "$workflow_dir/.." && pwd)"

"$workflow_dir/create_configs.sh"

run_stamp="$(date +%Y-%m-%d_%H%M%S)"
run_dir="$workflow_dir/runs/$run_stamp"
results_dir="$run_dir/model_results"
logs_dir="$run_dir/logs"
config_list="$run_dir/modeling_configs.txt"

mkdir -p "$results_dir" "$logs_dir" "$workflow_dir/precomputed_labels"

shopt -s nullglob
config_files=(
    "$workflow_dir"/configs/rotated_config_*.json
    "$workflow_dir"/configs/multi_config_*.json
)
if (( ${#config_files[@]} == 0 )); then
    echo "No generated configs were found" >&2
    exit 1
fi
printf '%s\n' "${config_files[@]}" > "$config_list"

label_job=$(qsub \
    -v "DSSIM_PCC_DIR=$workflow_dir" \
    -o "$logs_dir/precompute_labels.out" \
    "$workflow_dir/precompute_labels.pbs")

config_count=${#config_files[@]}
model_job=$(qsub \
    -W "depend=afterok:$label_job" \
    -J "1-${config_count}%8" \
    -v "CONFIG_LIST=$config_list,MODEL_RESULTS_DIR=$results_dir,PAPER_SUBMIT_DIR=$paper_submit_dir,PRECOMPUTED_LABEL_DIR=$workflow_dir/precomputed_labels" \
    -o "$logs_dir/" \
    "$paper_submit_dir/modeling_array.pbs")

plot_job=$(qsub \
    -W "depend=afterany:$model_job" \
    -v "DSSIM_PCC_DIR=$workflow_dir,DSSIM_PCC_RUN_DIR=$run_dir" \
    -o "$logs_dir/generate_plots.out" \
    "$workflow_dir/generate_plots.pbs")

cat > "$run_dir/run_manifest.txt" <<EOF
metrics=dssim,pcc
created=$run_stamp
label_job=$label_job
model_job=$model_job
plot_job=$plot_job
config_count=$config_count
precomputed_labels=$workflow_dir/precomputed_labels
model_results=$results_dir
EOF
printf '%s\n' "$run_dir" > "$workflow_dir/latest_run.txt"

echo "Submitted DSSIM+PCC workflow"
echo "  Label job: $label_job"
echo "  Modeling array: $model_job ($config_count tasks, at most 8 running)"
echo "  Plot job: $plot_job (runs after the modeling array finishes)"
echo "  Run directory: $run_dir"

