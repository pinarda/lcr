#!/bin/bash

set -euo pipefail

# Run from the directory containing this launcher and the config files.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

shopt -s nullglob
config_files=(
    "$script_dir"/rotated_config_*.json
    "$script_dir"/multi_config_*.json
)

if (( ${#config_files[@]} == 0 )); then
    echo "No rotated_config_*.json or multi_config_*.json files found" >&2
    exit 1
fi

# Give every subjob in this submission the same timestamped results directory.
run_stamp="$(date +%Y-%m-%d_%H%M%S)"
results_dir="$script_dir/data/model_results_${run_stamp}"
config_list="$results_dir/modeling_configs.txt"
mkdir -p "$results_dir"
printf '%s\n' "${config_files[@]}" > "$config_list"

config_count=${#config_files[@]}
job_id=$(qsub \
    -J "1-${config_count}%8" \
    -v "CONFIG_LIST=${config_list},MODEL_RESULTS_DIR=${results_dir},PAPER_SUBMIT_DIR=${script_dir}" \
    "$script_dir/modeling_array.pbs")

echo "Submitted $config_count modeling jobs as array $job_id"
echo "At most 8 array jobs will run at once"
echo "Results directory: $results_dir"
