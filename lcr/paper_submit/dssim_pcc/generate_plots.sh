#!/bin/bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 RUN_DIRECTORY" >&2
    exit 2
fi

workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
paper_submit_dir="$(cd "$workflow_dir/.." && pwd)"
config_dir="$workflow_dir/configs"
cache_dir="$workflow_dir/precomputed_labels"
run_dir="$(cd "$1" && pwd)"
results_dir="$run_dir/model_results"
single_plots="$run_dir/plots_single"
multi_plots="$run_dir/plots_multi"
table_dir="$run_dir/label_tables"

if [[ ! -d "$results_dir" ]]; then
    echo "Model-results directory does not exist: $results_dir" >&2
    exit 1
fi

mkdir -p "$single_plots" "$multi_plots" "$table_dir"

python -u "$paper_submit_dir/mainplots1.py" \
    --num_configs 47 \
    --config_prefix "$config_dir/rotated_config_" \
    --results-dir "$results_dir" \
    --output-dir "$single_plots" \
    --skip-missing

python -u "$paper_submit_dir/mainplots1.py" \
    --num_configs 47 \
    --config_prefix "$config_dir/multi_config_" \
    --results-dir "$results_dir" \
    --output-dir "$multi_plots" \
    --skip-missing

python -u "$paper_submit_dir/label_distribution_table.py" \
    --cache-dir "$cache_dir" \
    --metrics dssim pcc \
    --timesteps 2000 \
    --subset first-1600 \
    --output "$table_dir/single_variable_labels_first1600.csv"

python -u "$paper_submit_dir/label_distribution_table.py" \
    --cache-dir "$cache_dir" \
    --metrics dssim pcc \
    --timesteps 1600 \
    --subset all \
    --output "$table_dir/multi_variable_labels_1600.csv"

# The modeling programs already create decision-tree and confusion-matrix
# figures. Generate the additional per-variable stacked-bar products from
# each single-variable CNN/RF result pair.
detail_dir="$single_plots/details"
detail_logs="$run_dir/logs/plot_details"
mkdir -p "$detail_dir" "$detail_logs"

shopt -s nullglob
for config in "$config_dir"/rotated_config_*.json; do
    if [[ "$(jq -r '.ModelType' "$config")" != "cnn" ]]; then
        continue
    fi
    config_name="$(basename "$config" .json)"
    if ! python -u "$paper_submit_dir/main_plots.py" \
        --json "$config" \
        --model cnn \
        --labelsonly True \
        --results-dir "$results_dir" \
        --output-dir "$detail_dir" \
        > "$detail_logs/${config_name}.log" 2>&1
    then
        echo "Detailed plots skipped for $config_name; see $detail_logs/${config_name}.log" >&2
    fi
done

echo "Single-variable plots: $single_plots"
echo "Multi-variable plots: $multi_plots"
echo "Label tables: $table_dir"

