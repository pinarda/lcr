#!/bin/bash

set -euo pipefail

paper_submit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workflow_dir="$paper_submit_dir/dssim_pcc"
config_dir="$workflow_dir/configs"
label_dir="$workflow_dir/precomputed_labels"

command -v jq >/dev/null || {
    echo "jq is required." >&2
    exit 1
}
command -v qsub >/dev/null || {
    echo "qsub is required; run this launcher on an NCAR login node." >&2
    exit 1
}

"$workflow_dir/create_configs.sh"

if [[ ! -d "$label_dir" ]]; then
    echo "Missing precomputed labels: $label_dir" >&2
    echo "Run the DSSIM+PCC label-precomputation stage first." >&2
    exit 1
fi

run_stamp="$(date +%Y-%m-%d_%H%M%S)"
sweep_dir="$workflow_dir/runs/single_variable_seeds_1_10_$run_stamp"
logs_dir="$sweep_dir/logs"
task_file="$sweep_dir/model_tasks.tsv"
mkdir -p "$logs_dir"

variables=(LHFLX pom_a1_SRF TAUX)
models=(cnn rf)

find_config() {
    local variable=$1
    local model=$2
    local candidate
    local match=""

    for candidate in "$config_dir"/rotated_config_*.json; do
        if [[ "$(jq -r '.VarList[0][0]' "$candidate")" == "$variable" &&
              "$(jq -r '.ModelType' "$candidate")" == "$model" ]]; then
            if [[ -n "$match" ]]; then
                echo "More than one $model config found for $variable" >&2
                return 1
            fi
            match=$candidate
        fi
    done
    if [[ -z "$match" ]]; then
        echo "No $model config found for $variable" >&2
        return 1
    fi
    printf '%s\n' "$match"
}

: > "$task_file"
for seed in {1..10}; do
    seed_dir="$sweep_dir/seed_$(printf '%02d' "$seed")"
    results_dir="$seed_dir/model_results"
    mkdir -p "$results_dir" "$seed_dir/stacked_bars" "$seed_dir/plot_logs"
    for variable in "${variables[@]}"; do
        for model in "${models[@]}"; do
            config_file="$(find_config "$variable" "$model")"
            printf '%d\t%s\t%s\n' \
                "$seed" "$config_file" "$results_dir" >> "$task_file"
        done
    done
done

task_count="$(wc -l < "$task_file" | tr -d ' ')"
model_job="$(qsub \
    -J "1-${task_count}%8" \
    -v "TASK_FILE=$task_file,PAPER_SUBMIT_DIR=$paper_submit_dir,PRECOMPUTED_LABEL_DIR=$label_dir" \
    -o "$logs_dir/" \
    "$paper_submit_dir/single_seed_sweep_model.pbs")"

plot_job="$(qsub \
    -W "depend=afterany:$model_job" \
    -v "PAPER_SUBMIT_DIR=$paper_submit_dir,SWEEP_DIR=$sweep_dir" \
    -o "$logs_dir/generate_stacked_bars.out" \
    "$paper_submit_dir/single_seed_sweep_plots.pbs")"

cat > "$sweep_dir/run_manifest.txt" <<EOF
created=$run_stamp
metrics=dssim,pcc
seeds=1-10
variables=LHFLX,pom_a1_SRF,TAUX
models=cnn,rf
task_count=$task_count
maximum_concurrent_jobs=8
model_job=$model_job
plot_job=$plot_job
precomputed_labels=$label_dir
sweep_directory=$sweep_dir
EOF

printf '%s\n' "$sweep_dir" > "$workflow_dir/latest_single_seed_sweep.txt"

echo "Submitted single-variable seed sweep"
echo "  Modeling array: $model_job ($task_count tasks, at most 8 running)"
echo "  Plot job: $plot_job"
echo "  Results: $sweep_dir"
