#!/bin/bash
set -euo pipefail

cd /glade/derecho/scratch/apinard/lcr/lcr/paper_submit

variable_start="${VARIABLE_START:-1}"
variable_end="${VARIABLE_END:-46}"
timesteps="${TIMESTEPS:-2000}"
smoke_test="${SMOKE_TEST:-0}"

if ! [[ "$variable_start" =~ ^[0-9]+$ ]] || \
   ! [[ "$variable_end" =~ ^[0-9]+$ ]] || \
   ! [[ "$timesteps" =~ ^[0-9]+$ ]]; then
    echo "VARIABLE_START, VARIABLE_END, and TIMESTEPS must be positive integers" >&2
    exit 2
fi
if (( variable_start < 1 || variable_end > 46 || variable_start > variable_end )); then
    echo "Variable range must satisfy 1 <= VARIABLE_START <= VARIABLE_END <= 46" >&2
    exit 2
fi
if (( timesteps < 1 )); then
    echo "TIMESTEPS must be at least 1" >&2
    exit 2
fi
if [[ "$smoke_test" != "0" && "$smoke_test" != "1" ]]; then
    echo "SMOKE_TEST must be 0 or 1" >&2
    exit 2
fi

worker_qsub_options=()
if [[ "$smoke_test" == "1" ]]; then
    worker_qsub_options=(
        -q develop@desched1
        -l select=1:ncpus=16:mem=128GB
        -l walltime=01:00:00
    )
fi

printf 'Submitting variables %d-%d with %d timesteps (smoke test: %s)\n' \
    "$variable_start" "$variable_end" "$timesteps" "$smoke_test"

for variable_index in $(seq "$variable_start" "$variable_end"); do
    previous_job=$(qsub \
        "${worker_qsub_options[@]}" \
        -N "feat_v${variable_index}_t${timesteps}" \
        -v "VARIABLE_INDEX=${variable_index},TIMESTEPS=${timesteps}" \
        feature_generation/run_feature_generation.pbs)
    feature_job=$previous_job

    for compression_index in $(seq 1 10); do
        previous_job=$(qsub \
            "${worker_qsub_options[@]}" \
            -N "metric_v${variable_index}_c${compression_index}_t${timesteps}" \
            -v "VARIABLE_INDEX=${variable_index},COMPRESSION_INDEX=${compression_index},TIMESTEPS=${timesteps}" \
            -W "depend=afterok:${previous_job}" \
            feature_generation/run_metric_generation.pbs)
    done

    combine_job=$(qsub \
        -N "combine_v${variable_index}_t${timesteps}" \
        -v "VARIABLE_INDEX=${variable_index},TIMESTEPS=${timesteps}" \
        -W "depend=afterok:${previous_job}" \
        feature_generation/combine_rf_inputs.pbs)

    printf 'Variable %d: feature %s; final metric %s; combine %s\n' \
        "$variable_index" "$feature_job" "$previous_job" "$combine_job"
done
