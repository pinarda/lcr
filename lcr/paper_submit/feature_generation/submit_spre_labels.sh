#!/bin/bash
set -euo pipefail

cd /glade/derecho/scratch/apinard/lcr/lcr/paper_submit

variable_start="${VARIABLE_START:-1}"
variable_end="${VARIABLE_END:-43}"
timesteps="${TIMESTEPS:-2000}"
max_active_variables="${MAX_ACTIVE_VARIABLES:-8}"

for value in "$variable_start" "$variable_end" "$timesteps" "$max_active_variables"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Variable bounds, TIMESTEPS, and MAX_ACTIVE_VARIABLES must be positive integers" >&2
        exit 2
    fi
done
if (( variable_start < 1 || variable_end > 43 || variable_start > variable_end )); then
    echo "Variable range must satisfy 1 <= VARIABLE_START <= VARIABLE_END <= 43" >&2
    exit 2
fi
if (( timesteps != 2000 )); then
    echo "This workflow requires TIMESTEPS=2000" >&2
    exit 2
fi
if (( max_active_variables < 1 )); then
    echo "MAX_ACTIVE_VARIABLES must be at least 1" >&2
    exit 2
fi

declare -a slot_tails=()
compression_count=9

printf 'Submitting SPRE workflow for variables %d-%d; at most %d active variable chains\n' \
    "$variable_start" "$variable_end" "$max_active_variables"

for variable_index in $(seq "$variable_start" "$variable_end"); do
    slot=$(( (variable_index - variable_start) % max_active_variables ))
    previous_job="${slot_tails[$slot]:-}"

    for compression_index in $(seq 1 "$compression_count"); do
        dependency_options=()
        if [[ -n "$previous_job" ]]; then
            dependency_options=(-W "depend=afterok:${previous_job}")
        fi
        previous_job=$(qsub \
            -N "spre_v${variable_index}_c${compression_index}" \
            -v "VARIABLE_INDEX=${variable_index},COMPRESSION_INDEX=${compression_index},TIMESTEPS=${timesteps}" \
            "${dependency_options[@]}" \
            feature_generation/run_spre_metric_generation.pbs)
    done

    combine_job=$(qsub \
        -N "spre_combine_v${variable_index}" \
        -v "VARIABLE_INDEX=${variable_index},TIMESTEPS=${timesteps}" \
        -W "depend=afterok:${previous_job}" \
        feature_generation/combine_spre_inputs.pbs)
    slot_tails[$slot]="$combine_job"

    printf 'Variable %d: final metric %s; combine %s; slot %d\n' \
        "$variable_index" "$previous_job" "$combine_job" "$((slot + 1))"
done

if (( variable_start == 1 && variable_end == 43 )); then
    final_dependencies=$(IFS=:; echo "${slot_tails[*]}")
    label_job=$(qsub \
        -N spre_labels_table \
        -W "depend=afterok:${final_dependencies}" \
        precompute_spre_labels.pbs)
    printf 'Final SPRE labels and table: %s\n' "$label_job"
else
    echo "Partial variable range submitted; final label/table job was not submitted."
fi
