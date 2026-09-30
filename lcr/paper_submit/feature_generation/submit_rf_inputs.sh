#!/bin/bash
set -euo pipefail

cd /glade/derecho/scratch/apinard/lcr/lcr/paper_submit

for variable_index in $(seq 1 47); do
    previous_job=$(qsub \
        -N "feat_v${variable_index}" \
        -v "VARIABLE_INDEX=${variable_index}" \
        feature_generation/run_feature_generation.pbs)
    feature_job=$previous_job

    for compression_index in $(seq 1 10); do
        previous_job=$(qsub \
            -N "metric_v${variable_index}_c${compression_index}" \
            -v "VARIABLE_INDEX=${variable_index},COMPRESSION_INDEX=${compression_index}" \
            -W "depend=afterok:${previous_job}" \
            feature_generation/run_metric_generation.pbs)
    done

    combine_job=$(qsub \
        -N "combine_v${variable_index}" \
        -v "VARIABLE_INDEX=${variable_index}" \
        -W "depend=afterok:${previous_job}" \
        feature_generation/combine_rf_inputs.pbs)

    printf 'Variable %d: feature %s; final metric %s; combine %s\n' \
        "$variable_index" "$feature_job" "$previous_job" "$combine_job"
done
