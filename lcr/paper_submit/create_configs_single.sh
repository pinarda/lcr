#!/bin/bash

# Path to the original file
original_file="config_casper_test.json" # Original JSON file

# VarList structure
var_list=(
  '["TREFHTMX"]'
  '["TS"]'
  '["LHFLX"]'
  '["PRECSL"]'
  '["PRECT"]'
  '["PSL"]'
  '["Q200"]'
  '["Q500"]'
  '["Q850"]'
  '["SHFLX"]'
  '["T200"]'
  '["T500"]'
  '["T850"]'
  '["TAUX"]'
  '["TAUY"]'
  '["U010"]'
  '["FLNS"]'
  '["FLNSC"]'
  '["FSNS"]'
  '["PRECL"]'
  '["PRECSC"]'
  '["QBOT"]'
  '["T010"]'
  '["TMQ"]'
  '["TREFHT"]'
  '["TREFHTMN"]'
  '["U200"]'
  '["U500"]'
  '["U850"]'
  '["UBOT"]'
  '["V200"]'
  '["V500"]'
  '["V850"]'
  '["VBOT"]'
  '["WSPDSRFAV"]'
  '["Z050"]'
  '["Z500"]'
  '["bc_a1_SRF"]'
  '["dst_a1_SRF"]'
  '["dst_a3_SRF"]'
  '["pom_a1_SRF"]'
  '["so4_a1_SRF"]'
  '["so4_a2_SRF"]'
  '["so4_a3_SRF"]'
  '["soa_a1_SRF"]'
  '["soa_a2_SRF"]'
)
#
#var_list=(
#  '["TREFHTMX"]'
#  '["TS"]'
#  '["LHFLX"]'
#  '["PRECSL"]'
#  '["PRECT"]'
#  '["PSL"]'
#  '["Q200"]'
#  '["Q500"]'
#  '["Q850"]'
#  '["SHFLX"]'
#  '["T200"]'
#  '["T500"]'
#  '["T850"]'
#  '["TAUX"]'
#  '["TAUY"]'
#  '["U010"]'
#  '["FLNS"]'
#)

# Output directory (current directory)
output_dir="."

# Generate one single-variable config per variable for CNN and RF.
# Preserve the original config IDs while omitting former FSNSC slots 20 and 67.
model_types=("cnn" "rf")

for model_type in "${model_types[@]}"; do
    if [[ "$model_type" == "cnn" ]]; then
        offset=0
    else
        offset=47
    fi

    for ((i=0; i<${#var_list[@]}; i++)); do
        first_var="${var_list[i]}"
        rotated_var_list_json="[$first_var]"
        legacy_slot=$((i < 19 ? i + 1 : i + 2))
        output_file="$output_dir/rotated_config_$((offset+legacy_slot)).json"

        if [[ -e "$output_file" ]]; then
            echo "Skipping existing $output_file"
            continue
        fi

        jq \
            --argjson new_var_list "$rotated_var_list_json" \
            --arg model_type "$model_type" \
            '.VarList = $new_var_list | .ModelType = $model_type | .Times = [2000]' \
            "$original_file" > "$output_file"

        echo "Created $output_file for $model_type with VarList: $rotated_var_list_json"
    done
done
