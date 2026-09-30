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
  '["FSNSC"]'
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
# Configs 1-47 are CNN; configs 48-94 are RF.
model_types=("cnn" "rf")

for model_type in "${model_types[@]}"; do
    if [[ "$model_type" == "cnn" ]]; then
        offset=0
    else
        offset=${#var_list[@]}
    fi

    for ((i=0; i<${#var_list[@]}; i++)); do
        first_var="${var_list[i]}"
        rotated_var_list_json="[$first_var]"
        output_file="$output_dir/rotated_config_$((offset+i+1)).json"

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
