#!/bin/bash

# Path to the original file
original_file="config_casper_test.json" # Original JSON file

# Nested VarList structure
#var_list=(
#  '["TREFHTMX", "TS"]'
#  '["LHFLX"]'
#  '["PRECSL", "PRECT"]'
#  '["PSL"]'
#  '["Q200", "Q500", "Q850"]'
#  '["SHFLX"]'
#  '["T200", "T500", "T850"]'
#  '["TAUX", "TAUY"]'
#  '["U010"]'
#  '["FLNS"]'
#)

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

#  var_list = [
#    "TREFHTMX", "TS", "LHFLX", "PRECSL", "PRECT", "PSL", "Q200", "Q500", "Q850",
#    "SHFLX", "T200", "T500", "T850", "TAUX", "TAUY", "U010", "FLNS",
#
#]
#
#  var_list2 = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNSC",
#    "FSNS", "pom_a1_SRF",  "PRECL", "PRECSC", "QBOT",
#     "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "TMQ", "TREFHT",
#    "TREFHTMN", "U200", "U500", "U850", "UBOT",  "V200", "V500", "V850", "VBOT",  "WSPDSRFAV", "Z050", "Z500",




# Output directory (current directory)
output_dir="."

# Generate one leave-one-variable-out ordering per variable for CNN and RF.
# The first variable in each rotated list is the held-out test variable.
# Preserve the original config IDs while omitting former FSNSC slots 20 and 67.
model_types=("cnn" "rf")

for model_type in "${model_types[@]}"; do
    if [[ "$model_type" == "cnn" ]]; then
        offset=0
    else
        offset=47
    fi

    for ((i=0; i<${#var_list[@]}; i++)); do
        rotated_vars=("${var_list[@]:i}" "${var_list[@]:0:i}")
        rotated_var_list_json=$(printf '%s, ' "${rotated_vars[@]}" | sed 's/, $//')
        rotated_var_list_json="[$rotated_var_list_json]"
        legacy_slot=$((i < 19 ? i + 1 : i + 2))
        output_file="$output_dir/multi_config_$((offset+legacy_slot)).json"

        if [[ -e "$output_file" ]]; then
            echo "Skipping existing $output_file"
            continue
        fi

        jq \
            --argjson new_var_list "$rotated_var_list_json" \
            --arg model_type "$model_type" \
            '.VarList = $new_var_list
             | .ModelType = $model_type
             | .Times = [1600]
             | .CompDirs = ["zfp_p_16", "zfp_p_26"]
             | .Metric = ["dssim", "pcc", "spre"]' \
            "$original_file" > "$output_file"

        echo "Created $output_file for $model_type with held-out variable: ${rotated_vars[0]}"
    done
done
