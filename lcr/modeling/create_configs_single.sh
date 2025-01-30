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

# Loop through each rotation, one per group in the VarList
for ((i=0; i<${#var_list[@]}; i++)); do
    # Rotate entire list
    rotated_vars=("${var_list[@]:i}" "${var_list[@]:0:i}")

    # Select only the first element from the rotated list
    first_var="${rotated_vars[0]}"

    # Format the selected element as a nested JSON array
    rotated_var_list_json="[$first_var]"

    # Generate a new filename for this rotation
    output_file="$output_dir/rotated_config_$((i+1)).json"

    # Replace VarList in the JSON file
    jq --argjson new_var_list "$rotated_var_list_json" '.VarList = $new_var_list' "$original_file" > "$output_file"

    echo "Created $output_file with VarList containing: $rotated_var_list_json"
done
