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
  '["TREFHTMX"]'
)

#  var_list = [
#    "TREFHTMX", "TS", "LHFLX", "PRECSL", "PRECT", "PSL", "Q200", "Q500", "Q850",
#    "SHFLX", "T200", "T500", "T850", "TAUX", "TAUY", "U010", "FLNS",
#
#]
#
#  var_list2 = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNSC",
#    "FSNS", "FSNSC", "pom_a1_SRF",  "PRECL", "PRECSC", "QBOT",
#     "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "TMQ", "TREFHT",
#    "TREFHTMN", "U200", "U500", "U850", "UBOT",  "V200", "V500", "V850", "VBOT",  "WSPDSRFAV", "Z050", "Z500",




# Output directory (current directory)
output_dir="."

# Loop through each rotation, one per group in the nested VarList
for ((i=0; i<${#var_list[@]}; i++)); do
    # Rotate entire nested groups
    rotated_vars=("${var_list[@]:i}" "${var_list[@]:0:i}")

    # Format the rotated VarList as JSON
    rotated_var_list_json=$(printf '%s, ' "${rotated_vars[@]}" | sed 's/, $//')
    rotated_var_list_json="[$rotated_var_list_json]"

    # Generate a new filename for this rotation
    output_file="$output_dir/rotated_config_$((i+1)).json"

    # Replace VarList in the JSON file
    jq --argjson new_var_list "$rotated_var_list_json" '.VarList = $new_var_list' "$original_file" > "$output_file"

    echo "Created $output_file with rotated VarList group: ${rotated_vars[*]}"
done
