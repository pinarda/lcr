#!/bin/bash

# Path to the original file
original_file="config_casper_test.json" # Original JSON file

# Nested VarList structure
var_list=(
  '["TREFHTMX", "TS"]'
  '["LHFLX"]'
  '["PRECSL", "PRECT"]'
  '["PSL"]'
  '["Q200", "Q500", "Q850"]'
  '["SHFLX"]'
  '["T200", "T500", "T850"]'
  '["TAUX", "TAUY"]'
  '["U010"]'
  '["FLNS"]'
)

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
