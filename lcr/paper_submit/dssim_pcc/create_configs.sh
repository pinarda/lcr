#!/bin/bash

set -euo pipefail

workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
paper_submit_dir="$(cd "$workflow_dir/.." && pwd)"
config_dir="$workflow_dir/configs"

mkdir -p "$config_dir"
rm -f "$config_dir"/rotated_config_*.json "$config_dir"/multi_config_*.json

shopt -s nullglob
source_configs=(
    "$paper_submit_dir"/rotated_config_*.json
    "$paper_submit_dir"/multi_config_*.json
)

if (( ${#source_configs[@]} == 0 )); then
    echo "No source rotated_config_*.json or multi_config_*.json files found" >&2
    exit 1
fi

for source_config in "${source_configs[@]}"; do
    destination="$config_dir/$(basename "$source_config")"
    temporary="$destination.tmp"
    jq '.Metric = ["dssim", "pcc"]' "$source_config" > "$temporary"
    jq -e '.Metric == ["dssim", "pcc"]' "$temporary" > /dev/null
    mv "$temporary" "$destination"
done

config_count=$(find "$config_dir" -maxdepth 1 -type f \
    \( -name 'rotated_config_*.json' -o -name 'multi_config_*.json' \) | wc -l)

printf '%s\n' "${source_configs[@]##*/}" | sort > "$config_dir/config_manifest.txt"

echo "Created $config_count DSSIM+PCC configs in $config_dir"

