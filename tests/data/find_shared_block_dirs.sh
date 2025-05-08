#!/bin/bash

# Description: Find 5-10 directories with shared block IDs among directories named 'label_<label id>'

# Usage: Run this script in the parent directory containing 'label_<label id>' directories.

# Step 1: Collect block IDs from each directory

declare -A block_map

prefix="/nrs/turaga/jakob/autoproof_data/h01_test0_parents_v0/points/label_360287970189827*"
echo "Collecting block IDs from directories..."
for label_dir in $prefix; do
    if [ -d "$label_dir" ]; then
        for block_file in "$label_dir"/block_*; do
            if [ -f "$block_file" ]; then
                block_id=$(basename "$block_file" | cut -d'_' -f2)
                block_map["$block_id"]+="$label_dir ";
            fi
        done
    fi
done

# Step 2: Find shared block IDs and their corresponding directories

echo "Finding shared block IDs..."
for block_id in "${!block_map[@]}"; do
    dirs=(${block_map[$block_id]})
    if [ ${#dirs[@]} -ge 5 ]; then
        echo "Block ID: $block_id"
        echo "Shared by directories: ${dirs[@]}"
        echo "------------------------------------"
    fi
done

echo "Script completed."
