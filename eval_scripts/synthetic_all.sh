#!/bin/bash

# Example: /path/to/nerf_synthetic/ /path/to/output/ /path/to/obj/

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <test_case_path> <output_path> <ground_truth_mesh_path>"
    exit 1
fi

CASE_PATH="$1"
OUTPUT_PATH="$2"
GT_PATH="$3"

ns_cases="chair drums ficus hotdog lego materials mic ship"

script_path=$(dirname "$0")
script_path=$(cd "$script_path" && pwd)

for case in $ns_cases; do
    bash $script_path/synthetic.sh $case $CASE_PATH $OUTPUT_PATH $GT_PATH
done
