#!/bin/bash

# Set DTU evaluation directory
raster=gss
dtu_eval_dir="/path/to/eval/DTU_eval_717/${raster}"
parent_source=/path/to/dataset/DTU/DTU

# Iterate over all subdirectories
for dir in "$dtu_eval_dir"/*/; do
    # Extract the numeric part from the folder name
    scene_id=$(basename "$dir" | sed 's/[^0-9]*//g')

    # echo "Directory: $dir"

    # Build the ckpt path
    ckpt_path="$dir/chkpnt30000.pth"
    source="$parent_source/scan$scene_id"
    eval_path="$dtu_eval_dir/scan$scene_id"

    echo $source

    python render.py \
        --eval \
        --skip_train \
        --skip_mesh \
        --iteration 30000 \
        -s $source \
        -m $eval_path \
        --use_${raster}_rasterizer \
        --depth_ratio 1.0 \
        --num_cluster 1 \
        --voxel_size 0.004 \
        --sdf_trunc 0.016 \
        --depth_trunc 3.0

    python metrics.py -m $eval_path
done

