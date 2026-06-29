#!/bin/bash

# Set DTU evaluation directory
raster=original
dtu_eval_dir="/path/to/eval/DTU_eval_7_7/${raster}"
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

    python train_psnr.py -s $source \
        -m $eval_path \
        --iterations 50000 \
        --use_${raster}_rasterizer \
        --start_checkpoint $ckpt_path \
        --depth_ratio 1.0 \
        -r 2 \
        --lambda_dist 1000
done

