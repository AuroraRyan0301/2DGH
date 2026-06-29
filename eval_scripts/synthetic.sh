#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: $0 <nerf_synthetic_case_name> <test_case_path> <output_path> <ground_truth_mesh_path>"
    exit 1
fi

CASE="$1"
CASE_PATH="$2"
OUTPUT_PATH="$3"
GT_PATH="$4"

python train.py -s /${CASE_PATH}/${CASE} \
    -m /${OUTPUT_PATH}/${CASE}_orig_rb \
    --iterations 30000 \
    --lambda_dist 100 \
    -r 2 \
    --use_original_rasterizer \
    --white_background \
    --hg_opt_flag \
    --random_background

python train.py -s /${CASE_PATH}/${CASE} \
    -m /${OUTPUT_PATH}/${CASE}_gh_rb \
    --iterations 30000 \
    --lambda_dist 100 \
    -r 2 \
    --use_gh_rasterizer \
    --white_background \
    --gh_lr 0.001 \
    --hg_opt_flag \
    --random_background

# python train.py -s /${CASE_PATH}/${CASE} \
#     -m /${OUTPUT_PATH}/${CASE}_gss_rb \
#     --iterations 30000 \
#     --lambda_dist 100 \
#     -r 2 \
#     --use_gss_rasterizer \
#     --white_background \
#     --hg_opt_flag \
#     --random_background

python render.py -s /${CASE_PATH}/${CASE} \
    -m /${OUTPUT_PATH}/${CASE}_orig_rb \
    --iteration 30000 \
    -r 2 \
    --use_original_rasterizer \
    --hg_opt_flag \
    --white_background

python render.py -s /${CASE_PATH}/${CASE} \
    -m /${OUTPUT_PATH}/${CASE}_gh_rb \
    --iteration 30000 \
    -r 2 \
    --use_gh_rasterizer \
    --hg_opt_flag \
    --white_background

# python render.py -s /${CASE_PATH}/${CASE} \
#     -m /${OUTPUT_PATH}/${CASE}_gss_rb \
#     --iteration 30000 \
#     -r 2 \
#     --use_gss_rasterizer \
#     --hg_opt_flag \
#     --white_background

if [ ! -d "./ns_cd/" ]; then
    mkdir -p "./ns_cd/"
fi

file="./ns_cd/${CASE}.txt"
cat <<EOF > "$file"
Original:
$(python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_orig_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj)
Hermite:
$(python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_gh_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj)
Sinusodial:
$(python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_gss_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj)
EOF

# echo "Original:"
# python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_orig_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj
# echo "Hermite:"
# python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_gh_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj
# echo "Sinusodial:"
# python scripts/cd_helper.py /${OUTPUT_PATH}/${CASE}_gss_rb/train/ours_30000/fuse_post.ply /${GT_PATH}/${CASE}.obj

