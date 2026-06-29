python scripts/m360_eval.py \
    --mipnerf360 /path/to/dataset/MipNerf360 \
    --output_path /path/to/eval/m360_eval/original \
    --use_original_rasterizer

python scripts/m360_eval.py \
    --mipnerf360 /path/to/dataset/MipNerf360 \
    --output_path /path/to/eval/m360_eval/gss \
    --use_gss_rasterizer

python scripts/m360_eval.py \
    --mipnerf360 /path/to/dataset/MipNerf360 \
    --output_path /path/to/eval/m360_eval/ges \
    --use_ges_rasterizer

python scripts/m360_eval.py \
    --mipnerf360 /path/to/dataset/MipNerf360 \
    --output_path /path/to/eval/m360_eval/gh \
    --use_gh_rasterizer

sleep 30

/usr/bin/shutdown