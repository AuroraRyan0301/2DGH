
# python scripts/ns_eval.py \
#     --ns /path/to/dataset/nerf_synthetic \
#     --output_path /path/to/eval/ns_eval/gh \
#     --use_gh_rasterizer

# python scripts/ns_eval.py \
#     --ns /path/to/dataset/nerf_synthetic \
#     --output_path /path/to/eval/ns_eval/original \
#     --use_original_rasterizer

python scripts/ns_eval.py \
    --ns /path/to/dataset/nerf_synthetic \
    --output_path /path/to/eval/ns_eval/ges \
    --use_ges_rasterizer

# python scripts/ns_eval.py \
#     --ns /path/to/dataset/nerf_synthetic \
#     --output_path /path/to/eval/ns_eval/gss \
#     --skip_training \
#     --skip_rendering \
#     --use_gss_rasterizer
    # --skip_metrics

sleep 30

/usr/bin/shutdown