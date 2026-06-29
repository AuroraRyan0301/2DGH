python scripts/dtu_eval.py \
    --dtu /path/to/dataset/DTU/DTU \
    --DTU_Official /path/to/dataset/DTU/STLPoints \
    --output_path /path/to/eval/DTU_eval/original \
    --eval_path /path/to/eval/DTU_eval/original \
    --use_original_rasterizer

python scripts/dtu_eval.py \
    --dtu /path/to/dataset/DTU/DTU \
    --DTU_Official /path/to/dataset/DTU/STLPoints \
    --output_path /path/to/eval/DTU_eval/gh \
    --eval_path /path/to/eval/DTU_eval/gh \
    --use_gh_rasterizer

python scripts/dtu_eval.py \
    --dtu /path/to/dataset/DTU/DTU \
    --DTU_Official /path/to/dataset/DTU/STLPoints \
    --output_path /path/to/eval/DTU_eval/ges \
    --eval_path /path/to/eval/DTU_eval/ges \
    --use_ges_rasterizer

sleep 30

/usr/bin/shutdown