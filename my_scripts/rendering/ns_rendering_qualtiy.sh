obj_list=("lego" "ficus" "drums" "chair" "ship" "mic" "hotdog" "materials")
# obj_list=("lego")
method_list=("ges" "gh" "original")

output_dir=ns_rendering_train

nerf_synthetic_dir=/path/to/nerf_synthetic

for obj in "${obj_list[@]}"; do

    original_last_model=output/${output_dir}/${obj}/original_complete_loss_rb
    gh_last_model=output/${output_dir}/${obj}/gh_complete_loss_rb
    ges_last_model=output/${output_dir}/${obj}/ges_complete_loss_rb

    last_ckpt=${original_last_model}/chkpnt15000.pth

    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m ${original_last_model}\
        --iterations 15000 \
        --lambda_dist 100 \
        --use_original_rasterizer \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --densify_grad_threshold 0.0002
        

    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m  ${original_last_model}\
        --iterations 30000 \
        --lambda_dist 100 \
        --use_original_rasterizer \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --start_checkpoint ${last_ckpt} \
        --densify_grad_threshold 0.0002 

    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m ${gh_last_model} \
        --iterations 30000 \
        --lambda_dist 100 \
        --use_gh_rasterizer \
        --gh_lr 0.001 \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --start_checkpoint ${last_ckpt} \
        --densify_grad_threshold 0.0002 

    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m ${ges_last_model} \
        --iterations 30000 \
        --lambda_dist 100 \
        --use_ges_rasterizer \
        --beta_lr 0.001 \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --start_checkpoint ${last_ckpt} \
        --densify_grad_threshold 0.0002 

    for method in "${method_list[@]}"; do
        python render.py \
            -s ${nerf_synthetic_dir}/${obj} \
            -m output/${output_dir}/"${obj}"/"${method}"_complete_loss_rb \
            --use_"${method}"_rasterizer \
            --eval \
            --skip_train \
            --voxel_size 0.004 \
            --white_background \
            --hg_opt_flag
    done

    grand_path=output/${output_dir}
    ref_path=${nerf_synthetic_dir}/${obj}/${obj}.obj
    iter=30000

    for method in "${method_list[@]}"; do
        python metrics.py \
            -m ${grand_path}/"${obj}"/"${method}"_complete_loss_rb
        python scripts/cd_helper.py \
            ${grand_path}/"${obj}"/"${method}"_complete_loss_rb/train/ours_${iter}/fuse_post.ply \
            ${ref_path} \
            ${grand_path}/"${obj}"/"${method}"_complete_loss_rb
    done
done