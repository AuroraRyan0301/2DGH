obj_list=("lego" "ficus" "drums" "chair" "ship" "mic" "hotdog" "materials")
# obj_list=("lego")
method=gh

# GH (formerly gh_newv2) rank ablation.
# --max_gh_rank caps the GH polynomial degree; for this kernel the degree -> number
# of active GH coefficients mapping is:
#   degree 0 -> 0 coeffs   (degenerate; use the frozen-constant baseline below instead)
#   degree 1 -> 1 coeff    (constant)
#   degree 2 -> 4 coeffs   (const, dx, dy, dx*dy)
#   degree 3 -> 9 coeffs   (full 2nd-order tensor-product basis)
# The "0rank" block below is the original-2DGS + GL-activation baseline:
# max_gh_rank 1 with gh_lr 0 freezes the constant at 1, so no GH detail is learned.
rank_list=($((1)) $((2)) $((3)))

output_dir=ns_rank_ablation

nerf_synthetic_dir=/path/to/nerf_synthetic
for obj in "${obj_list[@]}"; do

    original_last_model=output/${output_dir}/${obj}/o_init
    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m ${original_last_model}\
        --iterations 15000 \
        --lambda_dist 100 \
        --use_original_rasterizer \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --densify_grad_threshold 0.0002

    gh_model=output/${output_dir}/${obj}/gh_complete_loss_rb_0rank

    last_ckpt=${original_last_model}/chkpnt15000.pth

    python train.py -s ${nerf_synthetic_dir}/${obj} \
        -m ${gh_model} \
        --iterations 30000 \
        --lambda_dist 100 \
        --use_gh_rasterizer \
        --gh_lr 0.000 \
        --white_background \
        --hg_opt_flag \
        --random_background \
        --start_checkpoint ${last_ckpt} \
        --densify_grad_threshold 0.0002 \
        --max_gh_rank 1

    python render.py \
        -s ${nerf_synthetic_dir}/${obj} \
        -m output/${output_dir}/"${obj}"/gh_complete_loss_rb_0rank \
        --use_gh_rasterizer \
        --eval \
        --skip_train \
        --voxel_size 0.004 \
        --white_background \
        --hg_opt_flag

    grand_path=output/${output_dir}
    ref_path=${nerf_synthetic_dir}/${obj}/${obj}.obj
    iter=30000


    python metrics.py \
        -m ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_0rank
    python scripts/cd_helper.py \
        ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_0rank/train/ours_${iter}/fuse_post.ply \
        ${ref_path} \
        ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_0rank
    
    
    for rank in "${rank_list[@]}"; do
        gh_model=output/${output_dir}/${obj}/gh_complete_loss_rb_${rank}rank

        last_ckpt=${original_last_model}/chkpnt15000.pth

        python train.py -s ${nerf_synthetic_dir}/${obj} \
            -m ${gh_model} \
            --iterations 30000 \
            --lambda_dist 100 \
            --use_gh_rasterizer \
            --gh_lr 0.001 \
            --white_background \
            --hg_opt_flag \
            --random_background \
            --start_checkpoint ${last_ckpt} \
            --densify_grad_threshold 0.0002 \
            --max_gh_rank ${rank}

        python render.py \
            -s ${nerf_synthetic_dir}/${obj} \
            -m output/${output_dir}/"${obj}"/gh_complete_loss_rb_${rank}rank \
            --use_gh_rasterizer \
            --eval \
            --skip_train \
            --voxel_size 0.004 \
            --white_background \
            --hg_opt_flag

        grand_path=output/${output_dir}
        ref_path=${nerf_synthetic_dir}/${obj}/${obj}.obj
        iter=30000


        python metrics.py \
            -m ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_${rank}rank
        python scripts/cd_helper.py \
            ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_${rank}rank/train/ours_${iter}/fuse_post.ply \
            ${ref_path} \
            ${grand_path}/"${obj}"/"${method}"_complete_loss_rb_${rank}rank
    
    done

done
