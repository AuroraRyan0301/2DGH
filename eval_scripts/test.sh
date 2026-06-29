python train.py -s /path/to/data/ficus \
    -m  /path/to/output/2dges/ges_test/gh \
    --iterations 30000 \
    --use_gh_rasterizer \
    --gh_detach_optimization \
    --stop_xyz_renew_at 30000 \
    --gh_addtional_step 20000 \
    --gh_lr 0.000 \
    --detach_opt_gh_lr 0.01 \