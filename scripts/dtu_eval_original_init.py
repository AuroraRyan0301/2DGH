# Usage:
# python scripts/dtu_eval_original_init.py --dtu /path/to/DTU/dataset/ \
# --DTU_Official /path/to/STLPoints/ \
# --output_path ./output/ \
# --eval_path ./output/



import os
from argparse import ArgumentParser

dtu_scenes = ['scan24','scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
# dtu_scenes = ['scan24']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./output/dtu_original_init")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    parser.add_argument('--eval_path', required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    # We do not use the --eval parameter here because the DTU dataset is used for evaluating geometric reconstruction and rendering quality.
    original_init_common_args = f" --depth_ratio 1.0 -r 2 --lambda_dist 1000 --use_original_rasterizer --iterations 15000 --hg_opt_flag"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + original_init_common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + original_init_common_args)
    common_args = f" --depth_ratio 1.0 -r 2 --lambda_dist 1000 --iterations 30000 --hg_opt_flag"
    # original
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        original_common_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth --use_original_rasterizer"
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + original_common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + original_common_args)
    # ges
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        ges_common_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth --use_ges_rasterizer"
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + "_ges" + ges_common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + "_ges" + ges_common_args)
    # gh
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        gh_common_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth --use_gh_rasterizer"
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + "_gh" + gh_common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + "_gh" + gh_common_args)


if not args.skip_rendering:
    all_sources = []
    # original
    common_args = f" --eval --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 --use_original_rasterizer --hg_opt_flag"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
    # ges
    common_args = f" --eval --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 --use_ges_rasterizer --hg_opt_flag"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + "_ges" + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + "_ges" + common_args)
    # gh
    common_args = f" --eval --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 --use_gh_rasterizer --hg_opt_flag"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + "_gh" + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + "_gh" + common_args)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # original
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
        iteration = 30000
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {args.output_path}/dtu_metrics/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)
    # ges
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}_ges/train/ours_30000/"
        iteration = 30000
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}_ges/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {args.output_path}/dtu_metrics_ges/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)
    # gh
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}_gh/train/ours_30000/"
        iteration = 30000
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}_gh/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {args.output_path}/dtu_metrics_gh/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)