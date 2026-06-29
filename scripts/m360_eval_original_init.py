# Usage: python scripts/m360_eval_original_init.py -m360 /path/to/m360/dataset/

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/m360_original_init")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    original_init_common_args = f" --eval --test_iterations -1 --lambda_dist 100 --use_original_rasterizer --hg_opt_flag --gh_lr 0.001 --iterations 15000"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + original_init_common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + original_init_common_args)
    # original
    common_args = f" --eval --test_iterations -1 --lambda_dist 100 --use_original_rasterizer --hg_opt_flag --gh_lr 0.001 --iterations 30000"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + scene_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + scene_args)
    # ges
    common_args = f" --eval --test_iterations -1 --lambda_dist 100 --use_ges_rasterizer --hg_opt_flag --gh_lr 0.001 --iterations 30000"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + "_ges" + scene_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + "_ges" + scene_args)
    # gh
    common_args = f" --eval --test_iterations -1 --lambda_dist 100 --use_gh_rasterizer --hg_opt_flag --gh_lr 0.001 --iterations 30000"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + "_gh" + scene_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_args = common_args + f" --start_checkpoint {args.output_path}/{scene}/chkpnt15000.pth"
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + "_gh" + scene_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    # original
    common_args = f" --quiet --eval --skip_train --use_original_rasterizer --hg_opt_flag"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    # ges
    common_args = f" --quiet --eval --skip_train --use_ges_rasterizer --hg_opt_flag"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + "_ges" + common_args)
    # gh
    common_args = f" --quiet --eval --skip_train --use_gh_rasterizer --hg_opt_flag"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + "_gh" + common_args)

if not args.skip_metrics:
    # original
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    os.system("python metrics.py -m " + scenes_string)
    # ges
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "_ges" + "\" "
    os.system("python metrics.py -m " + scenes_string)
    # gh
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "_gh" + "\" "
    os.system("python metrics.py -m " + scenes_string)