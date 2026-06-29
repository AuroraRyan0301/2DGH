import os
from argparse import ArgumentParser

# dtu_scenes = ['scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

# dtu_scenes = ['scan24','scan63']
# dtu_scenes = ['scan65']
# ns_scenes = ['lego', "mic", "hotdog", "materials", "ship", "chair", "ficus", "drums"]
ns_scenes = ['chair']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--use_original_rasterizer", action="store_true")
parser.add_argument("--use_ges_rasterizer", action="store_true")
parser.add_argument("--output_path", default="./eval/ns")
parser.add_argument('--ns', "-ns", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(ns_scenes)

if args.use_original_rasterizer:
    rasterizer_type = "use_original_rasterizer"
elif args.use_ges_rasterizer:
    rasterizer_type = "use_ges_rasterizer"
else:
    raise ValueError("Please specify a rasterizer type")

if not args.skip_training:
    common_args = " -r 2 --lambda_dist 100 --{}".format(rasterizer_type)
    for scene in ns_scenes:
        source = args.ns + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        
if not args.skip_rendering:
    all_sources = []
    common_args = " --eval --skip_train --voxel_size 0.004 --{}".format(rasterizer_type)
    # common_args = " --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 --{}".format(rasterizer_type)
    for scene in ns_scenes:
        source = args.ns + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in ns_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    
    os.system("python metrics.py -m " + scenes_string)