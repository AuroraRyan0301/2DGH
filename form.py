"""Unified result reporter for 2DGH experiments.

For each method on a dataset it prints per-scene and mean metrics:
CD, number of Gaussians, model size (MB), PSNR, SSIM, LPIPS.

Datasets / comparison sets are predefined in CONFIGS below; pick one or all.

Usage:
    python form.py                                  # report every config
    python form.py --config ns                      # NeRF-synthetic NVS comparison
    python form.py --config detail                  # Detail dataset comparison
    python form.py --config ablation                # GH rank ablation
    python form.py --config detail --grand-path /my/output/detail_nvs
    python form.py --config ns --iteration 30000
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

DEFAULT_ITER = 30000

# Each config maps a dataset to its scenes and a {method label -> output subdir} table.
# Output layout assumed: <grand_path>/<scene>/<subdir>/{cd_results.json,
# gaussian_info.json, results.json, point_cloud/iteration_<iter>/point_cloud.ply}
CONFIGS = {
    "ns": {
        "grand_path": "/path/to/output/ns_nvs",
        "objs": ["lego", "ficus", "drums", "chair", "ship", "mic", "hotdog", "materials"],
        "methods": {
            "original": "original_complete_loss_rb",
            "ges": "ges_complete_loss_rb",
            "gh": "gh_complete_loss_rb",
            "gh_large_lr": "gh_complete_loss_rb_large_lr",
        },
    },
    "detail": {
        "grand_path": "/path/to/output/detail_nvs",
        "objs": ["box", "arduino", "bonsai", "chinese_fan", "spaceship_sun",
                 "japan", "sars", "toad", "triple_trunk"],
        "methods": {
            "original": "original_complete_loss_rb",
            "ges": "ges_complete_loss_rb",
            "gh": "gh_complete_loss_rb",
        },
    },
    "ablation": {  # GH rank ablation (see my_scripts/ablation/rank_ablation.sh)
        "grand_path": "/path/to/output/detail_nvs",
        "objs": ["box", "arduino", "bonsai", "chinese_fan", "spaceship_sun",
                 "japan", "sars", "toad", "triple_trunk"],
        "methods": {
            "rank0": "gh_complete_loss_rb_rank0",
            "rank1": "gh_complete_loss_rb_rank1",
            "rank2": "gh_complete_loss_rb_rank2",
        },
    },
}


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# Each metric is a function(model_dir, iteration) -> value or None (missing).
def _cd(model_dir, _iter):
    j = _read_json(os.path.join(model_dir, "cd_results.json"))
    return j["loss"] if j else None


def _num_gaussians(model_dir, _iter):
    j = _read_json(os.path.join(model_dir, "gaussian_info.json"))
    return j["num_gaussians"] if j else None


def _model_size_mb(model_dir, iteration):
    ply = os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    return os.path.getsize(ply) / (1024 * 1024) if os.path.exists(ply) else None


def _quality(key):
    def fn(model_dir, iteration):
        j = _read_json(os.path.join(model_dir, "results.json"))
        run = f"ours_{iteration}"
        if not j or run not in j or key not in j[run]:
            return None
        return round(j[run][key], 4)
    return fn


METRICS = {
    "CD": _cd,
    "num_gaussians": _num_gaussians,
    "model_size(MB)": _model_size_mb,
    "PSNR": _quality("PSNR"),
    "SSIM": _quality("SSIM"),
    "LPIPS": _quality("LPIPS"),
}


def report(name, cfg, iteration):
    objs, methods, grand = cfg["objs"], cfg["methods"], cfg["grand_path"]
    print("\n" + "=" * 78)
    print(f"[{name}]  {grand}  (iter {iteration})")
    print("=" * 78)
    for metric_name, fn in METRICS.items():
        df = pd.DataFrame(index=list(methods), columns=objs + ["mean"], dtype=float)
        for label, subdir in methods.items():
            vals = []
            for obj in objs:
                v = fn(os.path.join(grand, obj, subdir), iteration)
                df.loc[label, obj] = np.nan if v is None else v
                vals.append(np.nan if v is None else v)
            df.loc[label, "mean"] = np.nanmean(vals) if any(not np.isnan(x) for x in vals) else np.nan
        print(f"\n--- {metric_name} ---")
        print(df.to_string())


def main():
    ap = argparse.ArgumentParser(description="Report 2DGH experiment metrics across datasets.")
    ap.add_argument("--config", choices=list(CONFIGS) + ["all"], default="all",
                    help="which experiment config to report (default: all)")
    ap.add_argument("--grand-path", default=None,
                    help="override the output root for the selected (single) config")
    ap.add_argument("--iteration", type=int, default=DEFAULT_ITER,
                    help=f"evaluation iteration to read (default: {DEFAULT_ITER})")
    args = ap.parse_args()

    names = list(CONFIGS) if args.config == "all" else [args.config]
    if args.grand_path and len(names) > 1:
        ap.error("--grand-path can only be used with a single --config")
    for name in names:
        cfg = dict(CONFIGS[name])
        if args.grand_path:
            cfg["grand_path"] = args.grand_path
        report(name, cfg, args.iteration)


if __name__ == "__main__":
    main()
