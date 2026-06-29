<br />
<p align="center">
  <h1 align="center">2DGH: 2D Gaussian-Hermite Splatting for<br/>High-quality Rendering and Better Geometry Features</h1>
</p>

<p align="center">
  IEEE Transactions on Visualization and Computer Graphics (TVCG) 2025
  <br />
  <a href="https://auroraryan0301.github.io/"><strong>Ruihan Yu</strong></a>*
  ·
  <a href="https://illumiart.net/"><strong>Tianyu Huang</strong></a>*
  ·
  <a href="https://gerwang.github.io/"><strong>Jingwang Ling</strong></a>
  ·
  <a href="http://xufeng.site/"><strong>Feng Xu</strong></a>&dagger;
  <br />
  <sub>* Equal contribution &nbsp;&middot;&nbsp; &dagger; Corresponding author &nbsp;&middot;&nbsp; Tsinghua University</sub>
</p>

<p align="center">
  <a href="https://auroraryan0301.github.io/2dgh/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"></a>
  <a href="https://ieeexplore.ieee.org/document/11204833/">
    <img src="https://img.shields.io/badge/IEEE-TVCG-red?style=flat-square" alt="IEEE TVCG"></a>
  <a href="https://arxiv.org/abs/2408.16982">
    <img src="https://img.shields.io/badge/arXiv-2408.16982-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/AuroraRyan2/Detail">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow?style=flat-square" alt="Data"></a>
</p>

<p align="center">
  <img src="https://auroraryan0301.github.io/2dgh/static/images/teaser.png" alt="2DGH teaser" width="100%"/>
</p>

<br>

Overview
--------

2DGH replaces the Gaussian primitive of
[2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting) with a
**Gaussian-Hermite (GH) kernel** — a Gaussian modulated by a Hermite-polynomial series.
Inspired by the higher-rank wavefunctions of the quantum harmonic oscillator, the GH
kernel is a unified family in which the standard Gaussian is the **rank-0 special case**;
higher ranks add anisotropy and deformation power, sharpening object silhouettes and
recovering fine, non-Gaussian detail. A dedicated activation keeps the alpha-blended
opacity within [0, 1] even with the large or negative high-order coefficients. This
repository contains the training, rendering and evaluation code, and reproduces the
comparison between three rasterizers:

| Method     | Flag                        | Rasterizer kernel                       |
| ---------- | --------------------------- | --------------------------------------- |
| `original` | `--use_original_rasterizer` | `diff-surfel-rasterization-original`    |
| `ges`      | `--use_ges_rasterizer`      | `diff-ges-surfel-rasterization`         |
| `gh`       | `--use_gh_rasterizer`       | `diff_gh_surfel_rasterization`          |


Installation
------------

The code is tested with **Python 3.8**, **PyTorch 2.4**, and a **CUDA 11.8 / 12.1**
toolkit on an NVIDIA RTX 4090 (compute capability 8.9).

### 1. Clone with submodules

The CUDA rasterizers and `simple-knn` are git submodules, so clone recursively:

```bash
git clone --recursive https://github.com/AuroraRyan0301/2DGH.git
cd 2DGH
# If you already cloned without --recursive:
git submodule update --init --recursive
```

| Submodule path                                  | Purpose                            |
| ----------------------------------------------- | ---------------------------------- |
| `submodules/diff_gh_surfel_rasterization`       | GH rasterizer (this paper)         |
| `submodules/diff-ges-surfel-rasterization`      | GES rasterizer baseline            |
| `submodules/diff-surfel-rasterization-original` | original 2DGS rasterizer           |
| `submodules/simple-knn`                         | KNN for point initialization       |

### 2. Create the environment

```bash
conda create -n 2dgh python=3.8.18
conda activate 2dgh
```

Install PyTorch matching your CUDA toolkit. For CUDA 12.x:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

(For CUDA 11.8, use `--index-url https://download.pytorch.org/whl/cu118`.)

### 3. Vendor GLM into the rasterizers

The CUDA kernels need the [GLM](https://github.com/g-truc/glm) headers. `glm.sh` clones
the correct GLM revision and copies it into each rasterizer's `third_party/`:

```bash
bash glm.sh
```

### 4. Install dependencies and the CUDA kernels

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"   # set to your GPU's compute capability
pip install -r requirements.txt
```

`requirements.txt` installs `pytorch3d`, `chamfer_distance`, and the four local
submodules in editable mode. `pytorch3d` is compiled from source, so make sure
`CUDA_HOME` matches the toolkit your PyTorch build was compiled against. Building the
CUDA kernels takes several minutes.

### 5. Verify

```bash
python -c "from diff_gh_surfel_rasterization import GaussianGHRasterizer; \
           from diff_ges_surfel_rasterization import GES2DRasterizer; \
           from diff_surfel_rasterization_original import OriginalGaussianRasterizer; \
           from simple_knn._C import distCUDA2; print('rasterizers OK')"
```


Datasets
--------

| Dataset        | Link                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| Synthetic NeRF | `python down.py` (Blender scenes with object GT meshes)                    |
| Detail         | [Hugging Face](https://huggingface.co/datasets/AuroraRyan2/Detail)         |
| Mip-NeRF 360   | [project page](https://jonbarron.info/mipnerf360/)                         |
| DTU            | [DTU Robot Image Data](https://roboimagedata.compute.dtu.dk/?page_id=36)   |

The **Detail** dataset (high-frequency synthetic scenes used in the paper) is hosted on
Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download AuroraRyan2/Detail --repo-type dataset --local-dir data/Detail
```

After downloading, point the dataset paths in the scripts under `my_scripts/` and
`eval_scripts/` to your dataset locations.


Training, rendering and evaluation
----------------------------------

```bash
# Train (choose one rasterizer flag)
python train.py -s <scene> -m <output_dir> --use_gh_rasterizer --hg_opt_flag \
    --white_background --lambda_dist 100

# Render / extract mesh
python render.py -s <scene> -m <output_dir> --use_gh_rasterizer --hg_opt_flag --eval

# Metrics (PSNR/SSIM/LPIPS) and Chamfer distance
python metrics.py -m <output_dir>
```

Reproduction scripts (edit the dataset paths inside first):

```bash
bash my_scripts/nvs/ns_nvs.sh                # novel-view synthesis, Synthetic NeRF
bash my_scripts/nvs/m360_nvs.sh              # novel-view synthesis, Mip-NeRF 360
bash my_scripts/rendering/ns_rendering_qualtiy.sh
bash my_scripts/rendering/dtu_rendering_quality.sh
bash my_scripts/overhead/overhead.sh
bash my_scripts/ablation/rank_ablation.sh    # GH rank ablation
```


GH rank ablation
----------------

The GH coefficients $c_{mn}$ sit on a square $N \times M$ arrangement of per-axis Hermite
ranks (the released kernel uses $N = M = 3$, i.e. ranks $H_0$–$H_2$ per axis, 9
coefficients in total); the standard Gaussian is the rank-0 case. `--max_gh_rank` controls
how many ranks are active, progressively enabling coefficients:

| `--max_gh_rank` | active coeffs | Hermite terms                            |
| --------------- | ------------- | ---------------------------------------- |
| 0 (baseline)    | 0             | Gaussian only (rank-0)                   |
| 1               | 1             | constant                                 |
| 2               | 4             | up to rank 1 (`dx`, `dy`, `dx·dy`)       |
| 3               | 9             | up to rank 2 (full 3×3 arrangement)      |

`my_scripts/ablation/rank_ablation.sh` sweeps these ranks; the rank-0 baseline freezes the
constant (`--max_gh_rank 1 --gh_lr 0`), i.e. original 2DGS plus the GH activation.


Reporting results
------------------

`form.py` aggregates per-scene and mean metrics (CD, #Gaussians, model size, PSNR, SSIM,
LPIPS) for a dataset/comparison and prints them as tables:

```bash
python form.py --config ns          # Synthetic NeRF comparison
python form.py --config detail      # Detail comparison
python form.py --config ablation    # GH rank ablation
python form.py --config detail --grand-path /your/output/detail_nvs
```


Citation
--------

```bibtex
@article{yu2025gaussianhermite,
  title   = {2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering
             and Better Geometry Features},
  author  = {Yu, Ruihan and Huang, Tianyu and Ling, Jingwang and Xu, Feng},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2025}
}
```


Acknowledgements
----------------

This code builds on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting).
See [`LICENSE.md`](LICENSE.md) for license terms (non-commercial research use).
