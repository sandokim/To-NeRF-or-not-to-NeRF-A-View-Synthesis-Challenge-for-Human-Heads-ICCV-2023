# Yonsei-KIST NeRF
## Mitigating Floating Artifacts through Imposing Occlusion Regularization and Backview Exclusion
To surpass the TensoRF baseline performance, we introduce occlusion regularization to reduce the floating artifacts commonly encountered in few-shot neural rendering tasks. The key idea of this regularization term is to penalize the density fields near the camera. Additionally, we find it beneficial to exclude all the backside views in the preprocessing step to avoid floating artifacts. This regularization significantly improves the baseline method. Our implementation extensively leverages the [FreeNeRF](https://github.com/Jiawei-Yang/FreeNeRF) codebase to incorporate the occlusion regularization.

## TensoRF
## [Project page](https://apchenstu.github.io/TensoRF/) |  [Paper](https://arxiv.org/abs/2203.09517)
This repository contains a pytorch implementation for the paper: [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517). Our work present a novel approach to model and reconstruct radiance fields, which achieves super
**fast** training process, **compact** memory footprint and **state-of-the-art** rendering quality.<br><br>

https://user-images.githubusercontent.com/16453770/158920837-3fafaa17-6ed9-4414-a0b1-a80dc9e10301.mp4
## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```

## To reproduce our results, set datadir to ILSH dataset in a configs file for each subject and run 

```
python train.py --config configs/002_00_backview_X_Occ_reg_ver4.txt
python train.py --config configs/003_00_backview_X_Occ_reg_ver4.txt
...
python train.py --config configs/050_00_backview_X_Occ_reg_ver4.txt
```

## Rendering

```
python train.py --config configs/002_00_backview_X_Occ_reg_ver4.txt --ckpt log/tensorf_002_00_backview_X_Occ_reg_ver4/tensorf_002_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1 
python train.py --config configs/003_00_backview_X_Occ_reg_ver4.txt --ckpt log/tensorf_003_00_backview_X_Occ_reg_ver4/tensorf_003_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1 
...
python train.py --config configs/050_00_backview_X_Occ_reg_ver4.txt --ckpt log/tensorf_050_00_backview_X_Occ_reg_ver4/tensorf_050_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## To reproduce baseline results, set datadir to ILSH dataset in a configs file for each subject and run 

```
python train.py --config configs/002_00.txt
python train.py --config configs/003_00.txt
...
python train.py --config configs/050_00.txt
```

## Rendering

```
python train.py --config configs/002_00.txt --ckpt log/tensorf_002_00/tensorf_002_00.th --render_only 1 --render_test 1 
python train.py --config configs/003_00.txt --ckpt log/tensorf_003_00/tensorf_003_00.th --render_only 1 --render_test 1 
...
python train.py --config configs/050_00.txt --ckpt log/tensorf_050_00/tensorf_050_00.th --render_only 1 --render_test 1 
```

## Pretrained weights

You can find our pretrained weights here https://drive.google.com/file/d/1PDfYt7z_KJVwxlYi7D0E3U9tSjqXwLZH/view?usp=sharing

To use pretrained weights to reproduce the results, simply run 

```
python train.py --config configs/002_00.txt --ckpt path/to/ckpt/tensorf_002_00_backview_X_Occ_reg_ver4/tensorf_002_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1
python train.py --config configs/003_00.txt --ckpt path/to/ckpt/tensorf_003_00_backview_X_Occ_reg_ver4/tensorf_003_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1
...
python train.py --config configs/050_00.txt --ckpt path/to/ckpt/tensorf_050_00_backview_X_Occ_reg_ver4/tensorf_050_00_backview_X_Occ_reg_ver4.th --render_only 1 --render_test 1
```

