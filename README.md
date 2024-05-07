# MIM: High-definition Maps Incorporated Multi-view Object Detection

https://github.com/WHU-xjs/MIM-3D-Det/assets/60166663/e2eb1709-3c6e-460b-9346-8299d9296815

- [x] release codes and scripts
- [x] release trained weights
- [x] instructions on installation
- [ ] instructions on using codes and scripts
- [ ] instructions on conducting HD maps analysis

## Data Preparation

You need the [nuScenes](https://www.nuscenes.org/nuscenes#download) dataset to train and validate MIM.

You can follow the [mmdetection3d(v0.17.1)](https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/nuscenes_det.html) to prepare the dataset. Generate annotation files and symbol-link the nuScenes folder under data/nuscenes. 

## Installation
```bash
git clone https://github.com/xxxxxx/mim-3d-det.git
cd mim-3d-det

conda create -n mim python=3.8 -y
conda activate mim
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install

cd ..
pip install pillow==8.4.0
python setup.py develop
```

## Training and Testing

Specify your configs and setting in the *train.sh*/*test.sh*.

**NOTE:** training *best_cam.py* and *best.py* requires MCDA, thus need to create image-based groundtruth database first. The code for generation is public in *tools/create_gtdb_img.py*, whose detailed usage will be updated later. To simply train without MCDA, refer to the train pipeline in *mim_tiny_maps.py*, overwrite configs in the training file with them. MCDA does not affect testing. 
```bash
n01kt4/train.sh
n01kt4/test.sh
```
Training MIM under higher resolution (requires ~20G GPU memory) can be extremely time consuming (2 weeks) and results may not turn out ideal, as we do not reach for distributed training and *batch size* is set to 1. Therefore using the trained weights instead is suggested for personal users. 

However, if you intend to train MIM with multiple GPUs and large *batch size* >= 8, we strongly recommend you to switch modules using **layer normalization** back to common modules using BN, which should produce better results than those reported in our paper. Details will be elucidated later. 

## Weights

Trained weights of MIM and Base(no HD maps) under higher resolution (512x1408) can be found at [mim-weights](https://pan.baidu.com/s/1V1oWQw_ic5H4gWnu-D-5_g?pwd=bt1q).

MIM under lower resolution contains a number of varities, we encourage to experiment with the config *mim_tiny_maps.py* and train your own model. It takes ~40h to train a model on one RTX3090. The *batch size* is 16 by default, allowing to train on GPUs with smaller memory using smaller batch size. 
