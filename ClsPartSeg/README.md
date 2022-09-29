# 3D Shape Classification and Part Segmentation

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf), DGCNN, PCT, and PointMLP in pytorch.


## Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (3DCoMPaT/ModelNet10/40)
### Data Preparation
Download alignment **3DCoMPaT** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/compat/`.

### Run

* If you want to train on ModelNet40/10, you can use `--dataset ModelNet40`.
```shell
# ModelNet40
## Select different models in ./models 

## e.g., pointnet2_ssg 
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
```

### Performance
| Model | Accuracy |
|--|--|
| PointNet2_SSG  |  74.89|
| PointNet2_MSG  |  **92.8**|

## Part Segmentation (ShapeNet)
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_part_seg_ssg
python test_partseg.py --log_dir pointnet2_part_seg_ssg
```
### Performance
| Model | Inctance avg IoU| Class avg IoU 
|--|--|--|
|PointNet2_SSG (Pytorch)|	84.9|	81.8	


## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)
```

This code repository is heavily borrowed from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PCT](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch), and [PointMLP](https://github.com/ma-xu/pointMLP-pytorch)

## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
