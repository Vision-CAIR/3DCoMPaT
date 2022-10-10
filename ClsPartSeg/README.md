# 3D Shape Classification and Part Segmentation

This repo includes the code for 3d Shape Classification and Part Segmentation on 3DCoMPaT dataset using prevalent 3D vision algorithms, including [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf), [DGCNN](https://arxiv.org/abs/1801.07829), [PCT](https://arxiv.org/pdf/2012.09688.pdf), and [PointMLP](https://arxiv.org/abs/2202.07123) in pytorch.

You can find the pretrained models and log files in [Gdrive](https://drive.google.com/drive/folders/1k1TDDzNvfnnxd_F8PxlsPBmnrr11-I-w?usp=sharing).

## 1. Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.7 and Python 3.7:
```shell
conda install pytorch==1.7.0 cudatoolkit=10.1 -c pytorch
```

## 2. Data Preparation
Download our preprocessed data **3DCoMPaT**  and save in `data/compat/`.


## 3. Classification (3DCoMPaT/ModelNet10/40)

```shell
# 3DCoMPaT
## Select different models in ./models 

## e.g., pointnet2_ssg 
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
```
* If you want to train on ModelNet40/10, you can use `--dataset ModelNet40`.

* Note that we use same data augmentations and training schedules for all comparing methods following [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
### Performance (Instance average Accuracy)
| Model | Previous | Val | Updated | 
|--|--|--|
| PointNet2_SSG  | - | - |73.78 |
| PointNet2_MSG  |  57.95|- | 74.70| 
| DGCNN  |  68.32|- | 72.22 |
| PCT  |  69.09 |- | 68.74|
| PointMLP  |  - |- | 70.83|


## 4. Part Segmentation (3DCoMPaT)

```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_part_seg_ssg
python test_partseg.py --log_dir pointnet2_part_seg_ssg
```

### Performance (Accuracy)
| Model | Previous| Val | Test |
|--|--|--|
|PointNet2_SSG|24.18|- | 51.22|
|PCT | 37.37 |- | 48.43| 

## 5. Sim2Rel:Transferring to ScanObjectNN
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_classification_sim2rel.py --model pointmlp --log_dir pointmlp_cls
python test_classification_sim2rel.py --log_dir pointmlp_cls
```

### Performance (Accuracy)
| Model | Previous | Test| 
|--|--|--|
|ModelNet40|24.33| 30.69|
|3DCoMPaT | 29.21 | 28.49| 

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

```
@article{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

```
@article{ma2022rethinking,
  title={Rethinking network design and local geometry in point cloud: A simple residual mlp framework},
  author={Ma, Xu and Qin, Can and You, Haoxuan and Ran, Haoxi and Fu, Yun},
  journal={arXiv preprint arXiv:2202.07123},
  year={2022}
}
```

```
@article{dgcnn,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019}
}
```
```
@article{Guo_2021,
   title={PCT: Point cloud transformer},
   volume={7},
   ISSN={2096-0662},
   url={http://dx.doi.org/10.1007/s41095-021-0229-5},
   DOI={10.1007/s41095-021-0229-5},
   number={2},
   journal={Computational Visual Media},
   publisher={Springer Science and Business Media LLC},
   author={Guo, Meng-Hao and Cai, Jun-Xiong and Liu, Zheng-Ning and Mu, Tai-Jiang and Martin, Ralph R. and Hu, Shi-Min},
   year={2021},
   month={Apr},
   pages={187–199}
}
```

