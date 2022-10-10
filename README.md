# 3D CoMPaT: Composition of Materials on Parts of 3D Things (ECCV 2022)
[Website Badge](https://3dcompat-dataset.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!--[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)]-->
Created by: [Yuchen Li](http://liyc.tech/), [Ujjwal Upadhyay](https://ujjwal9.com/), [Habib Slim](https://habibslim.github.io/), [Ahmed Abdelreheem](https://samir55.github.io/), [Arpit Prajapati](https://www.polynine.com/), [Suhail Pothigara](https://www.polynine.com/), [Peter Wonka](https://peterwonka.net/), [Mohamed Elhoseiny](http://www.mohamed-elhoseiny.com/)
![image](https://user-images.githubusercontent.com/38585175/182629905-812f1c6f-8906-4485-9710-760cff150df1.png)
## Introduction

This work is based on the arXiv tech report which is provisionally accepted in ECCV-2022, for an Oral presentation.

Citation
If you find this work useful in your research, please consider citing:

```
@article{li20223dcompat,
    title={3D CoMPaT: Composition of Materials on Parts of 3D Things (ECCV 2022)},
    author={Yuchen Li, Ujjwal Upadhyay, Ujjwal Upadhyay, Ujjwal Upadhyay, Ahmed Abdelreheem, Arpit Prajapati, Suhail Pothigara, Peter Wonka, Mohamed Elhoseiny},
    journal = {ECCV},
    volume = {XXXX},
    year={2022}
}
```
## Dataset
To get the most out of the github repository, please download the data associated with 3d compat by filling this [form](https://docs.google.com/forms/d/e/1FAIpQLSeOxWVkVNdXz-nCfFIWOeOARc_Atk9fi5PSIKw1Ib1cr3ENpA/viewform?fbzx=-7103523806700241333).

## Browser
You can browse the 3D models using the following link: [3D CoMPaT Browser](http://54.235.12.220:50/index.html)

## Compositional Benchmarks
### BPNET
Existing segmentation methods are mostly unidirectional, i.e. utilizing 3D for 2D segmentation or vice versa. Obviously 2D and 3D information can nicely complement each other in both directions, during the segmentation. This is the goal of bidirectional projection network.
 
#### Installation
```
# Torch
$ pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
# MinkowskiEngine 0.4.1
$ conda install numpy openblas
$ git clone https://github.com/StanfordVL/MinkowskiEngine.git
$ cd MinkowskiEngine
$ git checkout f1a419cc5792562a06df9e1da686b7ce8f3bb5ad
$ python setup.py install
# Others
$ pip install imageio==2.8.0 opencv-python==4.2.0.32 pillow==7.0.0 pyyaml==5.3 scipy==1.4.1 sharedarray==3.2.0 tensorboardx==2.0 tqdm==4.42.1
```

#### Config
- BPNet with 10 Compositions: ```config/compat/bpnet_10.yaml``` 
- BPNet with 50 Compositions: ```config/compat/bpnet_50.yaml``` 

#### Training


- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**


For Example, we train 10 compositions with:

```sh tool/train.sh com10 /config/compat/bpnet_10.yaml 1```

#### Test

For Example, we evaluate  10 compositions with:

```sh tool/test.sh com10 /config/compat/bpnet_10.yaml /PATH/TO/PRETRAIN_MODEL 1```

#### Pretrain Model

Our pretrained models is in:

https://drive.google.com/drive/folders/1k1TDDzNvfnnxd_F8PxlsPBmnrr11-I-w?usp=sharing


## Non-Compositional Benchmarks
### 1. Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.7 and Python 3.7:
```shell
conda install pytorch==1.7.0 cudatoolkit=10.1 -c pytorch
```

### 2. Data Preparation
Download our preprocessed data [3DCoMPaT](https://drive.google.com/drive/folders/1ZeX7sXaaumjaeI9UWrFAoHz8DO_ZcN-J?usp=sharing) and save in `data/compat/`.

### 3. Classification (3DCoMPaT)

```shell
## 3DCoMPaT
### Select different models in ./models 

### e.g., pointnet2_ssg 
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
```

* Note that we use same data augmentations and training schedules for all comparing methods following [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We report performance on both validation and test sets.
#### Performance (Instance average Accuracy)
| Model | Previous | Val | Test | Pretrained| 
|--|--|--|--|--|
| PointNet2_SSG  | - | - |73.78 | [gdrive](https://drive.google.com/drive/folders/1S9sdkk3m2rGTcOE8Iv1NY2CMDKskwRUo?usp=sharing) | 
| PointNet2_MSG  |  57.95| - | 74.70|  [gdrive](https://drive.google.com/drive/folders/1YkXI5ouvigcET-JycoUhprjSgaAyrTQE?usp=sharing) | 
| DGCNN  |  68.32| - | 72.22 | [gdrive](https://drive.google.com/drive/folders/12FWcSsqiTtVKoL_twynhmCPApdYD-sAa?usp=sharing) | 
| PCT  |  69.09 | - | 68.74| [gdrive](https://drive.google.com/drive/folders/1YAmNJrxiWRIyHpc2sSD828ELM-swyoFh?usp=sharing) | 
| PointMLP  |  - | - | 70.83| [gdrive](https://drive.google.com/drive/folders/1B5CPHuPQRsn3SmW5ZNo88JuVBR8fqYg8?usp=sharing) | 


### 4. Part Segmentation (3DCoMPaT)

```
### Check model in ./models 
### e.g., pointnet2_ssg
python train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_part_seg_ssg
python test_partseg.py --log_dir pointnet2_part_seg_ssg
```

* Note that we use same data augmentations and training schedules for all comparing methods following [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We report performance on both validation and test sets.

#### Performance (Accuracy)
| Model | Previous| Val | Test | Pretrained|
|--|--|--|--|--|
|PointNet2_SSG|24.18| - | 51.22| [gdrive](https://drive.google.com/drive/folders/1yoGpiwCxHM-cqE_T2s4RrH7XiMvbhjlu?usp=sharing) | 
|PCT | 37.37 | - | 48.43| [gdrive](https://drive.google.com/drive/folders/1X8fN1PXFqnFmoMY1EUwjRzWBJEiBbdwB?usp=sharing) | 

### 5. Sim2Rel:Transferring to ScanObjectNN
```
### Check model in ./models 
### e.g., pointnet2_ssg
python train_classification_sim2rel.py --model pointmlp --log_dir pointmlp_cls
python test_classification_sim2rel.py --log_dir pointmlp_cls
```
Note that we use same data augmentations and training schedules for all comparing methods following [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We report performance on the test set of ScanObjectNN.
#### Performance (Accuracy)
| Model | Previous | Test| Pretrained|
|--|--|--|--|
|ModelNet40|24.33| 30.69| [gdrive](https://drive.google.com/drive/folders/155vEhpSTfBLbievkEKoEMA4ZW4t5CaHC?usp=sharing) | 
|3DCoMPaT | 29.21 | 28.49| [gdrive](https://drive.google.com/drive/folders/10jFftppEPvZzid0TXbWyqBovyQEPWBqa?usp=sharing) | 

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




## License
This code is released under MIT License (see LICENSE file for details). In simple words, if you copy/use parts of this code please keep the copyright note in place.
