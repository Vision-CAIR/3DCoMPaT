# This folder includes the code for the 2D/3D *Grounded CoMPaT Recognition (GCR)* Task.

![image](./imgs/fig_GCR.png)
## 0. Environment
To run BPNet successfully, please follow the setting on this environment
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

## 1. Data Preparation
- Please download the data associated with 3d compat by filling this form. 
- We defaultly stored the dataset in `3dcompat/` (2D Images in `3dcompat/image/`, 3d Point Cloud in `3dcompat/pc/`)  
You can also modify the `data_root` and `data_root2d` in the config files (e.g. `config/compat/bpnet_10.yaml`)

- For the efficiency of the datanet, we prvoide the point clouds generated from the 3dcompat models. The processing of how we generate the data is from this file `util/glb2pc.py`. You can also generate the point cloud from models in your sides
## 2. 2D/3D Material Segmentation using BPNet (non-compositional)

For the non-compositional material segmentation results, we build from origin BPNet.

- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

For Example, we train 10 compositions with:

```sh tool/train.sh com10 /config/mat_seg/bpnet_10.yaml 1```

- Test

For Example, we evaluate 10 compositions with:

```sh tool/test.sh com10 /config/mat_seg/bpnet_10.yaml 1```

## 3. GCR using BPNet

### 3.1 Config
- BPNet with 10 Compositions: ```config/compat/bpnet_10.yaml``` 

[//]: # (- BPNet with 50 Compositions: ```config/compat/bpnet_50.yaml``` )

### 3.2 Training



- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**


For Example, we train 10 compositions with:

```sh tool/train.sh com10 /config/compat/bpnet_10.yaml 1```

### 3.3 Pretrain Models

Our pretrained Compositions of 10 models is in [comp10](https://drive.google.com/file/d/1PXzUXTcEd4AfzCyjFz-13s76FdljsG5p/view?usp=sharing)


[//]: # (Our pretrained Compositions of 50 models is in:)

[//]: # (https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing)

### 3.4 Test

For Example, we evaluate  10 compositions with:

```sh tool/test.sh com10 /config/compat/bpnet_10.yaml 1```

<div id="Mark"></div>


## License
This code is released under MIT License (see LICENSE file for details). In simple words, if you copy/use parts of this code please keep the copyright note in place.


## Citation
If you find this work useful in your research, please consider citing:

```
@article{li20223dcompat,
    title={3D CoMPaT: Composition of Materials on Parts of 3D Things (ECCV 2022)},
    author={Yuchen Li, Ujjwal Upadhyay, Habib Slim, Ahmed Abdelreheem, Arpit Prajapati, Suhail Pothigara, Peter Wonka, Mohamed Elhoseiny},
    journal = {ECCV},
    year={2022}
}
```
for BPNet

```
@inproceedings{hu-2021-bidirectional,
        author      = {Wenbo Hu, Hengshuang Zhao, Li Jiang, Jiaya Jia and Tien-Tsin Wong},
        title       = {Bidirectional Projection Network for Cross Dimensional Scene Understanding},
        booktitle   = {CVPR},
        year        = {2021}
    }
```