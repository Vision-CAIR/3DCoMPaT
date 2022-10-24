# This folder includes the code for the 2D/3D *Grounded CoMPaT Recognition (GCR)* Task.

![image](./imgs/fig_GCR.png)

## 1. Data Preparation
- Please download the data associated with 3d compat by filling this form. 
- We defaultly stored the dataset in `3dcompat/` (2D Images in `3dcompat/images/`, 3d Point Cloud in `3dcompat/models`)  
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

Our pretrained Compositions of 10 models is in [comp10](https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing)


[//]: # (Our pretrained Compositions of 50 models is in:)

[//]: # (https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing)

### 3.4 Test

For Example, we evaluate  10 compositions with:

```sh tool/test.sh com10 /config/compat/bpnet_10.yaml 1```

<div id="Mark"></div>

## 4. GCR using PointGroup
To be updated.

## License
This code is released under MIT License (see LICENSE file for details). In simple words, if you copy/use parts of this code please keep the copyright note in place.


## Citation
If you find this work useful in your research, please consider citing:

```
@article{li20223dcompat,
    title={3D CoMPaT: Composition of Materials on Parts of 3D Things (ECCV 2022)},
    author={Yuchen Li, Ujjwal Upadhyay, Habib Slim, Ahmed Abdelreheem, Arpit Prajapati, Suhail Pothigara, Peter Wonka, Mohamed Elhoseiny},
    journal = {ECCV},
    volume = {XXXX},
    year={2022}
}
```
