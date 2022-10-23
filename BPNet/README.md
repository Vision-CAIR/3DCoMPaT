# This folder includes the code for the 2D/3D *Grounded CoMPaT Recognition (GCR)* Task.

![image](fig_GCR.png)

## Data Preparation
To be updated

## 2D/3D Material Segmentation using BPNet (non-compositional)
To be updated.

## GCR using BPNet

### Config
- BPNet with 10 Compositions: ```config/compat/bpnet_10.yaml``` 
- BPNet with 50 Compositions: ```config/compat/bpnet_50.yaml``` 

### Training

- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**


For Example, we train 10 compositions with:

```sh tool/train.sh com10 /config/compat/bpnet_10.yaml 1```

### Pretrain Models

Our pretrained Compositions of 10 models is in:

[https://drive.google.com/drive/folders/1k1TDDzNvfnnxd_F8PxlsPBmnrr11-I-w?usp=sharing](https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing)


Our pretrained Compositions of 50 models is in:
https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing

### Test

For Example, we evaluate  10 compositions with:

```sh tool/test.sh com10 /config/compat/bpnet_10.yaml 1```

<div id="Mark"></div>

## GCR using PointGroup
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
