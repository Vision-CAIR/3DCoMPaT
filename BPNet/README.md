# 3D CoMPaT: Composition of Materials on Parts of 3D Things (ECCV 2022)
[Website Badge](https://3dcompat-dataset.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!--[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)]-->
Created by: [Yuchen Li](http://liyc.tech/), [Ujjwal Upadhyay](https://ujjwal9.com/), [Habib Slim](https://habibslim.github.io/), [Ahmed Abdelreheem](https://samir55.github.io/), [Arpit Prajapati](https://www.polynine.com/), [Suhail Pothigara](https://www.polynine.com/), [Peter Wonka](https://peterwonka.net/), [Mohamed Elhoseiny](http://www.mohamed-elhoseiny.com/)

![image](https://user-images.githubusercontent.com/38585175/182629905-812f1c6f-8906-4485-9710-760cff150df1.png)
## Introduction

This work is based on the arXiv tech report which is provisionally accepted in ECCV-2022, for an Oral presentation.

## Dataset
To get the most out of the github repository, please download the data associated with 3d compat by filling this [form](https://docs.google.com/forms/d/e/1FAIpQLSeOxWVkVNdXz-nCfFIWOeOARc_Atk9fi5PSIKw1Ib1cr3ENpA/viewform?fbzx=-7103523806700241333).

## Browser
You can browse the 3D models using the following link: [3D CoMPaT Browser](http://54.235.12.220:50/index.html)

## Benchmarks
### BPNET
How to Use Train the BPNet 


## Config
- BPNet with 10 Compositions: ```config/compat/bpnet_10.yaml``` 
- BPNet with 50 Compositions: ```config/compat/bpnet_50.yaml``` 

## Training


- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**


For Example, we train 10 compositions with:

```sh tool/train.sh com10 /config/compat/bpnet_10.yaml 1```

## Pretrain Models

Our pretrained Compositions of 10 models is in:

[https://drive.google.com/drive/folders/1k1TDDzNvfnnxd_F8PxlsPBmnrr11-I-w?usp=sharing](https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing)


Our pretrained Compositions of 50 models is in:
https://drive.google.com/file/d/1u7CkloqHEkezFuUBnZQRnbxdW420Xgug/view?usp=sharing

## Test

For Example, we evaluate  10 compositions with:

```sh tool/test.sh com10 /config/compat/bpnet_10.yaml 1```

<div id="Mark"></div>
## 2D/3D Material Segmentation
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
