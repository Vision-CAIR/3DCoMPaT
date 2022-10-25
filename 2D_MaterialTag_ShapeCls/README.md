# This folder includes code for 2D shape classification and material tagging.

## 1. Introduction
For both 2D shape classification and 2D material tagging, we train ResNet50 models on rendered images.

## 2. Dataset Preparation
Download our rendered images and put them in ./shards/ folder.

## 3. Training
Run the following script:

	python main.py --num-workers 4 \
	    --use-tmp \
	    --exp-name PRET_CAN_C1 \
	    --exp-tag PRET-CAN \
	    --root-url ./shards/ \
	    --models-dir ./PRET_CAN_C10/ \
	    --batch-size 64 \
	    --nbatches 10000 \
	    --num-classes 43 \
	    --resnet-type resnet50 \
	    --seed 222 \
	    --n-comp 10 \
	    --view-type canonical \
	    --use-pretrained \


Params:

- n-comp indicates the number of selected compositions for each shape.

- view-type indicates the camera view type, choose from ['canonical', 'random', 'all'].

Please check main.py for details about hyperparameters.

We also provide ready-to-use scripts for model training in ./scripts folder.
For example, "SCRT-CAN/SCRT_CAN_C16.sh" includes the script for training a ResNet50 model from scratch using 16 compositions of rendered images from canonical views.

## 4. Evaluation and pretrained models


	python main.py --num-workers 4 \
	    --use-tmp \
	    --exp-name PRET_CAN_C1 \
	    --exp-tag PRET-CAN \
	    --root-url ./shards/ \
	    --models-dir ./PRET_CAN_C10/ \
	    --batch-size 64 \
	    --nbatches 10000 \
	    --num-classes 41 \
	    --resnet-type resnet50 \
	    --seed 222 \
	    --n-comp 10 \
	    --view-type canonical \
	    --use-pretrained \
	    --is-validation \



| Model | Previous | Test| Pretrained|
|--|--|--|--|
|ResNet18|-| | [resnet18]() | 
|ResNet50 | 76.82 | | [resnet50]() | 


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

