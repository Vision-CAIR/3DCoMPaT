# 2D Material Tagging

### 0. Requirements
To be updated.

### 1. Dataset Preparation
To be updated.

### 2. Training
Run the following script:

	python main_material.py \
		--data ./data \
		--view -1 \
		--arch resnet50 \
		--workers 4 \
		--epochs 20 \
		--batch-size 256 \
		--learning-rate 0.1 \
		--print-freq 10 \
		--pretrained True \

Params:

- arch indicates the model used for material tagging provided in torchvision.

- view indicates the camera view type, choose from ['canonical', 'random', 'all'].

Please check main_material.py for details about hyperparameters.


### 3. Evaluation and pretrained models
Run the following script:

	python main_material.py \
		--data ./data \
		--view -1 \
		--arch resnet50 \
		--workers 4 \
		--epochs 20 \
		--batch-size 256 \
		--learning-rate 0.1 \
		--print-freq 10 \
		--pretrained True \
		--evaluate


| Model | Previous | Test| Pretrained|
|--|--|--|--|
|ResNet50 | F1=0.53, AP=0.67 | | [resnet50]() | 
