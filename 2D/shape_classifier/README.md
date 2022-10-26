# 2D Shape Classification

We provide here details about the 

### 0. Requirements
To be updated.

### 1. Dataset Preparation
To be updated.

### 2. Training
To start training, run the `main.py` script:

`python3 main.py [options]`

Parameters are as follows:

**Input/Output**
- `num-workers`: Number of subprocesses to use for the dataloader.
- `use-tmp`: Use local temporary cache when loading the dataset.
- `root-url`: Root URL for WebDataset shards.


**Dataset**
- `n-comp`: Total number of compositions to use for training and evaluation.
- `view-type`: Train on a specific view type (one of `canonical`, `random` or `all` (default)).


**Training**
- `batch-size`: Size of each batch (same for generation/filtering/training, default: 64).
- `weight-decay`: SGD Weight decay.
- `momentum`: SGD Momentum.

- `nbatches`: Maximum number of batches to use while training.
- `resnet-type`: ResNet variant to be used for training, (one of `resnet10` or `resnet50`).
- `use-pretrained`: Use a model pre-trained on ImageNet.

- `resume`: Resume training with the last saved model.
- `seed`: Random seed.

**Output**

- `models-dir`: Output model directory to use to save models.

An example run is provided below, for a batch size of `64`, with `10` compositions while training on canonical views only from an ImageNet-pretrained ResNet50:

```
	python main.py --num-workers 4 \
	    --use-tmp \
	    --root-url ./shards/ \
	    --models-dir ./models_out/ \
	    --batch-size 64 \
	    --nbatches 10000 \
	    --num-classes 43 \
	    --resnet-type resnet50 \
	    --seed 0 \
	    --n-comp 10 \
	    --view-type canonical \
	    --use-pretrained \
```

### 3. Evaluation and pretrained models
To be updated.

| Model | Previous | Test| Pretrained|
|--|--|--|--|
|ResNet18|-| | [resnet18]() | 
|ResNet50 | 76.82 | | [resnet50]() | 

