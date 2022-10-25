
import os
import os.path as osp
import glob
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import trimesh

import torch
from torch.utils.data import Dataset
from PIL import Image

class MaterialDataset(Dataset):
    def __init__(self, data_dir="./", split='train', view=None, transforms=None):
        self.data_dir = data_dir

        assert view is not None
        # Read the splits
        splits = next(unpickle_data(os.path.join(data_dir, 'dataset_v1.pkl')))
        train_model_ids = splits['train']
        test_model_ids = splits['test']
        val_model_ids = splits['val']

        # Read the labels
        self.labels = next(unpickle_data('v_{}_labels.pkl'.format(view)))

        # Read the models data
        # self.models = pd.read_csv('v_{}_{}_dataset.csv'.format(view, t))

        # Get a list of all jpeg files in the data dir
        result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]

        # use only models in the split
        self.models = []
        for el in result:
            img_id =  os.path.basename(el).split('.jpg')[0]
            if img_id in self.labels:
                _id = img_id.split('_')[0]

                if split == 'train':
                    if _id in train_model_ids:
                        self.models.append(el)
                elif split == 'val':
                    if _id in val_model_ids:
                        self.models.append(el)
                else:
                    if _id in test_model_ids:
                        self.models.append(el)

        if split == 'train':
            print("training on view: {} with {} images".format(view, len(self.models)))
        elif split == 'val':
            print("validating on view: {} with {} images".format(view, len(self.models)))
        elif split == 'test':
            print("testing on view: {} with {} images".format(view, len(self.models)))

        # Get the transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item):
        # Read the image
        # record = self.models.iloc[item]
        record = self.models[item]

        p = record

        image = Image.open(p)

        # Apply transformation
        image = self.transforms(image)

        # Get the label
        # img_id = record['img_id']
        img_id = os.path.basename(p).split('.jpg')[0]

        label = np.array(self.labels[img_id])

        return image, label


