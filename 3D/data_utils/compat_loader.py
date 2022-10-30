""""
Dataloaders for the preprocessed point clouds from 3D 3DCoMPaT dataset.
"""

import os
import glob
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
import logging
import torch
import json
import pdb

def load_data(data_dir, partition):
    print(os.getcwd())
    h5_name = os.path.join(data_dir, 'new{}.hdf5'.format(partition))
    with h5py.File(h5_name, 'r') as f:
        data = np.array(f['pc'][:]).astype('float32')
        seg = np.array(f['seg'][:]).astype('int64')
        color = np.array(f['color']).astype('float32')
        model_id = f['id'][:].astype('str')
        
    return data, seg, color, model_id

class CompatLoader3D(Dataset):

    """
    Base class for loading preprocessed 3D point clouds.

    Args:
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(self,
                meta_dir = '../../metadata/',
                data_root='data/compat',
                split='train',
                num_points=4096,
                transform=None,
                ):
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.seg, self.feat, model_ids = load_data(data_root, self.partition)
        
        f = open(meta_dir + 'labels.json')
        all_labels = json.load(f)
        labels = []
        for sid in model_ids:
            labels.append(all_labels[sid[0]])

        self.label = np.array(labels)

        logging.info(f'==> sucessfully loaded {self.partition} data')

        self.num_points = num_points
        self.transform = transform
        print('num_classes', self.num_classes())
        # pdb.set_trace()

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        feat = self.feat[item][idx]
        seg = self.seg[item][idx]

        pointcloud = torch.from_numpy(pointcloud)
        feat = torch.from_numpy(feat)
        seg = torch.from_numpy(seg)
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1


class CompatLoader3DCls(Dataset):

    """
    Classification data loader using preprocessed 3D point clouds.

    Args:
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(self,
                meta_dir = '../../metadata/',
                data_root='data/compat',
                split='train',
                num_points=4096,
                transform=None,
                ):
        super().__init__(data_root, split, num_points, transform)

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        feat = self.feat[item][idx]
        seg = self.seg[item][idx]

        pointcloud = torch.from_numpy(pointcloud)
        feat = torch.from_numpy(feat)
        seg = torch.from_numpy(seg)
        return pointcloud, label

if __name__ == '__main__':
    train = CompatLoader3D(meta_dir = '../../metadata/', data_root = '../data/', num_points=4096)
    for data, label, color in train:
        print(data.shape)
        print(color.shape)
        print(label.shape)
