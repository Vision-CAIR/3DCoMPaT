import os
import glob
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
# from timm3d.dataset.build import DATASETS
import logging
import torch

import pdb

"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/modelnet_cls
"""


# @DATASETS.register_module()
class CompatSeg(Dataset):

    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """

    def __init__(self,
                 data_root='data/compat',
                 test_area=5,
                 voxel_size=0.04, voxel_max=None,
                 num_points=4096,
                 split='train',
                 transform=None, shuffle_index=False, loop=1, merge=False
                 ):
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.seg, self.feat, model_id = load_data(data_root, self.partition)
        self.label = self.get_model_ids(model_id, data_root)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
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

    def get_model_ids(self, index, data_dir):
        # test_dir = '/ibex/scratch/liy0r/cvpr/BPNet/'
        df = pd.read_csv(data_dir + "/model.csv")
        part_index = dict(zip(df['id'].tolist(), df['model'].tolist()))
        cat = list(set(df['model'].tolist()))
        classes = dict(zip(cat, range(len(cat))))
        label = []
        for key in range(len(index)):
            label.append(classes[part_index[index[key].item()]])
        return np.array(label).astype('int64')  # , classes


def load_data(data_dir, partition):
    print(os.getcwd())
    h5_name = os.path.join(data_dir, 'new{}.hdf5'.format(partition))
    with h5py.File(h5_name, 'r') as f:
        data = np.array(f['pc'][:]).astype('float32')
        seg = np.array(f['seg'][:]).astype('int64')
        color = np.array(f['color']).astype('float32')
        model_id = f['id'][:].astype('str')
        
    return data, seg, color, model_id


# @DATASETS.register_module()
class CompatSegCls(Dataset):

    """
    This is the data loader for ModelNet 40

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """

    def __init__(self,
                 data_root='data/compat',
                 test_area=5,
                 voxel_size=0.04, voxel_max=None,
                 num_classes=43,
                 num_points=4096,
                 split='train',
                 transform=None, shuffle_index=False, loop=1, merge=False
                 ):
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.seg, self.feat, model_ids = load_data(data_root, self.partition)
        self.label = self.get_model_ids(model_ids, data_root)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform
        self.eye=torch.eye(num_classes)
        # print(self.num_classes())

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        seg = self.seg[item][idx]
        pointcloud = torch.from_numpy(pointcloud)
        feat = self.eye[label,].repeat(pointcloud.shape[0], 1)
        seg = torch.from_numpy(seg)
        return pointcloud, feat, seg

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

    def get_model_ids(self, model_ids, data_dir):
        # test_dir = '/ibex/scratch/liy0r/cvpr/BPNet/'
        df = pd.read_csv(data_dir + "/model.csv")
        part_index = dict(zip(df['id'].tolist(), df['model'].tolist()))
        cat = list(set(df['model'].tolist()))
        classes = dict(zip(cat, range(len(cat))))
        label = []
        for i in range(len(model_ids)):
            label.append(classes[part_index[model_ids[i].item()]])
        return np.array(label).astype('int64')  # , classes


def load_data(data_dir, partition):
    print(os.getcwd())
    h5_name = os.path.join(data_dir, 'new{}.hdf5'.format(partition))
    with h5py.File(h5_name, 'r') as f:
        data = np.array(f['pc'][:]).astype('float32')
        seg = np.array(f['seg'][:]).astype('int64')
        color = np.array(f['color']).astype('float32')
        model_ids = f['id'][:].astype('str')

    return data, seg, color, model_ids


def translate_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud
    :param pointcloud:
    :return:
    """
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, scale), shift).astype('float32')
    return translated_pointcloud


if __name__ == '__main__':
    train = CompatSegCls(num_points=4096)


    test = CompatSegCls(num_points=1024, split='test')
    for data, color, label in train:
        print(data.shape)
        print(color.shape)
        print(label.shape)
        # print(dd)
