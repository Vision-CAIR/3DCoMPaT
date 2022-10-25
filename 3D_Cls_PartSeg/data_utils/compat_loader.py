import os
import glob
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
# from ..build import DATASETS
import logging
import pdb

"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/modelnet_cls
"""


# @DATASETS.register_module()
class Compat(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/compat",
                 split='train',
                 transform=None,
                 class_choices=None, mapped_labels=None
                 ):
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.seg, label = load_data(data_dir, self.partition)
        self.label = self.get_model_ids(label, data_dir)
        print('number of shapes and classes', len(self.data), self.get_num_classes())
        if class_choices is not None:
            cls_idx = []
            label_new = []
            for i, c in enumerate(class_choices):
                c_idx = np.where(self.label==c)[0]
                cls_idx.extend(c_idx)
                label_new.extend(len(c_idx)*[mapped_labels[i]])
            cls_idx = np.array(cls_idx)
            self.data = self.data[cls_idx]
            self.label = np.array(label_new)
            print('selected number of shapes', len(self.data), np.unique(self.label))
            
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item][np.random.choice(5000, self.num_points, False)]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        out_dict = {'x': pointcloud, 'y': label}
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def get_num_classes(self):
        return np.max(self.label) + 1

    def get_model_ids(self, index, data_dir):
        # test_dir = '/ibex/scratch/liy0r/cvpr/BPNet/'
        df = pd.read_csv(data_dir + "/model.csv")
        part_index = dict(zip(df['id'].tolist(), df['model'].tolist()))
        cat = sorted(list(set(df['model'].tolist())))
        classes = dict(zip(cat, range(len(cat))))
        label = []
        print(classes)
        for key in range(len(index)):
            label.append(classes[part_index[index[key].item()]])
        return np.array(label).astype('int64')  # , classes


def load_data(data_dir, partition):
    print(os.getcwd())
    h5_name = os.path.join(data_dir, 'new{}.hdf5'.format(partition))
    with h5py.File(h5_name, 'r') as f:
        data = f['pc'][:].astype('float32')
        seg = f['seg'][:].astype('int64')
        id = f['id'][:].astype('str')
        # seg = f['seg'][:].astype('int64')
        # all_data.append(data)
        # all_label.append(label)
    # all_data = np.concatenate(all_data, axis=0)
    # all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    # split = os.path.join(data_dir, 'compat', 'split.txt')
    # number = 2011 + 82
    return data, seg, id
    # if partition=="train":
    #     return data[:number], seg[:number], label[:number]
    # else:
    #     return data[number:], seg[number:], label[number:]


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
    train = Compat(1024)
    test = Compat(1024, split='test')
    for data, label in train:
        print(data['pts'].shape)
        print(label)


