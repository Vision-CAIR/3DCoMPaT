import os
import glob
# print(os.chdir('.'))
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
from timm3d.dataset.build import DATASETS
import logging

"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/modelnet_cls
"""


@DATASETS.register_module()
class CompatSeg(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """

    def __init__(self,
                 num_points=4096,
                 data_dir="./data/compat",
                 split='train',
                 transform=None
                 ):
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.seg, self.feat, label = load_data(data_dir, self.partition)
        self.label = self.get_model_ids(label, data_dir)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform
        print(self.num_classes())

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        feat = self.feat[item][idx]
        seg = self.seg[item][idx]
        # if self.partition == 'train':
        #     np.random.shuffle(pointcloud)
        # out_dict = {'pts': pointcloud}

        return pointcloud, feat, seg

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
        data = f['pc'][:].astype('float32')
        seg = f['seg'][:].astype('int64')
        color = f['color'].astype('uint8')
        id = f['id'][:].astype('str')
        # seg = f['seg'][:].astype('int64')
        # all_data.append(data)
        # all_label.append(label)
    # all_data = np.concatenate(all_data, axis=0)
    # all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    # split = os.path.join(data_dir, 'compat', 'split.txt')
    # number = 2011 + 82
    return data, seg, color, id
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
    train = CompatSeg(1024)
    test = CompatSeg(1024, split='test')
    for data, color, label in train:
        print(data.shape)
        print(color.shape)
        print(label.shape)

# SFC Order
# Reference: https://github.com/guochengqian/SFC-Net/blob/main/datasets/S3DIS.py
# self.order = order
# if self.order:
#     self.p = p  
#     if not os.path.exists(os.path.join(self.data_dir, f"hilbercurve_table_p{self.p}.npy")):
#         import itertools
#         self.hilbert_curve = HilbertCurve(self.p, 3)
#         voxel_index = [list(range(2**p))]*3
#         voxel_indicies = [np.array(i, dtype=np.int8) for i in itertools.product(*voxel_index)]
#         voxel_indicies = np.vstack(voxel_indicies)
#         hilbercurve_p = self.hilbert_curve.distances_from_points(voxel_indicies)
#         self.hilbercurve_table = np.empty([2**p, 2**p, 2**p], dtype=np.int32)
#         for i in range(len(voxel_indicies)):
#             x, y, z = voxel_indicies[i]
#             self.hilbercurve_table[x, y, z]=hilbercurve_p[i]
#         np.save(os.path.join(self.data_dir, f"hilbercurve_table_p{self.p}.npy"), self.hilbercurve_table)

#     else:
#         self.hilbercurve_table = np.load(os.path.join(self.data_dir, f"hilbercurve_table_p{self.p}.npy"))
