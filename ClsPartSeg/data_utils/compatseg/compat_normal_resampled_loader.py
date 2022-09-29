'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from ..build import DATASETS
import torch
import logging

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class compat(Dataset):
    def __init__(self,
                 data_dir, num_points, num_classes,
                 use_normals=False,
                 split='train',
                 transform=None
                 ):
        self.root = os.path.join(data_dir, 'compat')
        self.npoints = num_points
        self.use_normals = use_normals
        self.num_category = num_classes
        self.preprocess = False
        self.uniform = True
        split = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.split = split

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        logging.info('The size of %s data is %d' % (split, len(self.datapath)))

        self.transform = transform

        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.pkl' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.pkl' % (self.num_category, split, self.npoints))

        if self.preprocess:
            # TODO: + parallel pre-processing. 
            if not os.path.exists(self.save_path):
                logging.info('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                logging.info('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.preprocess:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
            # if self.uniform:
            #     point_set = farthest_point_sample(point_set, self.npoints)
            # else:
            #     point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]
    # def _get_item(self, index):
    #     # TODO: write a pre-processed. 
    #     # if self.preprocess:
    #     #     point_set, label = self.list_of_points[index], self.list_of_labels[index]
    #     # else:
    #     fn = self.datapath[index]
    #     cls = self.classes[self.datapath[index][0]]
    #     label = np.array([cls]).astype(np.int32)
    #     point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

    #     # if self.uniform:
    #     #     point_set = farthest_point_sample(point_set, self.npoints)
    #     # else:
    #     #     point_set = point_set[0:self.npoints, :]

    #     point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    #     if not self.use_normals:
    #         point_set = point_set[:, 0:3]

    #     return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs]
        out_dict = {'pts': current_points}

        if self.transform is not None:
            out_dict = self.transform(out_dict)
        return out_dict, label
