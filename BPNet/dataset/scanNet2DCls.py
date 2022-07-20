#!/usr/bin/env python
"""
    File Name   :   CoSeg-scanNet3D
    date        :   14/10/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import imageio

from dataset.utils import unpickle_data, pickle_data
import pandas as pd
import json
import torch.utils.data as data
import torch
import os
import os.path as osp
import trimesh
import numpy as np
from os.path import join, exists
from glob import glob
import multiprocessing as mp
import SharedArray as SA
import dataset.augmentation as t
from dataset.voxelizer import Voxelizer
from collections import defaultdict, Counter
import dataset.augmentation_2d as t_2d


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=float)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def collation_fn(batch):
    """

    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels = list(zip(*batch))

    return torch.stack(coords), torch.stack(feats), torch.stack(labels)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels = list(zip(*batch))

    return torch.stack(coords), torch.stack(feats), torch.stack(labels)


import numpy as np
from torch.utils.data import Dataset

N_POINTS = 1024


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ObjectDataset(Dataset):
    def __init__(self, data_dict: dict, type: str, class_to_idx, n_points=N_POINTS):
        # Create a list of objects
        self.part_pc = data_dict[type]['pc']
        self.part_labels = data_dict[type]['labels']
        self.seg = data_dict[type]['seg']
        # self.model_class = data_dict[type]['class']
        self.model_ids = data_dict[type]['model_ids']
        assert len(self.part_labels) == len(self.part_pc)

        self.part_label_to_idx = class_to_idx
        self.n_points = n_points

    def __getitem__(self, index):
        part_pc = self.part_pc[index]
        part_pc_label = self.part_labels[index]
        segment = self.seg[index]
        # model_class = self.model_class[index]

        idx = np.random.choice(np.arange(0, part_pc.shape[0]), self.n_points,
                               replace=self.n_points > part_pc.shape[0])
        model_ids = self.model_ids[index]
        # return {
        #     'pc': pc_normalize(part_pc[idx]),
        #     'label': self.part_label_to_idx[part_pc_label], # TODO-5 uncomment this line
        # }
        return pc_normalize(part_pc[idx]), self.part_label_to_idx[part_pc_label], segment[idx], model_ids

    def __len__(self):
        return len(self.part_pc)


class ScanNet3D(data.Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    VIEW_NUM = 4
    IMG_DIM = (400, 400)

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=True, identifier=1244, loop=1,
                 data_aug_color_trans_ratio=0.1, data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False
                 ):
        super(ScanNet3D, self).__init__()
        self.split = split
        self.identifier = identifier
        self.voxelSize = voxelSize
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        # lengths = {'train': 1238, 'test': 172, 'val': 112}
        # self.length = lengths[split]
        self.voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)
        with open("data/newid_{}.txt".format(split), "r") as f:
            model_ids = f.readlines()
        newmodel = []
        for i in model_ids:
            newmodel.append(i.strip())
        # for i in model_ids:
        self.data_paths = newmodel
        self.length = len(self.data_paths)

        with open("data/newid.txt".format(split), "r") as f:
            allids = f.readlines()
        allmodel = []
        for i in allids:
            allmodel.append(i.strip())
        with open("data/parts_annotation.json") as f:
            self.new_part_names = json.load(f)
        part_set = set()
        for i in self.new_part_names.keys():
            if i in allmodel:
                for j in self.new_part_names[i].values():
                    part_set.add(j)

        self.part_map = dict()
        self.part_map['background'] = 0
        for i, j in enumerate(sorted(list(part_set))):
            self.part_map[j] = i + 1
        # self.part_ids = {}
        # model_ids = []
        print("our part number is here", len(part_set))
        print(self.length, "here are the sizes of dataset {}".format(split))
        # print("")
        model_class_id = pd.read_csv("data/model_class_id.csv")
        with open("data/newid_{}.txt".format(split), "r") as f:
            model_ids = f.readlines()
        cls = []
        self.cls = []
        for i in model_ids:
            i = i[:-1]
            cls.append(model_class_id[model_class_id["product_id"] == i]['class'].values[0].lower())
        cls_dic = Counter(cls)
        cls_ids = {}
        for i, k in enumerate(cls_dic):
            cls_ids[k] = i
        for i in range(len(cls)):
            self.cls.append(cls_ids[cls[i]])
        print(len(self.cls))
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        if self.aug:
            self.transform_2d = t_2d.Compose([
                # t_2d.RandomGaussianBlur(),
                # t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='rand', padding=mean,
                #           ignore_label=0),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            self.transform_2d = t_2d.Compose([
                # t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='rand', padding=mean,
                #           ignore_label=0),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        self.imgs = []
        self.segs = []
        if memCacheInit or (not exists("/dev/shm/wbhu_scannet_2d_%s_%06d_locs_%08d" % (split, identifier, 0))):
            print('[*] Starting shared memory init ...')

            print(len(self.new_part_names.keys()))
            jj = 0
            seg_path = "../seg_maps_v1/"

            for model_id in self.data_paths:
                imgs, labels = [], []

                for view in range(self.VIEW_NUM):
                    f = os.path.join(dataPathPrefix, model_id + "_" + str(view) + ".jpg")
                    # f = self.data2D_paths[v][room_id]
                    model_id = f.split('/')[-1].split('_')[0]
                    try:
                        img = imageio.imread(f)
                    except:
                        print("============================")
                        print(f)
                        img = np.ones((400, 400, 3))
                    # label_ads = os.path.join("/data/dataset/seg_maps_v0/", model_id + "_" + view_id, "segmentation0106.png")
                    label_ads = os.path.join(seg_path, model_id + "_" + str(view) + ".jpg")
                    try:
                        label = imageio.imread(label_ads, as_gray=True).astype(int)
                    except:
                        print(label_ads)
                        label = np.ones((400, 400))

                    # label = self.remapper[label]
                    label = np.floor(np.array(label) / 36).astype(int)
                    # self.part_map
                    according = np.zeros(8).astype(int)
                    model_part = self.new_part_names[model_id]
                    for i, k in enumerate(model_part.keys()):
                        according[i + 1] = self.part_map[model_part[k]]
                    label = according[label]
                    img, label = self.transform_2d(img, label)
                    # img, label = torch.tensor(img), torch.tensor(label)
                    imgs.append(img)
                    labels.append(label)
                imgs = torch.stack(imgs, dim=-1)
                labels = torch.stack(labels, dim=-1)
                self.imgs.append(imgs)
                self.segs.append(labels)
                # if jj == 7:
                #     self.length = 8
                #     break
                # sa_create("shm://wbhu_scannet_2d_%s_%06d_locs_%08d" % (split, identifier, jj), imgs)
                # sa_create("shm://wbhu_scannet_2d_%s_%06d_feats_%08d" % (split, identifier, jj), labels)
                jj += 1
        print('[*] %s (%s) loading done (%d)! ' % (dataPathPrefix, split, self.length))

    def __getitem__(self, index_long):
        index = index_long % self.length
        # print(self.length)
        # img = SA.attach("shm://wbhu_scannet_2d_%s_%06d_locs_%08d" % (self.split, self.identifier, index)).copy()
        # seg = SA.attach("shm://wbhu_scannet_2d_%s_%06d_feats_%08d" % (self.split, self.identifier, index)).copy()
        img = self.imgs[index]
        seg = self.segs[index]
        category = torch.tensor(self.cls[index])
        img = img
        seg = torch.LongTensor(seg)
        return img, seg, category

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import time, random
    from tensorboardX import SummaryWriter

    # data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    data_root = '/ibex/scratch/liy0r/cvpr/no_style_v1'
    train_data = ScanNet3D(dataPathPrefix=data_root, aug=True, split='train', memCacheInit=True, voxelSize=0.05)
    val_data = ScanNet3D(dataPathPrefix=data_root, aug=False, split='val', memCacheInit=True, voxelSize=0.05,
                         eval_all=True)

    manual_seed = 123


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=8, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    trainLog = SummaryWriter('Exp/scannet/statistic/train')
    valLog = SummaryWriter('Exp/scannet/statistic/val')
    coords, feats, labels = train_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape)

    coords, feats, labels = val_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape)
    for idx in range(1):
        end = time.time()
        for step, (coords, feat, label) in enumerate(train_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
            trainLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            trainLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            trainLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            trainLog.add_histogram('color', feat, global_step=step)
            # time.sleep(0.3)
            end = time.time()

        for step, (coords, feat, label) in enumerate(val_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(val_loader), time.time() - end))
            valLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            valLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            valLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            valLog.add_histogram('color', feat, global_step=step)
            # time.sleep(0.3)
            end = time.time()

    trainLog.close()
    valLog.close()
