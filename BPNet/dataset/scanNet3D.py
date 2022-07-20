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

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons)


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

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=True, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1, data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False
                 ):
        super(ScanNet3D, self).__init__()
        self.split = split
        self.identifier = identifier
        # self.data_paths = sorted(glob(join(dataPathPrefix, split, '*.pth')))
        self.data_paths = sorted(glob(join(dataPathPrefix, '*.glb')))

        self.voxelSize = voxelSize
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        lengths = {'train': 1238, 'test': 172, 'val': 112}
        self.length = lengths[split]
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

        with open("data/parts_annotation.json") as f:
            self.new_part_names = json.load(f)
        part_set = set()
        for i in self.new_part_names.values():
            for j in i.values():
                part_set.add(j)
        self.part_map = dict()
        self.part_map['background'] = 0
        for i, j in enumerate(sorted(list(part_set))):
            self.part_map[j] = i + 1
        self.part_ids = {}
        print(len(self.data_paths), "here is the size of dataset {}".format(split))
        if memCacheInit and (not exists("/dev/shm/wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, 0))):
            print('[*] Starting shared memory init ...')

            # data, part_label_to_idx, count = self.read_data()
            # # pickle_data(f"data/dataset_{exp_name}.pkl", data)
            # print("Reading splits", list(data.keys()))
            # print("Train split: {} models".format(len(data['train']['labels'])))
            # print("Val split: {} models".format(len(data['val']['labels'])))
            # print("Test split: {} models".format(len(data['test']['labels'])))
            splits = next(unpickle_data('dataset_nov.pkl'))
            model_ids = splits[split]
            # dataset = ObjectDataset(data, split, part_label_to_idx)
            # weights = make_weights_for_balanced_classes(dataset, len(part_label_to_idx.keys()))
            # weights = torch.DoubleTensor(weights)
            # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

            # The data loaders
            # dloader = torch.utils.data.DataLoader(dataset, batch_size=1,
            #                                             drop_last=True, sampler=sampler)
            all_models = pd.read_csv('data/products_all.csv')
            total_models = len(all_models)
            print("Reading {} models".format(total_models))
            with open('data/models_to_be_used.json') as fin:
                models_to_included = json.load(fin)

                # models_to_included = ['0a6da141-f3e2-4097-9f8c-16707f2b8abc'] # todo remove

            print(len(self.new_part_names.keys()))

            def should_be_included(record):
                model_id = record['product_id']
                return model_id in models_to_included

            all_models = all_models[all_models.apply(should_be_included, axis=1)]
            WORKING_DIR = dataPathPrefix
            indexs = []
            j = 0
            for i, model_id in enumerate(model_ids):
                # print(model_id)
                # if model_id not in new_part_names and model_id not in model_ids:

                if model_id not in self.new_part_names:
                    continue

                gltf_path = '{}/{}.glb'.format(WORKING_DIR, model_id)
                try:
                    gltf_stats = os.stat(gltf_path)
                    model_size = gltf_stats.st_size / 1024000
                except:
                    print("model can't found {}".format(model_id))
                    continue
                # print('Model:{} is of size {}'.format(model_id, model_size))
                # return model_id, None
                #
                if model_size < 0.09:  # Ignore extremely small (corrupted) models
                    print("Error reading2")
                    print("Model:{}, Failed!".format(model_id))
                    continue
                    # return model_id, None

                # Try to sample now
                try:
                    mesh = trimesh.load(gltf_path)
                    print("read successfully")
                except:
                    print("Error in reading")
                    continue
                    # print("Model:{}, Failed!".format(model_id))
                    # return model_id, None
                colors = {}
                v = {}
                segment = {}
                for g_name, g_mesh in mesh.geometry.items():
                    try:

                        new_part_name = self.new_part_names[model_id][g_name]
                        # colors[g_name] = np.asarray(mesh.geometry[g_name].visual.to_color().vertex_colors)
                        v[g_name] = np.asarray(mesh.geometry[g_name].vertices)
                        colors[g_name] = np.ones_like(v[g_name])
                        segment[g_name] = np.full(colors[g_name].shape[0], self.part_map[new_part_name])
                    except:
                        try:

                            for p in self.new_part_names[model_id].keys():
                                if p in g_name:
                                    new_name = p
                                    break
                            new_part_name = self.new_part_names[model_id][new_name]
                            # FOR STYLE MODEL WE USE THIS COLOR, FOR NON STYLE MODEL WE USE 1
                            # colors[g_name] = np.asarray(mesh.geometry[g_name].visual.to_color().vertex_colors)
                            v[g_name] = np.asarray(mesh.geometry[g_name].vertices)
                            colors[g_name] = np.ones_like(v[g_name])
                            segment[g_name] = np.full(colors[g_name].shape[0], self.part_map[new_part_name])
                        except:
                            print(model_id)
                            print(g_name)
                            print(self.new_part_names[model_id])

                    # finally:
                    #     print(model_id)
                    #     print(g_name)
                    #     print(new_part_names[model_id])

                a = []
                for key in v.keys():
                    a.append(np.hstack((v[key], colors[key], np.expand_dims(segment[key], axis=1))))
                b = np.vstack(a)

                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, i))
                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, i))
                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, i))
                #     print(model_ids)

                sa_create("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, j), b[:, :3])
                sa_create("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, j), b[:, 3:6])
                sa_create("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, j), b[:, 6])
                indexs.append(model_id)
                j += 1

            # self.length = len(indexs)
            # np.savetxt('indexs.out', indexs)
            self.data_paths = indexs
            with open("indexs_{}.txt".format(split), "w") as f:
                for i in indexs:
                    f.write(i + "\n")
            # for i, (locs, feats, labels, model_ids) in enumerate(dataset):
            #     labels[labels == -100] = 255
            #     labels = labels.astype(np.uint8)
            #     # Scale color to 0-255
            #     feats = (feats + 1.) * 127.5
            #
            #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, i))
            #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, i))
            #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, i))
            # print(model_ids)
            # sa_create("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, i), locs)
            # sa_create("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, i), feats)
            # sa_create("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, i), labels)
        else:
            with open("indexs_{}.txt".format(split), "r") as f:
                model_ids = f.readlines()
            self.data_paths = model_ids
        print('[*] %s (%s) loading done (%d)! ' % (dataPathPrefix, split, self.length))

    def __getitem__(self, index_long):
        index = index_long % self.length
        print(index_long)
        print(self.length)
        locs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (self.split, self.identifier, index)).copy()
        feats_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (self.split, self.identifier, index)).copy()
        labels_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (self.split, self.identifier, index)).copy()
        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(locs, feats_in, labels_in)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import time, random
    from tensorboardX import SummaryWriter

    # data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    # data_root = '/data/dataset/scannetv2'
    data_root = '/data/dataset/glbfile'
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=False, num_workers=4, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    trainLog = SummaryWriter('Exp/scannet/statistic/train')
    valLog = SummaryWriter('Exp/scannet/statistic/val')

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

        for step, (coords, feat, label, inds_reverse) in enumerate(val_loader):
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
