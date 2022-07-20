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

    def __init__(self, dataPathPrefix='Data', dataPathPrefix2D='/data/compat/2d_image',voxelSize=0.01,
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

        if memCacheInit and (not exists("/dev/shm/wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, 0))):
            print('[*] Starting shared memory init ...')

            # data, part_label_to_idx, count = self.read_data()
            # # pickle_data(f"data/dataset_{exp_name}.pkl", data)
            # print("Reading splits", list(data.keys()))
            # print("Train split: {} models".format(len(data['train']['labels'])))
            # print("Val split: {} models".format(len(data['val']['labels'])))
            # print("Test split: {} models".format(len(data['test']['labels'])))
            splits=next(unpickle_data('/home/liy0r/3d-few-shot/download/BPNet/'+'dataset_nov.pkl'))
            model_ids=splits[split]

            all_models = pd.read_csv('/home/liy0r/3d-few-shot/download/BPNet/data/products_all.csv')
            total_models = len(all_models)
            print("Reading {} models".format(total_models))
            with open('/home/liy0r/3d-few-shot/download/BPNet/data/models_to_be_used.json') as fin:
                models_to_included = json.load(fin)

                # models_to_included = ['0a6da141-f3e2-4097-9f8c-16707f2b8abc'] # todo remove
            with open("/home/liy0r/3d-few-shot/download/BPNet/data/parts_annotation.json") as f:
                new_part_names = json.load(f)
            part_set = set()
            for i in new_part_names.values():
                for j in i.values():
                    part_set.add(j)
            part_map = dict()
            for i, j in enumerate(sorted(list(part_set))):
                part_map[j] = i

            print(len(new_part_names.keys()))

            def should_be_included(record):
                model_id = record['product_id']
                return model_id in models_to_included

            all_models = all_models[all_models.apply(should_be_included, axis=1)]
            WORKING_DIR = '/data/dataset/glbfile'
            indexs = []
            j=0
            for i, model_id in enumerate(list(all_models.product_id)):
                # print(model_id)
                if model_id not in new_part_names and model_id not in model_ids:
                # if model_id not in new_part_names:
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
                    # print("Model:{}, Failed!".format(model_id))
                    # return model_id, None
                colors = {}
                v = {}
                segment = {}
                for g_name, g_mesh in mesh.geometry.items():
                    try:
                        new_part_name = new_part_names[model_id][g_name]
                        # colors[g_name] = np.asarray(mesh.geometry[g_name].visual.to_color().vertex_colors)
                        v[g_name] = np.asarray(mesh.geometry[g_name].vertices)
                        colors[g_name] = np.ones_like(v[g_name])
                        segment[g_name] = np.full(colors[g_name].shape[0], part_map[new_part_name])
                    except:
                        try:

                            for p in new_part_names[model_id].keys():
                                if p in g_name:
                                    new_name= p
                                    break
                            new_part_name = new_part_names[model_id][new_name]
                            # FOR STYLE MODEL WE USE THIS COLOR, FOR NON STYLE MODEL WE USE 1
                            # colors[g_name] = np.asarray(mesh.geometry[g_name].visual.to_color().vertex_colors)
                            v[g_name] = np.asarray(mesh.geometry[g_name].vertices)
                            colors[g_name] = np.ones_like(v[g_name])
                            segment[g_name] = np.full(colors[g_name].shape[0], part_map[new_part_name])
                        except:
                            print(model_id)
                            print(g_name)
                            print(new_part_names[model_id])

                    # finally:
                    #     print(model_id)
                    #     print(g_name)
                    #     print(new_part_names[model_id])

                a = []
                for key in v.keys():
                    a.append(np.hstack((v[key], colors[key], np.expand_dims(segment[key],axis=1))))
                b = np.vstack(a)

                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, i))
                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, i))
                #     SA.delete("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, i))
                #     print(model_ids)

                sa_create("shm://wbhu_scannet_3d_%s_%06d_locs_%s" % (split, identifier, model_id), b[:, :3])
                sa_create("shm://wbhu_scannet_3d_%s_%06d_feats_%s" % (split, identifier, model_id), b[:, 3:6])
                sa_create("shm://wbhu_scannet_3d_%s_%06d_labels_%s" % (split, identifier, model_id), b[:, 6])
                indexs.append(model_id)
                j+=1
            # np.savetxt('indexs.out', indexs)

            with open("indexs.txt","w") as f:
                for i in indexs:
                    f.write(i+"\n")
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
        print('[*] %s (%s) loading done (%d)! ' % (dataPathPrefix, split, len(indexs)))

    def read_data(self):
        # Initialize the random state
        rnd_state = np.random.RandomState(seed=95)

        # Read the sampled data
        all_data = next(
            unpickle_data(
                '/home/liy0r/3d-few-shot/3dcompat/3dcompat/baselines/parts_segment/parts_point_cloud_v1_seg.pkl'))
        # print("all_data keys: ", len(all_data.keys()))

        # Group by the part name
        part_pc = defaultdict(list)
        model_pc_model_ids = defaultdict(list)

        count = 0
        for model_id, model_pc in all_data.items():
            # if count == 40:
            #     break
            # count += 1
            for part_name, part_data in model_pc.items():
                part_pc[part_data['label']].append((part_data['pc'], part_data['seg']))
                model_pc_model_ids[part_data['label']].append(part_data['model_id'])

        # with open('part_label_to_idx.json') as fin:
        #     part_label_to_idx = json.load(fin)

        df = pd.read_csv("/home/liy0r/3d-few-shot/3dcompat/3dcompat/baselines/parts_segment/model_class_id.csv")
        df['class'] = df['class'].str.lower()
        count = df['class'].str.lower().value_counts().to_dict()
        classes = sorted(df['class'].str.lower().unique())
        part_label_to_idx = {}
        idx = 0
        for i in range(len(classes)):
            if count[classes[i]] < 6:
                continue
            part_label_to_idx[classes[i]] = idx
            idx += 1
        with open('part_label_to_idx.json', 'w') as fp:
            json.dump(part_label_to_idx, fp)

        # Create the splits
        train_data = {
            'pc': [],
            'labels': [],
            'seg': [],
            'model_ids': []
        }

        test_data = {
            'pc': [],
            'labels': [],
            'seg': [],
            'model_ids': []
        }

        val_data = {
            'pc': [],
            'labels': [],
            'seg': [],
            'model_ids': []
        }

        for part_name, pcs in part_pc.items():
            if part_name not in part_label_to_idx.keys():
                continue

            class_count = {'test': {}, 'valid': {}}
            for i_, data_ in enumerate(pcs):
                pc = data_[0]
                seg = data_[1]
                # assert len(pc) == 5000 # Check the number of points is consistent
                if part_name not in class_count['test'].keys():
                    test_data['pc'].append(pc)
                    test_data['labels'].append(part_name)
                    test_data['seg'].append(seg)
                    test_data['model_ids'].append(model_pc_model_ids[part_name][i_])
                    class_count['test'][part_name] = 1
                    continue

                if part_name not in class_count['valid'].keys():
                    val_data['pc'].append(pc)
                    val_data['labels'].append(part_name)
                    val_data['model_ids'].append(model_pc_model_ids[part_name][i_])
                    val_data['seg'].append(seg)
                    class_count['valid'][part_name] = 1
                    continue

                n = rnd_state.rand()

                if n <= 0.85:  # Train
                    train_data['pc'].append(pc)
                    train_data['labels'].append(part_name)
                    train_data['seg'].append(seg)
                    train_data['model_ids'].append(model_pc_model_ids[part_name][i_])
                elif n <= 0.95:  # Test
                    test_data['pc'].append(pc)
                    test_data['labels'].append(part_name)
                    test_data['seg'].append(seg)
                    test_data['model_ids'].append(model_pc_model_ids[part_name][i_])
                else:  # Validation
                    val_data['pc'].append(pc)
                    val_data['labels'].append(part_name)
                    val_data['seg'].append(seg)
                    val_data['model_ids'].append(model_pc_model_ids[part_name][i_])

        count_ = Counter(train_data['labels'])
        count = {}
        for key in part_label_to_idx.keys():
            count[key] = count_[key]
        print("Total Classes: ", len(part_label_to_idx.keys()))
        print("classes in test set: ", len(Counter(test_data['labels'])))
        return {
                   'train': train_data,
                   'test': test_data,
                   'val': val_data
               }, part_label_to_idx, count

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
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
        return


if __name__ == '__main__':
    import time, random
    from tensorboardX import SummaryWriter

    # data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    data_root = '/data/dataset/scannetv2'
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
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
