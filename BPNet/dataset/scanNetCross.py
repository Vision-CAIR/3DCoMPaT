#!/usr/bin/env python
import math
import random
import torch
import numpy as np
from os.path import join
from glob import glob
import SharedArray as SA
import imageio
import os
import dataset.augmentation_2d as t_2d
from dataset.scanNet3D import ScanNet3D
from dataset.utils import unpickle_data, pickle_data
from collections import defaultdict, Counter
import tqdm
import json
import pandas as pd
import trimesh
import pandas as pd
from collections import Counter


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class LinkCreator(object):
    def __init__(self, fx=2666.6667, fy=2666.6667, mx=960.0000, my=540.0000, image_dim=(400, 400), voxelSize=0.05):
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[1920, 1080], image_dim=image_dim)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    #
    def computeLinking(self, camera_to_world, coords, depth):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        # path = "0bafd365-01d2-491f-a308-2ef20e35af34"
        # trimesh.load(path)
        link = np.zeros((3, coords.shape[0]), dtype=np.int)
        # coords=(coords + 1) / 2
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T

        assert coordsNew.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coordsNew)
        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = -((p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2])
        pi = np.round(p).astype(np.int)
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                - p[2][inside_mask]) <= self.voxel_size
        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1
        return link.T


class ScanNetCross(ScanNet3D):
    VIEW_NUM = 4
    IMG_DIM = (400, 400)

    def __init__(self, dataPathPrefix='/ibex/scratch/liy0r/cvpr/glbfile',
                 dataPathPrefix2D='/ibex/scratch/liy0r/cvpr/no_style_v1',
                 voxelSize=0.05,
                 split='train', aug=True, memCacheInit=False,
                 identifier=7796, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False,
                 val_benchmark=False

                 ):
        super(ScanNetCross, self).__init__(dataPathPrefix=dataPathPrefix,
                                           voxelSize=voxelSize,
                                           split=split, aug=aug, memCacheInit=memCacheInit,
                                           identifier=identifier, loop=loop,
                                           data_aug_color_trans_ratio=data_aug_color_trans_ratio,
                                           data_aug_color_jitter_std=data_aug_color_jitter_std,
                                           data_aug_hue_max=data_aug_hue_max,
                                           data_aug_saturation_max=data_aug_saturation_max,
                                           eval_all=eval_all)
        self.val_benchmark = val_benchmark
        if self.val_benchmark:
            self.offset = 0
        # Prepare for 2D

        self.data2D_paths = [[] for i in range(self.VIEW_NUM)]
        self.seg2D_paths = [[] for i in range(self.VIEW_NUM)]
        self.depth2D_paths = [[] for i in range(self.VIEW_NUM)]
        # self.data2D_paths = [[] for i in range(self.VIEW_NUM)]

        # data_path = "/home/liy0r/3d-few-shot/3dcompat/3dcompat/baselines/material_classifier/"
        # splits = next(unpickle_data(data_path + 'dataset_v1.pickle'))
        splits = next(unpickle_data('dataset_nov.pkl'))
        seg_path = "../seg_maps_v1/"
        depth_path = "../depth_maps_v1"
        # first=True
        # self.img=""
        # self.img = dataPathPrefix2D + '/a_rem_c3/2340000/548b2dbf-51d7-47e4-ba37-ac69d88fe73d_789719_3.jpg'
        model_class_id = pd.read_csv("data/model_class_id.csv")
        with open("indexs_{}.txt".format(split), "r") as f:
            model_ids = f.readlines()
        self.cls = []

        # for i in model_ids:
        #     i = i[:-1]
        #     self.cls.append(model_class_id[model_class_id["product_id"] == i]['class'].values[0].lower())
        # cls_dic = Counter(self.cls)
        # self.cls = cls_dic[self.cls]
        for view in range(self.VIEW_NUM):

            # Read the models data
            # self.data2D_paths = pd.read_csv('v_{}_{}_dataset.csv'.format(view, t))
            # Get a list of all jpeg files in the data dir
            # self.labels = next(unpickle_data(data_path + 'v_{}_labels.pkl'.format(view)))
            # data_dir = os.path.join(dataPathPrefix2D, "a_rem_c{}".format(view))

            # result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
            # result = [os.path.join(dataPathPrefix2D,x) for x in os.listdir(dataPathPrefix2D)]
            for i in model_ids:
                i = i[:-1]
                # self.model_ids.append(i)
                self.data2D_paths[view].append(os.path.join(dataPathPrefix2D, i + "_" + str(view) + ".jpg"))
                self.seg2D_paths[view].append(os.path.join(seg_path, i + "_" + str(view) + ".jpg"))
                self.depth2D_paths[view].append(os.path.join(depth_path, i + "_" + str(view) + ".jpg"))

            # for el in result:
            #     img_id = os.path.basename(el).split('.jpg')[0]
            #     # if img_id in self.labels:
            #     #     _id = img_id.split('_')[0]
            #     _id = img_id.split('_')[0]
            #     if split == 'train':
            #         if _id in train_model_ids:
            #             self.data2D_paths[view].append(el)
            #     elif split == 'val':
            #         if _id in val_model_ids:
            #             self.data2D_paths[view].append(el)
            #     else:
            #         if _id in test_model_ids:
            #             self.data2D_paths[view].append(el)
            #     # break

            if split == 'train':
                print("training on view: {} with {} images".format(view, len(self.data2D_paths[view])))
            elif split == 'val':
                print("validating on view: {} with {} images".format(view, len(self.data2D_paths[view])))

            assert len(self.data2D_paths[view]) == len(self.data_paths)
        # self.remapper = np.ones(256) * 255
        # for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        #     self.remapper[x] = i

        self.linkCreator = LinkCreator(voxelSize=voxelSize)
        # 2D AUG
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        if self.aug:
            self.transform_2d = t_2d.Compose([
                t_2d.RandomGaussianBlur(),
                t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='rand', padding=mean,
                          ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            self.transform_2d = t_2d.Compose([
                t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='rand', padding=mean,
                          ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])

    def __getitem__(self, index_long):
        index = index_long % self.length
        locs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (self.split, self.identifier, index))
        feats_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (self.split, self.identifier, index))
        labels_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (self.split, self.identifier, index))
        # print(index)
        colors, labels_2d, links = self.get_2d(index, locs_in)

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct, links = self.voxelizer.voxelize(locs, feats_in, labels_in, link=links)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, colors, labels_2d, links, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels, colors, labels_2d, links

    def get_2d(self, room_id, coords: np.ndarray):
        """
        :param      room_id:
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor
        """
        # frames_path = self.data2D_paths[room_id]
        # partial = int(len(frames_path) / self.VIEW_NUM)
        imgs, labels, links = [], [], []
        # coords = coords[0, :, :]
        for v in range(self.VIEW_NUM):
            f = self.data2D_paths[v][room_id]
            # f = self.img
            model_id = f.split('/')[-1].split('_')[0]
            # material_id = f.split("_")[-2]
            # view_id = f.split("_")[-1][0]

            try:
                img = imageio.imread(f)
            except:

                img = np.ones((400, 400, 3))
            # label_ads = os.path.join("/data/dataset/seg_maps_v0/", model_id + "_" + view_id, "segmentation0106.png")
            label_ads = self.seg2D_paths[v][
                room_id]  # os.path.join("../seg_maps_v1/", model_id + "_" + view_id + ".jpg")
            try:
                label = imageio.imread(label_ads, as_gray=True).astype(int)
            except:
                label = np.ones((400, 400))

            # label = self.remapper[label]
            label = np.floor(np.array(label) / 36).astype(int)
            # self.part_map
            according = np.zeros(8).astype(int)
            model_part = self.new_part_names[model_id]
            for i, k in enumerate(model_part.keys()):
                according[i + 1] = self.part_map[model_part[k]]
            label = according[label]
            # self.remapper = np.ones(256) * 255
            # for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            #     self.remapper[x] = i

            depth_ads = self.depth2D_paths[v][
                room_id]  # os.path.join("../depth_maps_v1", model_id + "_" + view_id + ".jpg")
            try:
                depth = imageio.imread(depth_ads, as_gray=True) / 1000.0  # convert to meter
            except:
                depth = np.ones((400, 400))
            posePath = '../2d_image/{}.txt'.format(v)
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                 (x.split(" ") for x in open(posePath).read().splitlines())])
            link = np.ones([coords.shape[0], 4], dtype=np.int64)
            link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)
            img, label = self.transform_2d(img, label)
            imgs.append(img)
            labels.append(label)
            links.append(link)

        imgs = torch.stack(imgs, dim=-1)
        labels = torch.stack(labels, dim=-1)
        links = np.stack(links, axis=-1)
        links = torch.from_numpy(links)
        return imgs, labels, links


def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    """
    coords, feats, labels, colors, labels_2d, links = list(zip(*batch))
    # pdb.set_trace()

    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    """
    coords, feats, labels, colors, labels_2d, links, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.cat(inds_recons)


if __name__ == '__main__':
    import time
    from tensorboardX import SummaryWriter

    # data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    # data_root = '/data/dataset/processed_models'
    # data_root2d = '/data/compat/2d_image'
    # data_root = '/data/dataset/glbfile'
    train_data = ScanNetCross(aug=True, split='train',
                              memCacheInit=True, voxelSize=0.05)
    val_data = ScanNetCross(aug=False, split='val',
                            memCacheInit=True, voxelSize=0.05,
                            eval_all=True)

    coords, feats, labels, colors, labels_2d, links = train_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape)

    coords, feats, labels, colors, labels_2d, links, inds_recons = val_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_recons.shape)
    # exit(0)

    manual_seed = 123


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    # _ = iter(train_loader).__next__()
    trainLog = SummaryWriter('Exp/scannet/statistic_cross/train')
    valLog = SummaryWriter('Exp/scannet/statistic_cross/val')

    # for i, batch_data in enumerate(train_loader):
    #     print(i)

    for idx in range(1):
        end = time.time()
        for step, (coords, feats, labels, colors, labels_2d, links) in enumerate(train_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
            trainLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            trainLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            trainLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            trainLog.add_histogram('color', feats, global_step=step)
            trainLog.add_histogram('2D_image', colors, global_step=step)
            # time.sleep(0.3)
            end = time.time()

        for step, (coords, feats, labels, colors, labels_2d, links, inds_reverse) in enumerate(val_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(val_loader), time.time() - end))
            valLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            valLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            valLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            valLog.add_histogram('color', feats, global_step=step)
            valLog.add_histogram('2D_image', colors, global_step=step)
            # time.sleep(0.3)
            end = time.time()

    trainLog.close()
    valLog.close()
