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
from dataset.scanNet3DCls import ScanNet3D
from dataset.utils import unpickle_data, pickle_data
from collections import defaultdict, Counter
import tqdm
import json
import pandas as pd
import trimesh


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
                 dataPathPrefix2D='/ibex/scratch/liy0r/cvpr/img1', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=False,
                 identifier=7793, loop=1,
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

        # self.data2D_paths = [[] for i in range(self.VIEW_NUM)]
        # data_path = "/home/liy0r/3d-few-shot/3dcompat/3dcompat/baselines/material_classifier/"
        # splits = next(unpickle_data(data_path + 'dataset_v1.pickle'))
        # splits =next(unpickle_data('/home/liy0r/3d-few-shot/download/BPNet/'+'dataset_nov.pkl'))
        data_path = '../'
        self.data2D_paths = [[] for i in range(self.VIEW_NUM)]
        self.seg2D_paths = [[] for i in range(self.VIEW_NUM)]
        self.depth2D_paths = [[] for i in range(self.VIEW_NUM)]
        self.model_maps = {}
        seg_path = "../seg_maps_v1"
        depth_path = "../depth_maps_v1"
        model_class_id = pd.read_csv("data/model_class_id.csv")
        with open("data/newid_{}.txt".format(split), "r") as f:
            model_ids = f.readlines()
        cls = []
        self.cls = []
        self.model_ids = []
        # part materials cooresponding index of model id
        self.part_materials = {}
        # self.model_maps = {}
        for j, i in enumerate(model_ids):
            i = i[:-1]
            self.model_ids.append(i)
            self.part_materials[i] = j
            self.model_maps[i] = 0
            cls.append(model_class_id[model_class_id["product_id"] == i]['class'].values[0].lower())
        cls_dic = Counter(cls)
        cls_ids = {}
        for i, k in enumerate(cls_dic):
            cls_ids[k] = i
        for i in range(len(cls)):
            self.cls.append(cls_ids[cls[i]])
        model_set = set(self.model_ids)
        # self.part_materials = {}

        with open("data/material_indexing.json") as f:
            self.material_index = json.load(f)
        # model maps

        data_dir = glob(os.path.join(dataPathPrefix2D, "a_rem_c0", '*.jpg'))
        print(len(data_dir))
        for el in data_dir:
            img_id = os.path.basename(el).split('.jpg')[0]
            mat_id = os.path.basename(el)[:-5]
            jpg = el[-4:]
            _id = img_id.split('_')[0]
            if _id in model_set:
                # if self.model_maps[_id] >= 500:
                #     continue
                for view in range(4):
                    # el.split('/')
                    tmp = os.path.join(dataPathPrefix2D, "a_rem_c{}".format(view), mat_id + str(view) + jpg)
                    if os.path.exists(tmp):
                        self.model_maps[_id] += 1
                        self.data2D_paths[view].append(tmp)
                        self.seg2D_paths[view].append(os.path.join(seg_path, _id + "_" + str(view) + ".jpg"))
                        self.depth2D_paths[view].append(os.path.join(depth_path, _id + "_" + str(view) + ".jpg"))
                    else:
                        print("There is no file named {}".format(tmp))
        for view in range(4):
            if split == 'train':
                print("training on view: {} with {} images".format(view, len(self.data2D_paths[view])))
            elif split == 'val':
                print("validating on view: {} with {} images".format(view, len(self.data2D_paths[view])))
        # for view in range(self.VIEW_NUM):
        #     # Get a list of all jpeg files in the data dir
        #     # self.labels = next(unpickle_data(data_path + 'v_{}_labels.pkl'.format(view)))
        #     data_dir = os.path.join(dataPathPrefix2D, "a_rem_c{}".format(view))
        #     result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
        #     i = 0
        #     for el in result:
        #         # i += 1
        #         # if i > 50:
        #         #     break
        #         img_id = os.path.basename(el).split('.jpg')[0]
        #         # if img_id in self.labels:
        #         _id = img_id.split('_')[0]
        #         if _id in model_set:
        #             self.data2D_paths[view].append(el)
        #             self.seg2D_paths[view].append(os.path.join(seg_path, _id + "_" + str(view) + ".jpg"))
        #             self.depth2D_paths[view].append(os.path.join(depth_path, _id + "_" + str(view) + ".jpg"))
        #             # if split == 'train':
        #             #     if _id in train_model_ids:
        #             #         self.data2D_paths[view].append(el)
        #             # elif split == 'val':
        #             #     if _id in val_model_ids:
        #             #         self.data2D_paths[view].append(el)
        #             # else:
        #             #     if _id in test_model_ids:
        #             #         self.data2D_paths[view].append(el)
        #         # break
        #     if split == 'train':
        #         print("training on view: {} with {} images".format(view, len(self.data2D_paths[view])))
        #     elif split == 'val':
        #         print("validating on view: {} with {} images".format(view, len(self.data2D_paths[view])))
        # for x in self.data_paths:
        #     ps = glob(join(x[:-15].replace(split, '2D'), 'color', '*.jpg'))
        #     assert len(ps) >= self.VIEW_NUM, '[!] %s has only %d frames, less than expected %d samples' % (
        #         x, len(ps), self.VIEW_NUM)
        #     ps.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        #     if val_benchmark:
        #         ps = ps[::5]
        #     self.data2D_paths.append(ps)

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
                t_2d.Crop([self.IMG_DIM[1], self.IMG_DIM[0]], crop_type='rand', padding=mean,
                          ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            self.transform_2d = t_2d.Compose([
                t_2d.Crop([self.IMG_DIM[1], self.IMG_DIM[0]], crop_type='rand', padding=mean,
                          ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])

    def __getitem__(self, index_long):
        index = index_long % len(self.data2D_paths[0])
        f = self.data2D_paths[0][index]
        model_id = f.split('/')[-1].split('_')[0]
        model_index = self.part_materials[model_id]
        according, pmmap = self.get_2d(index)
        category = self.cls[model_index]
        return category, according, pmmap

    def get_2d(self, room_id):
        """
        :param      room_id:
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor
        """
        # frames_path = self.data2D_paths[room_id]
        # partial = int(len(frames_path) / self.VIEW_NUM)
        imgs, labels, links, materials = [], [], [], []
        # coords = coords[0, :, :]
        for v in range(self.VIEW_NUM):
            f = self.data2D_paths[v][room_id]
            model_id = f.split('/')[-1].split('_')[0]
            material_id = f.split("_")[-2]
            view_id = f.split("_")[-1][0]
            # part and materials cooresponding index
            pmvalue = self.material_index[model_id]
            pm = pmvalue[str(material_id)]
            pm_map = np.zeros(8).astype(int)
            for i, j in enumerate(pm.keys()):
                # try:
                pm_map[i + 1] = pm[j]
                # except:
                #     pm_map[]
            try:
                img = imageio.imread(f)
            except:
                print("============================")
                print(f)
                img = np.ones((400, 400, 3))

            label_ads = self.seg2D_paths[v][
                room_id]
            try:
                label = imageio.imread(label_ads, as_gray=True).astype(int)
            except:
                print(label_ads)
                label = np.ones((400, 400))

            # label = self.remapper[label]
            label = np.floor(np.array(label) / 36).astype(int)
            # self.part_map
            material = label.copy()
            material = pm_map[material]

            material = torch.from_numpy(material)
            according = np.zeros(8).astype(int)
            model_part = self.new_part_names[model_id]
            for i, k in enumerate(model_part.keys()):
                according[i + 1] = self.part_map[model_part[k]]
            label = according[label]

        return according, pm_map

    def __len__(self):
        return len(self.data2D_paths[0])


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
    coords, feats, labels, colors, labels_2d, links, cls, mat = list(zip(*batch))
    # pdb.set_trace()

    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
    # print(cls)
    # print(torch.stack(labels_2d).shape)
    # print(labels_2d.shape)
    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.stack(cls), torch.stack(mat)


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
    try:
        category, according, pmmap = list(zip(*batch))
    except:
        category, according, pmmap = list(zip(*batch))
        cls = 0
        print(len(list(zip(*batch))))
    # inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0


    return np.stack(category), np.stack(according), np.stack(pmmap)


if __name__ == '__main__':
    import time
    from tensorboardX import SummaryWriter

    # if we use style model how would we solve it.
    data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    # data_root = '/data/dataset/processed_models'
    data_root2d = '/data/compat/img'
    data_root = '/data/dataset/glbfile'
    train_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=True, split='train',
                              memCacheInit=True, voxelSize=0.05)
    val_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=False, split='val',
                            memCacheInit=True, voxelSize=0.05,
                            eval_all=True)
    # test_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=False, split='val',
    #                          memCacheInit=True, voxelSize=0.05,
    #                          eval_all=True)
    # coords, feats, labels, colors, labels_2d, links = train_data.__getitem__(0)
    # print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape)
    #
    # coords, feats, labels, colors, labels_2d, links, inds_recons = val_data.__getitem__(0)
    # print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_recons.shape)
    # coords, feats, labels, colors, labels_2d, links, inds_recons = test_data.__getitem__(0)
    # print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_recons.shape)
    # exit(0)

    manual_seed = 123


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=32, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=32, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    # _ = iter(train_loader).__next__()
    trainLog = SummaryWriter('Exp/scannet/statistic_cross/train')
    valLog = SummaryWriter('Exp/scannet/statistic_cross/val')
    cls, ac, pm = [], [], []
    cls1, ac1, pm1 = [], [], []
    for idx in range(1):
        end = time.time()
        for step, (category, according, pmmap) in enumerate(train_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
            cls.append(category)
            ac.append(according)
            pm.append(pmmap)

        for step, (category, according, pmmap) in enumerate(
                val_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(val_loader), time.time() - end))
            cls1.append(category)
            ac1.append(according)
            pm1.append(pmmap)

            # time.sleep(0.3)
            end = time.time()
    np.save(join( 'cls.npy'), np.concatenate(cls) )
    np.save(join( 'ac.npy'), np.concatenate(ac) )
    np.save(join( 'pm.npy'), np.concatenate(pm) )
    np.save(join( 'cls1.npy'), np.concatenate(cls1) )
    np.save(join( 'ac1.npy'), np.concatenate(ac1) )
    np.save(join( 'pm1.npy'), np.concatenate(pm1) )
    trainLog.close()
    valLog.close()
