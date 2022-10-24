#!/usr/bin/env python
import math
import random
import torch
import numpy as np
import difflib
from tqdm import tqdm, trange
from os.path import join
from glob import glob
import SharedArray as SA
import imageio
import os
import dataset.augmentation_2d as t_2d
from dataset.scanNet3DClsNew import ScanNet3D
from dataset.utils import unpickle_data, pickle_data
from collections import defaultdict, Counter
import json
import pandas as pd
import cv2
import ast

# parts = ['arm', 'armrest', 'back', 'back_horizontal_bar', 'back_panel', 'back_vertical_bar', 'backrest', 'bag_body',
#          'base', 'base1', 'body', 'bottom_panel', 'bulb', 'bush', 'cabinet', 'caster', 'channel', 'container',
#          'containing_things', 'cushion', 'design', 'door', 'doorr', 'drawer', 'drawerr', 'foot_base', 'footrest',
#          'glass', 'handle', 'harp', 'head', 'headboard', 'headrest', 'keyboard_tray', 'knob', 'lamp_surrounding_frame',
#          'leg', 'legs', 'leveller', 'lever', 'lid', 'mattress', 'mechanism', 'neck', 'pillow', 'plug', 'pocket', 'pole',
#          'rod', 'seat', 'seat_cushion', 'shade_cloth', 'shelf', 'socket', 'stand', 'stretcher', 'support', 'supports',
#          'tabletop_frame', 'throw_pillow', 'top', 'top_panel', 'vertical_divider_panel', 'vertical_side_panel',
#          'vertical_side_panel2', 'wall_mount', 'wheel', 'wire', 'none']
# material = ['Nvidia.vMaterials.AEC.Metal.Metal_Brushed_Herringbone.metal_brushed_herringbone_bricks_steel',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_gold_and_charcoal',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_black',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_red',
#             'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_carbonized_matte', 'Green Glass',
#             'Nvidia.vMaterials.AEC.Wood.Birch.birch_bleached_matte',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_redgold',
#             'Nvidia.vMaterials.AEC.Wood.Oak.oak_rustic_matte',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_sand',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_bluesilver',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Imperial_Marble.stone_imperial_marble_copper_tones',
#             'Nvidia.vMaterials.Design.Leather.Leather_Snake.leather_snake_black',
#             'Nvidia.vMaterials.Design.Wood.Teak.teak_carbonized_matte',
#             'Nvidia.vMaterials.Design.Leather.Leather_Pebbled.pebbled_1',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_violet',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Yellow',
#             'Nvidia.vMaterials.AEC.Wood.white_oak_floorboards.white_oak_floorboards', 'Crackle_Bromine_SW-017-F844',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Light_Blue', 'Blue iron',
#             'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_blueblack',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_copper_and_sunset', 'Rough Bronze',
#             'Brushed copper', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_gold_and_emerald',
#             'Nvidia.vMaterials.AEC.Wood.birch_wood_oiled.birch_wood_oiled',
#             'Nvidia.vMaterials.Design.Wood.burlwood.burlwood',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_silver_and_dapple_gray',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_sky_blue',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Dark_Red',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_midblue', 'Leaf Green',
#             'Nvidia.vMaterials.Design.Metal.diamond_plate.diamond_plate',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_ebony',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_brown',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Patina_Hammered_Bricks.metal_patina_hammered_bricks_pearl',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_purple',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_green',
#             'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_redblack',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_sapphirenvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_sapphire',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Aged_Quilted_Backsplash.metal_aged_quilted_backsplash_antique_bronze',
#             'Glass', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_dark_blue',
#             'Grey_with_Gold_Crackled',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_black',
#             'Nvidia.vMaterials.AEC.Wood.Cherry.cherry_chocolate_coated',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_black',
#             'Nvidia.vMaterials.AEC.Wood.Oak.oak_mahogany_coated',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Brushed_Disc_Mosaic.metal_brushed_disc_mosaic_silver',
#             'Nvidia.vMaterials.Design.Wood.Teak.teak_ebony_coated',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain_Cracked.cracked_porcelain_beige',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Pitted_Steel.pitted_steel', 'Mirror',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_lime_green',
#             'Nvidia.vMaterials.AEC.Wood.Oak.oak_ebony_coated',
#             'Nvidia.vMaterials.Design.Leather.Leather_Snake.leather_snake_eggshell',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Cheery_Red',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_emerald',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_dark_gray',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_black',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Petrol',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_orange',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_red',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Isometric_Parquet.maple_isometric_parquet_timber_wolf',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Hammered_Bricks.metal_hammered_bricks_pearl', 'Antique Gold',
#             'Nvidia.vMaterials.AEC.Stone.Marble.River_Marble.river_marble_rust',
#             'Nvidia.vMaterials.Design.Fabric.Denim.denim_lightblue', 'Dented gold',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_green',
#             'Nvidia.vMaterials.Design.Metal.bare_metal.bare_metal',
#             'Nvidia.vMaterials.AEC.Wood.Beech.beech_ebony_oiled',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Orange',
#             'Nvidia.vMaterials.Design.Metal.Cast_Metal.cast_metal_copper_vein',
#             'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_brown', 'Rough silver',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Prisma_Tiles.metal_prisma_tiles_yellow_gold',
#             'Nvidia.vMaterials.AEC.Wood.Beech.beech_carbonized_oiled',
#             'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_natural_oiled',
#             'Nvidia.vMaterials.AEC.Wood.Laminate_Oak.oak_hazelnut',
#             'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_tan',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Fan_Parquet.maple_fan_parquet_shades_of_gray_matte',
#             'Nvidia.vMaterials.AEC.Wood.Beech.beech_bleached_matte', 'Glass_Reflective',
#             'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_darkgray',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_blue',
#             'Nvidia.vMaterials.AEC.Wood.Cherry.cherry_ebony_matte',
#             'Nvidia.vMaterials.Design.Metal.Cast_Metal.cast_metal_antique_nickel',
#             'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_ash_coated',
#             'Nvidia.vMaterials.AEC.Wood.Laminate_Oak_2.Laminate_Oak_Slight_Smudges',
#             'Nvidia.vMaterials.Design.Leather.Suede.suede_black',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Chevron_Parquet.maple_chevron_parquet_golden_honey_varnish',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain_Cracked.cracked_porcelain_gold_veined',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_beige',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Classic_Marble.classic_marble_black_and_white',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Dark_Red',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Inlaid_Parquet.maple_inlaid_parquet_rich_coffee',
#             'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_navyblue',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Herringbone_Parquet.maple_herringbone_parquet_rich_coffee',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Cheery_Red',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_ruby',
#             'Nvidia.vMaterials.AEC.Wood.Wood_Bark.Wood_Bark', 'Nvidia.vMaterials.Design.Metal.gun_metal.gun_metal',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_yellow',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Dark_Green',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_orange', 'Black Glass',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_red',
#             'Nvidia.vMaterials.AEC.Metal.metal_beams.metal_beams',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Patina_Hammered.metal_patina_hammered_darkened_silver',
#             'Nvidia.vMaterials.Design.Leather.Leather_Pebbled.pebbled_2',
#             'Nvidia.vMaterials.AEC.Wood.Wood_Cork.Wood_Cork', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_white',
#             'Nvidia.vMaterials.AEC.Wood.mahogany_floorboards.mahogany_floorboards',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Modern_Stacked_Panels.metal_modern_stacked_panels_titanium',
#             'Nvidia.vMaterials.AEC.Wood.beech_wood_matte.beech_wood_matte',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Tesselated_Parquet.maple_tessellated_parquet_ivory',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Double_Herringbone_Parquet.maple_double_herringbone_parquet_mahogany_varnish',
#             'Nvidia.vMaterials.Design.Leather.Suede.suede_brown',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven',
#             'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_black',
#             ' nvidia.vMaterials.AEC.Metal.Metal_Aged_Disc_Mosaic.metal_aged_disc_mosaic_antique_copper',
#             'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Orange',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Dark_Ash',
#             'Nvidia.vMaterials.Design.Fabric.Felt.felt_blue', 'Nvidia.vMaterials.Design.Fabric.Denim.denim_darkblue',
#             'Gold metal', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_blue',
#             'Nvidia.vMaterials.AEC.Wood.bamboo_coated.bamboo_coated', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_red',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Hammered_Bricks.metal_hammered_bricks_pale_gold',
#             'Nvidia.vMaterials.AEC.Metal.Metal_brushed_antique_copper_patinated.brushed_antique_copper_minimal_patina',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Inlaid_Parquet.maple_inlaid_parquet_natural_mix',
#             'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_lemon',
#             'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_beige',
#             'Nvidia.vMaterials.AEC.Metal.Metal_Modern_Offset_Panels.metal_modern_offset_panels_carbon_steel',
#             'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_green',
#             'Nvidia.vMaterials.AEC.Wood.maple_wood_coated.maple_wood_coated', 'Acrylic',
#             'Nvidia.vMaterials.AEC.Wood.Maple.maple_ash_coated',
#             'Nvidia.vMaterials.Design.Metal.metal_cast_iron.metal_cast_iron',
#             'Nvidia.vMaterials.Design.Leather.Suede.suede_gray',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Hexagon_Parquet.maple_hexagon_parquet_rough_ivory_varnish',
#             'Nvidia.vMaterials.AEC.Metal.Metal_brushed_antique_copper.brushed_antique_copper_shiny',
#             'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_darkbrown',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_clear',
#             'Nvidia.vMaterials.AEC.Wood.Maple_Isometric_Parquet.maple_isometric_parquet_rustic_mix', 'Shiny Nickel',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_silver_and_storm',
#             'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_blackgray',
#             'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_amber',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven',
#             'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Green',
#             'Nvidia.vMaterials.AEC.Wood.cherry_wood_oiled.cherry_wood_oiled', 'none']
# material = [i.lower() for i in material]
# material1 = dict(zip(material, range(len(material))))
# objs = ['bench', 'planter', 'dresser', 'stool', 'chair', 'cabinet', 'shelf', 'love seat', 'table', 'ottoman', 'vase',
#         'bed', 'trolley', 'sofa', 'lamp', 'none']

mat_id = {"none": 0,
          "metal": 1,
          "granite": 2,
          "fabric": 3,
          "plastic": 4,
          "rubber": 5,
          "glass": 6,
          "marble": 7,
          "leather": 8,
          "wax": 9,
          "paper": 10,
          "wood": 11,
          "ceramic": 12,
          "paint": 13,
          "velvet": 14}


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
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[400, 400], image_dim=image_dim)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, camera_to_world, coords, depth):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=int)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coordsNew.shape[0] == 4, "[!] Shape error"
        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coordsNew)
        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = -((p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2])
        pi = np.round(p).astype(int)
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

    def __init__(self, dataPathPrefix='data_root/pc',
                 dataPathPrefix2D='data_root/image', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=False,
                 identifier=10233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False,
                 val_benchmark=False, com=10, args=None
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
        self.json_paths = []
        self.mat_map = []
        root_2d=dataPathPrefix2D
        # Todo what is this: The number of models
        self.model_maps = {}
        # 2d image address
        # root_2d = '/ibex/scratch/projects/c2090/3dcompat/canonical_new'
        lines = []
        stylev0 = (np.genfromtxt('data/style{}v0.txt'.format(com), dtype='str'))
        stylev1 = (np.genfromtxt('data/style{}v1.txt'.format(com), dtype='str'))
        import pickle
        with open('data/resnet50_preds.pickle', 'rb') as handle:
            resnet_predictions = pickle.load(handle)
        style=list(resnet_predictions.keys())
        df = pd.read_csv("data/meta-data/materials_df.csv")
        df = df.loc[:, ["name", "type"]]
        df.append({'name': 'none', 'type': 'none'}, ignore_index=True)
        self.mat_ids = df.applymap(lambda s: s.lower() if type(s) == str else s)
        # shape categories finished self
        self.cls = self.get_model_ids()
        df = pd.read_csv('data/part_mat_tuples.csv')
        part_mat = dict(zip(df['model_id'], df['tuples']))
        for ii in part_mat:
            tem = {}
            for i, j in ast.literal_eval(part_mat[ii]):
                tem[i] = j
            part_mat[ii] = tem
        self.part_mat = part_mat
        df = pd.read_csv('data/part_index.csv')
        self.part_index = dict(zip(df['orgin'].tolist(), df['new'].tolist()))

        for el in tqdm(list(stylev0) + list(stylev1)):
            _id = el.split('_')[0]
            if _id not in self.data_paths_index:
                if not os.path.exists(os.path.join(root_2d, _id)):
                    print(os.path.join(root_2d, _id))
                continue
            if not os.path.exists(os.path.join(root_2d, _id)):
                print(os.path.join(root_2d, _id))
                continue
            if not os.path.exists(os.path.join(root_2d, _id, 'segmentation.json')) :
                continue

            self.json_paths.append(os.path.join(root_2d, _id, 'segmentation.json'))
            for view in range(4):
                tmp = el + '_' + str(view)
                self.data2D_paths[view].append(os.path.join(root_2d, tmp, "Image0080.png"))
                self.seg2D_paths[view].append(
                    os.path.join(root_2d, _id, "Segmentation0080" + "_" + str(view) + ".png"))
                self.depth2D_paths[view].append(
                    os.path.join(root_2d, _id, 'Depth0080' + "_" + str(view) + ".png"))
        print(len(self.depth2D_paths[0]))
        for view in range(4):
            print("{} on view: {} with {} images".format(split, view, len(self.data2D_paths[view])))
        np.savetxt("com1.txt", self.data2D_paths[view], fmt="%s")
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

    def get_model_ids(self):
        # test_dir = '/ibex/scratch/liy0r/cvpr/BPNet/'
        df = pd.read_csv("data/model.csv")
        part_index = dict(zip(df['id'].tolist(), df['model'].tolist()))
        cat = sorted(list(set(df['model'].tolist())))
        classes = dict(zip(cat, range(len(cat))))
        for key in part_index:
            part_index[key] = classes[part_index[key]]
        return part_index  # , classes

    def __getitem__(self, index_long):
        index = index_long % len(self.data2D_paths[0])
        f = self.data2D_paths[0][index]
        model_id = f.split('/')[-2].split('_')[0]
        model_index = self.data_paths_index[model_id]
        locs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (self.split, self.identifier, model_index))
        feats_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (self.split, self.identifier, model_index))
        labels_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (self.split, self.identifier, model_index))
        colors, labels_2d, links, materials, part13 = self.get_2d(index, locs_in, f)
        category = torch.tensor(self.cls[model_id])
        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct, links = self.voxelizer.voxelize(locs, feats_in, labels_in, link=links)
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        mat_3d = labels.copy()
        try:
            # style = np.vectorize(part_bb.get)(style).astype(np.int_)
            mat_3d = np.vectorize(part13.get)(mat_3d).astype(np.int_)
            # if there are some wrong labels put it into style none 0
            mat_3d = np.where((mat_3d is not None) | (mat_3d < 15), mat_3d, 0).astype(np.int_)
        except:
            # print("===========mat change fails=================")
            # print(label_ads)
            u, inv = np.unique(mat_3d, return_inverse=True)
            part13 = defaultdict(int, part13)
            mat_3d = np.array([part13[x] for x in u])[inv].reshape(mat_3d.shape).astype(np.int_)
            mat_3d = np.where((mat_3d is not None) | (mat_3d < 15), mat_3d, 0).astype(np.int_)
        # print(mat_3d.shape)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()
        mat_3d = torch.from_numpy(mat_3d).long()
        if self.eval_all:
            return coords, feats, labels, colors, labels_2d, links, torch.from_numpy(
                inds_reconstruct).long(), category, materials, mat_3d, part13, f.split('/')[-2]
        return coords, feats, labels, colors, labels_2d, links, category, materials, mat_3d, part13, f.split('/')[-2]  # compositions id

    def get_2d(self, room_id, coords: np.ndarray, name: str):
        """
        :param      room_id:
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor

d        """
        # frames_path = self.data2D_paths[room_id]
        # partial = int(len(frames_path) / self.VIEW_NUM)
        imgs, labels, links, materials = [], [], [], []
        # print(self.json_paths[room_id])
        seg = json.load(open(self.json_paths[room_id]))
        seg1 = {}

        # get part label names
        seg1[0] = self.part_classes['none']
        # Json files of part names
        for i in seg:
            g_name = i.lower()
            if g_name in self.part_classes:
                part_name = g_name
            elif g_name in self.part_index:
                part_name = self.part_index[g_name]
            else:
                # if there are not parts in the selected parts, find the similar one
                try:
                    part_name = difflib.get_close_matches(g_name, list(self.part_classes.keys()))[0]
                except:
                    try:
                        part_name = difflib.get_close_matches(g_name.split('_')[0], self.part_classes.keys())[0]
                    except:
                        if g_name.startswith('archmodels59_footstool1'):
                            part_name = g_name[24:]
                        if part_name not in self.part_index:
                            part_name = "none"
                            print("doesn't find part names files of {}".format(g_name))
                # part_name = 'none'
            seg1[seg[i]] = self.part_classes[part_name]
        # df = pd.read_csv(self.mat_map[room_id])
        if 260 in seg1:
            seg1[255] = seg1[260]
        df = self.part_mat[name.split('/')[-2][:-2]]
        # Get the material names
        part13 = {}
        part13[self.part_classes['none']] = mat_id['none']
        for i in df:
            g_name = i.lower()
            m_name = df[i].lower()
            if g_name in self.part_classes:
                part_name = g_name
            elif g_name in self.part_index:
                part_name = self.part_index[g_name]
            else:
                # if there are not parts in the selected parts, find the similar one
                try:
                    part_name = difflib.get_close_matches(g_name, list(self.part_classes.keys()))[0]
                except:
                    try:
                        part_name = difflib.get_close_matches(g_name.split('_')[0], list[self.part_classes.keys()])[0]
                    except:
                        part_name = "none"
                        print("doesn't find part names files of {}".format(g_name))
            if m_name == 'velvet':
                print(name)
            part13[self.part_classes[part_name]] = mat_id[m_name]
        for v in range(self.VIEW_NUM):
            f = self.data2D_paths[v][room_id]
            self.remapper = np.zeros(194) * 12
            try:
                img = imageio.imread(f)
                img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_NEAREST).astype(int)
            except:
                print("===========rendered image fails=================")
                print(f)
                img = np.ones((400, 400, 3))
            label_ads = self.seg2D_paths[v][
                room_id]
            try:
                label = imageio.imread(label_ads, as_gray=True).astype(int)
                label = cv2.resize(label, dsize=(400, 400), interpolation=cv2.INTER_NEAREST).astype(np.int_)
            except:
                label = np.ones((400, 400))
            label = np.array(label)
            try:
                # label = np.vectorize(seg1.get)(label)
                u, inv = np.unique(label, return_inverse=True)
                seg1 = defaultdict(int, seg1)
                label = np.array([seg1[x] for x in u])[inv].reshape(label.shape).astype(np.int_)
            except:
                pass

            try:
                style = label.copy()
                u, inv = np.unique(style, return_inverse=True)
                part13 = defaultdict(int, part13)
                style = np.array([part13[x] for x in u])[inv].reshape(style.shape).astype(np.int_)

            except:
                # pass

                style = np.zeros((400, 400)).astype(np.int_)
            style = torch.from_numpy(style)
            depth_ads = self.depth2D_paths[v][
                room_id]
            try:
                depth = imageio.imread(depth_ads, as_gray=True) / 1000.0  # convert to meter
                label = cv2.resize(label, dsize=(400, 400), interpolation=cv2.INTER_NEAREST).astype(int)
            except:
                depth = np.ones((400, 400))
            posePath = '{}.txt'.format(v)
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                 (x.split(" ") for x in open(posePath).read().splitlines())])
            link = np.ones([coords.shape[0], 4], dtype=np.int64)
            link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)
            img, label = self.transform_2d(img, label)

            imgs.append(img)
            labels.append(label)
            links.append(link)
            materials.append(style)
        labels = torch.stack(labels, dim=-1)
        materials = torch.stack(materials, dim=-1)
        imgs = torch.stack(imgs, dim=-1)

        links = np.stack(links, axis=-1)
        links = torch.from_numpy(links)
        return imgs, labels, links, materials, part13

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
    coords, feats, labels, colors, labels_2d, links, cls, mat, mat_3d, part13, model_id = list(zip(*batch))
    # pdb.set_trace()

    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.stack(cls), torch.stack(
        mat), torch.cat(mat_3d), part13, model_id


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
        coords, feats, labels, colors, labels_2d, links, inds_recons, cls, mat, mat_3d, part13, model_id = list(
            zip(*batch))
    except:
        coords, feats, labels, colors, labels_2d, links, inds_recons = list(zip(*batch))
        cls = 0
        print(len(list(zip(*batch))))
    inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.cat(inds_recons), torch.stack(
        cls), torch.stack(mat), torch.cat(mat_3d), part13, model_id


if __name__ == '__main__':
    import time
    from tensorboardX import SummaryWriter

    # if we use style model how would we solve it.
    # data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    # data_root = '/data/dataset/processed_models'
    # data_root2d = '/data/compat/2d_image'
    data_root2d = '/var/remote/lustre/project/k1546/ujjwal/data/canonical_v0'
    data_root = '/ibex/scratch/liy0r/processed_models_v5'
    # train_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=False, split='train',
    #                           memCacheInit=True, voxelSize=0.05)
    val_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=False, split='test',
                            memCacheInit=True, voxelSize=0.05,
                            eval_all=True)
    # test_data = ScanNetCross(dataPathPrefix=data_root, dataPathPrefix2D=data_root2d, aug=False, split='val',
    #                          memCacheInit=True, voxelSize=0.05,
    #                          eval_all=True)
    # coords, feats, labels, colors, labels_2d, links, cls, mat, mat_3d = train_data.__getitem__(0)
    # print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape)

    coords, feats, labels, colors, labels_2d, links, inds_recons, cls, mat, mat_3d = val_data.__getitem__(0)
    print(torch.unique(mat))
    print(torch.unique(mat_3d))
    print(len(val_data))
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_recons.shape)
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

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=32, pin_memory=True,
    #                                            worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=32, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    # _ = iter(train_loader).__next__()
    trainLog = SummaryWriter('Exp/scannet/statistic_cross/train')
    valLog = SummaryWriter('Exp/scannet/statistic_cross/val')

    for idx in range(1):
        end = time.time()
        # for step, (coords, feats, labels, colors, labels_2d, links, categories, mat, mat_3d) in enumerate(train_loader):
        #     print(
        #         'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
        #     trainLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
        #     trainLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
        #     trainLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
        #     trainLog.add_histogram('color', feats, global_step=step)
        #     trainLog.add_histogram('2D_image', colors, global_step=step)
        #     # time.sleep(0.3)
        #
        #     end = time.time()

        for step, (coords, feats, labels, colors, labels_2d, links, inds_reverse, categories, mat, mat_3d) in enumerate(
                val_loader):
            print("cor{},labels{}".format(coords.shape, labels.shape))
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
