import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2


from PIL import Image, ImageDraw, ImageFont

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from utils import get_bounding_box_mat, unpickle_data, get_bounding_box_part, get_bounding_box_mat

########################################################
parts = ['arm', 'armrest', 'back', 'back_horizontal_bar', 'back_panel', 'back_vertical_bar', 'backrest', 'bag_body', 'base', 'base1', 'body', 'bottom_panel', 'bulb', 'bush', 'cabinet', 'caster', 'channel', 'container', 'containing_things', 'cushion', 'design', 'door', 'doorr', 'drawer', 'drawerr', 'foot_base', 'footrest', 'glass', 'handle', 'harp', 'head', 'headboard', 'headrest', 'keyboard_tray', 'knob', 'lamp_surrounding_frame', 'leg', 'legs', 'leveller', 'lever', 'lid', 'mattress', 'mechanism', 'neck', 'pillow', 'plug', 'pocket', 'pole', 'rod', 'seat', 'seat_cushion', 'shade_cloth', 'shelf', 'socket', 'stand', 'stretcher', 'support', 'supports', 'tabletop_frame', 'throw_pillow', 'top', 'top_panel', 'vertical_divider_panel', 'vertical_side_panel', 'vertical_side_panel2', 'wall_mount', 'wheel', 'wire','none']
material = ['Nvidia.vMaterials.AEC.Metal.Metal_Brushed_Herringbone.metal_brushed_herringbone_bricks_steel', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_gold_and_charcoal', 'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_black', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_red', 'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_carbonized_matte', 'Green Glass', 'Nvidia.vMaterials.AEC.Wood.Birch.birch_bleached_matte', 'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_redgold', 'Nvidia.vMaterials.AEC.Wood.Oak.oak_rustic_matte', 'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_sand', 'Nvidia.vMaterials.Design.Ceramic.Porcelain_Floral.porcelain_floral_bluesilver', 'Nvidia.vMaterials.AEC.Stone.Marble.Imperial_Marble.stone_imperial_marble_copper_tones', 'Nvidia.vMaterials.Design.Leather.Leather_Snake.leather_snake_black', 'Nvidia.vMaterials.Design.Wood.Teak.teak_carbonized_matte', 'Nvidia.vMaterials.Design.Leather.Leather_Pebbled.pebbled_1', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_violet', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Yellow', 'Nvidia.vMaterials.AEC.Wood.white_oak_floorboards.white_oak_floorboards', 'Crackle_Bromine_SW-017-F844', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Light_Blue', 'Blue iron', 'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_blueblack', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_copper_and_sunset', 'Rough Bronze', 'Brushed copper', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_gold_and_emerald', 'Nvidia.vMaterials.AEC.Wood.birch_wood_oiled.birch_wood_oiled', 'Nvidia.vMaterials.Design.Wood.burlwood.burlwood', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_silver_and_dapple_gray', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_sky_blue', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Dark_Red', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_midblue', 'Leaf Green', 'Nvidia.vMaterials.Design.Metal.diamond_plate.diamond_plate', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_ebony', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_brown', 'Nvidia.vMaterials.AEC.Metal.Metal_Patina_Hammered_Bricks.metal_patina_hammered_bricks_pearl', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_purple', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_green', 'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_redblack', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_sapphirenvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_sapphire', 'Nvidia.vMaterials.AEC.Metal.Metal_Aged_Quilted_Backsplash.metal_aged_quilted_backsplash_antique_bronze', 'Glass', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_dark_blue', 'Grey_with_Gold_Crackled', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_black', 'Nvidia.vMaterials.AEC.Wood.Cherry.cherry_chocolate_coated', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_black', 'Nvidia.vMaterials.AEC.Wood.Oak.oak_mahogany_coated', 'Nvidia.vMaterials.AEC.Metal.Metal_Brushed_Disc_Mosaic.metal_brushed_disc_mosaic_silver', 'Nvidia.vMaterials.Design.Wood.Teak.teak_ebony_coated', 'Nvidia.vMaterials.Design.Ceramic.Porcelain_Cracked.cracked_porcelain_beige', 'Nvidia.vMaterials.AEC.Metal.Metal_Pitted_Steel.pitted_steel', 'Mirror', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_lime_green', 'Nvidia.vMaterials.AEC.Wood.Oak.oak_ebony_coated', 'Nvidia.vMaterials.Design.Leather.Leather_Snake.leather_snake_eggshell', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Cheery_Red', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_emerald', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_dark_gray', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_black', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Petrol', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_orange', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_red', 'Nvidia.vMaterials.AEC.Wood.Maple_Isometric_Parquet.maple_isometric_parquet_timber_wolf', 'Nvidia.vMaterials.AEC.Metal.Metal_Hammered_Bricks.metal_hammered_bricks_pearl', 'Antique Gold', 'Nvidia.vMaterials.AEC.Stone.Marble.River_Marble.river_marble_rust', 'Nvidia.vMaterials.Design.Fabric.Denim.denim_lightblue', 'Dented gold', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_green', 'Nvidia.vMaterials.Design.Metal.bare_metal.bare_metal', 'Nvidia.vMaterials.AEC.Wood.Beech.beech_ebony_oiled', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Orange', 'Nvidia.vMaterials.Design.Metal.Cast_Metal.cast_metal_copper_vein', 'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_brown', 'Rough silver', 'Nvidia.vMaterials.AEC.Metal.Metal_Prisma_Tiles.metal_prisma_tiles_yellow_gold', 'Nvidia.vMaterials.AEC.Wood.Beech.beech_carbonized_oiled', 'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_natural_oiled', 'Nvidia.vMaterials.AEC.Wood.Laminate_Oak.oak_hazelnut', 'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_tan', 'Nvidia.vMaterials.AEC.Wood.Maple_Fan_Parquet.maple_fan_parquet_shades_of_gray_matte', 'Nvidia.vMaterials.AEC.Wood.Beech.beech_bleached_matte', 'Glass_Reflective', 'Nvidia.vMaterials.Design.Leather.Upholstery.leather_upholstery_darkgray', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_blue', 'Nvidia.vMaterials.AEC.Wood.Cherry.cherry_ebony_matte', 'Nvidia.vMaterials.Design.Metal.Cast_Metal.cast_metal_antique_nickel', 'Nvidia.vMaterials.AEC.Wood.Bamboo.bamboo_ash_coated', 'Nvidia.vMaterials.AEC.Wood.Laminate_Oak_2.Laminate_Oak_Slight_Smudges', 'Nvidia.vMaterials.Design.Leather.Suede.suede_black', 'Nvidia.vMaterials.AEC.Wood.Maple_Chevron_Parquet.maple_chevron_parquet_golden_honey_varnish', 'Nvidia.vMaterials.Design.Ceramic.Porcelain_Cracked.cracked_porcelain_gold_veined', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_beige', 'Nvidia.vMaterials.AEC.Stone.Marble.Classic_Marble.classic_marble_black_and_white', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Dark_Red', 'Nvidia.vMaterials.AEC.Wood.Maple_Inlaid_Parquet.maple_inlaid_parquet_rich_coffee', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_navyblue', 'Nvidia.vMaterials.AEC.Wood.Maple_Herringbone_Parquet.maple_herringbone_parquet_rich_coffee', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Cheery_Red', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_ruby', 'Nvidia.vMaterials.AEC.Wood.Wood_Bark.Wood_Bark', 'Nvidia.vMaterials.Design.Metal.gun_metal.gun_metal', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_yellow', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Dark_Green', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_orange', 'Black Glass', 'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_red', 'Nvidia.vMaterials.AEC.Metal.metal_beams.metal_beams', 'Nvidia.vMaterials.AEC.Metal.Metal_Patina_Hammered.metal_patina_hammered_darkened_silver', 'Nvidia.vMaterials.Design.Leather.Leather_Pebbled.pebbled_2', 'Nvidia.vMaterials.AEC.Wood.Wood_Cork.Wood_Cork', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_white', 'Nvidia.vMaterials.AEC.Wood.mahogany_floorboards.mahogany_floorboards', 'Nvidia.vMaterials.AEC.Metal.Metal_Modern_Stacked_Panels.metal_modern_stacked_panels_titanium', 'Nvidia.vMaterials.AEC.Wood.beech_wood_matte.beech_wood_matte', 'Nvidia.vMaterials.AEC.Wood.Maple_Tesselated_Parquet.maple_tessellated_parquet_ivory', 'Nvidia.vMaterials.AEC.Wood.Maple_Double_Herringbone_Parquet.maple_double_herringbone_parquet_mahogany_varnish', 'Nvidia.vMaterials.Design.Leather.Suede.suede_brown', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven', 'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_black', ' nvidia.vMaterials.AEC.Metal.Metal_Aged_Disc_Mosaic.metal_aged_disc_mosaic_antique_copper', 'Nvidia.vMaterials.Design.Fabric.Fabric_Cotton_Fine_Woven.Fabric_Cotton_Fine_Woven_Orange', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Dark_Ash', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_blue', 'Nvidia.vMaterials.Design.Fabric.Denim.denim_darkblue', 'Gold metal', 'Nvidia.vMaterials.Design.Fabric.Stripes.fabric_stripes_blue', 'Nvidia.vMaterials.AEC.Wood.bamboo_coated.bamboo_coated', 'Nvidia.vMaterials.Design.Fabric.Felt.felt_red', 'Nvidia.vMaterials.AEC.Metal.Metal_Hammered_Bricks.metal_hammered_bricks_pale_gold', 'Nvidia.vMaterials.AEC.Metal.Metal_brushed_antique_copper_patinated.brushed_antique_copper_minimal_patina', 'Nvidia.vMaterials.AEC.Wood.Maple_Inlaid_Parquet.maple_inlaid_parquet_natural_mix', 'Nvidia.vMaterials.Design.Plastic.Plastic_Thick_Translucent_Flakes.plastic_lemon', 'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_beige', 'Nvidia.vMaterials.AEC.Metal.Metal_Modern_Offset_Panels.metal_modern_offset_panels_carbon_steel', 'Nvidia.vMaterials.Design.Ceramic.Porcelain.porcelain_green', 'Nvidia.vMaterials.AEC.Wood.maple_wood_coated.maple_wood_coated', 'Acrylic', 'Nvidia.vMaterials.AEC.Wood.Maple.maple_ash_coated', 'Nvidia.vMaterials.Design.Metal.metal_cast_iron.metal_cast_iron', 'Nvidia.vMaterials.Design.Leather.Suede.suede_gray', 'Nvidia.vMaterials.AEC.Wood.Maple_Hexagon_Parquet.maple_hexagon_parquet_rough_ivory_varnish', 'Nvidia.vMaterials.AEC.Metal.Metal_brushed_antique_copper.brushed_antique_copper_shiny', 'Nvidia.vMaterials.Design.Leather.Textured.leather_textured_darkbrown', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_clear', 'Nvidia.vMaterials.AEC.Wood.Maple_Isometric_Parquet.maple_isometric_parquet_rustic_mix', 'Shiny Nickel', 'Nvidia.vMaterials.AEC.Stone.Marble.Veined_Marble.veined_marble_silver_and_storm', 'Nvidia.vMaterials.Design.Fabric.Flannel.flannel_blackgray', 'Nvidia.vMaterials.AEC.Stone.Marble.Large_Quartz_And_Marble.large_quartz_marble_coated_amber', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven', 'Nvidia.vMaterials.Design.Fabric.Cotton_Roughly_Woven.Fabric_Cotton_Roughly_Woven_Green', 'Nvidia.vMaterials.AEC.Wood.cherry_wood_oiled.cherry_wood_oiled']
material = [i.lower() for i in material]
objs =  ['bench', 'planter', 'dresser', 'stool', 'chair', 'cabinet', 'shelf', 'love seat', 'table', 'ottoman', 'vase', 'bed', 'trolley', 'sofa', 'lamp','none']
data_dir = "/var/remote/lustre/project/k1546/ujjwal/data/elevation_30_v5/"

num_cl_part = len(parts)
num_qu_part = 30
num_cl_obj = len(objs)
num_qu_obj = 10
num_cl_mat = len(material)
num_qu_mat = 30

seed = 42
null_class_coef = 0.5
LR = 2e-5
EPOCHS = 50
BATCH_SIZE = 32
#######################################################
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

id_name = json.load(open("/home/varshnt/tezuesh/test/tezuesh/id_model.json")) 

def norm_trans():
    return T.Compose([
        T.Resize((1920//4,1080//4)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),            
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      A.ToGray(p=0.01),                      
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.Resize(height=224, width=224    , p=1),
                      A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,

                    #   bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                    #   bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )


def box_xywh_to_cxcywh(lis):
    return [lis[0] + lis[2]/2, lis[1] + lis[3]/2, lis[2], lis[3]]

def boxx_miny_minx_maxy_max_to_cxcywh(lis):
    return [(lis[0]+lis[2])/2,(lis[1]+lis[3])/2,lis[2]-lis[0],lis[3]-lis[1]]

class PartDataset(Dataset):
    def __init__(self, data_dir, t='train', transforms=None):
        self.data_dir = data_dir

        # Read the splits
        splits = unpickle_data('/home/varshnt/tezuesh/test/tezuesh/test_train_split_50.pkl')

        train_model_ids = splits['train']['model_ids']
        test_model_ids = splits['test']['model_ids']
        val_model_ids = splits['val']['model_ids']


        
        # Get a list of all jpeg files in the data dir
        result = os.listdir(data_dir)
        
        # use only models in the split
        self.models = []
        
        self.path_bb = "/home/varshnt/tezuesh/test/tezuesh/bb_final_50/"
        self.part_bb = []

        for el in result:
            _id = el.split('_')[0]

            if t == 'train':
                if _id in train_model_ids:
                    if os.path.exists(self.path_bb + el):
                        self.part_bb.append(self.path_bb + el)
                        self.models.append(el)
            elif t == 'val':
                if _id in val_model_ids:
                    if os.path.exists(self.path_bb + el):
                        self.part_bb.append(self.path_bb + el)
                        self.models.append(el)
            else:
                if _id in test_model_ids:
                    if os.path.exists(self.path_bb + el):
                        self.part_bb.append(self.path_bb + el)
                        self.models.append(el)
        self.models = self.models[len(self.models)%BATCH_SIZE:]
        if t == 'train':
            print("training with {} images".format( len(self.models)))
        elif t == 'val':
            print("validating with {} images".format(len(self.models)))

        # Get the transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item):
        # print("--------------------------------------------------------------------------------------------------------------")
        record = self.models[item]
        p = data_dir + record + "/Image0080.png"
        ori_image = Image.open(p)
        ori_image = ori_image.convert('RGB')
        image = ori_image.copy()
        h,w = 1920, 1080


        # partname_bb = get_bounding_box_part(self.part_bb[item])
        partname_bb = get_bounding_box_part(self.path_bb + record)
        material_bb = get_bounding_box_mat(self.path_bb + record)
        # material_bb = get_bounding_box_mat(self.part_bb[item])
        # print(item, self.part_bb[item], self.path_bb + record)


        # Part Param -------------------------------------------------------------------------------------
        part_boxes = []
        label_part = []
        for key, value in partname_bb.items():
            key = key.lower()
            part_boxes.append(boxx_miny_minx_maxy_max_to_cxcywh(value))
            # print(value)
            if key not in parts:
                label_part.append(parts.index('none'))
            else:
                label_part.append(parts.index(key))
        
        part_boxes = torch.tensor(part_boxes)
        # print("partboxes" ,part_boxes.shape)
        # print("partboxes" ,torch.unsqueeze(part_boxes,0).shape)

        label_part =  torch.tensor(label_part)

        area = part_boxes[:,2]*part_boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms:
            image = self.transforms(ori_image.copy())
        
        _,h,w = image.shape
        # print("h and w ", h, w)  
        boxes = part_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        target_part = {}
        target_part['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_part['labels'] = torch.as_tensor(label_part,dtype=torch.long)
        target_part['image_id'] = torch.tensor([item])
        target_part['area'] = area


        # Object Param -------------------------------------------------------------------------------------

        partname_bb= get_bounding_box_part(self.part_bb[item])
        part_boxes = []
        for key, value in partname_bb.items():
            part_boxes.append(value)
        # part_boxes = list(partname_bb.values())
        part_boxes = torch.tensor(part_boxes)
        obj_x_min = part_boxes[:,0].min()
        obj_y_min = part_boxes[:,1].min()
        obj_x_max = part_boxes[:,2].max()
        obj_y_max = part_boxes[:,3].max()

        obj_bb = torch.tensor([boxx_miny_minx_maxy_max_to_cxcywh([obj_x_min, obj_y_min, obj_x_max, obj_y_max])])
        obj_id = record.split("_")[0]
        if id_name[obj_id] in objs:
            label_obj = torch.tensor([objs.index(id_name[obj_id])])
        else:
            label_obj = torch.tensor([objs.index('none')])


        if self.transforms:
            image = self.transforms(ori_image.copy())   
            
        _,h,w = image.shape
        boxes = obj_bb / torch.tensor([w, h, w, h], dtype=torch.float32)
        target_obj = {}
        target_obj['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_obj['labels'] = torch.as_tensor(label_obj,dtype=torch.long)
        target_obj['image_id'] = torch.tensor([item])
        target_obj['area'] = area


        # Material Param -------------------------------------------------------

        mat_boxes = []
        label_mat = []
        for key, value in material_bb.items():
            key = key.lower()
            mat_boxes.append(boxx_miny_minx_maxy_max_to_cxcywh(value))
            if key not in material:
                label_mat.append(material.index('none'))
            else:
                label_mat.append(material.index(key))
            
        
        mat_boxes = torch.tensor(mat_boxes)        
        label_mat =  torch.tensor(label_mat)

        area = mat_boxes[:,2]*mat_boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms:
            image = self.transforms(ori_image.copy())
        
        _,h,w = image.shape
        boxes = mat_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        # boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target_mat = {}
        target_mat['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_mat['labels'] = torch.as_tensor(label_mat,dtype=torch.long)
        target_mat['image_id'] = torch.tensor([item])
        target_mat['area'] = area

        return image, target_part,target_obj, target_mat, record


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETRModel(nn.Module):
    def __init__(self,num_cl_parts, num_que_parts, num_cl_obj, num_que_obj, num_cl_ma, num_que_ma):
        super(DETRModel,self).__init__()
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model_part = nn.Linear(in_features=self.in_features,out_features=num_cl_parts)
        self.bbox_part_ma = MLP(self.in_features, self.in_features,4,3)
        self.num_que_part = num_que_parts

        
    def forward(self,images):

        activation = {}
        def getActivation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        
        h = self.model.transformer.decoder.register_forward_hook(getActivation('norm'))
        features = self.model(images)
        features = activation['norm']
        return {'pred_logits_part': self.model_part(features)[-1], 
                'pred_boxes_pa_ma': self.bbox_part_ma(features).sigmoid()[-1]}
                


matcher_part = HungarianMatcher()
matcher_obj = HungarianMatcher()
matcher_mat = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    criterion_part, criterion_obj, criterion_mat = criterion
    criterion_part.train()
    criterion_obj.train()
    criterion_mat.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for step, (images, target_part,target_obj, target_mat, record) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        target_part = [{k: v.to(device) for k, v in t.items()} for t in target_part]
        target_obj = [{k: v.to(device) for k, v in t.items()} for t in target_obj]
        target_mat = [{k: v.to(device) for k, v in t.items()} for t in target_mat]
        

        output = model(images)
        part_log = output['pred_logits_part'].reshape(-1, 100, num_cl_part)
        bb_pa_mat = output['pred_boxes_pa_ma'].reshape(-1,100,4)

        part = {'pred_boxes': bb_pa_mat , 'pred_logits':part_log }
        # mat = {'pred_boxes': bb_pa_mat, 'pred_logits':mat_log }
        # obj = {'pred_boxes':bb_obj , 'pred_logits':obj_log }

       
        loss_di_part = criterion_part(part, target_part)
        # loss_di_mat = criterion_mat(mat, target_mat)
        # loss_di_obj = criterion_obj(obj, target_obj)
        
        weight_dict = criterion_part.weight_dict
        
        losses_part = sum(loss_di_part[k] * weight_dict[k] for k in loss_di_part.keys() if k in weight_dict)
        # losses_obj = sum(loss_di_obj[k] * weight_dict[k] for k in loss_di_obj.keys() if k in weight_dict)
        # losses_mat = sum(loss_di_mat[k] * weight_dict[k] for k in loss_di_mat.keys() if k in weight_dict)

        losses = losses_part 
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss


def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion_part, criterion_obj, criterion_mat = criterion
    criterion_part.eval()
    criterion_obj.eval()
    criterion_mat.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        
        for step, (images, target_part,target_obj, target_mat, record) in enumerate(tk0):
            
            images = list(image.to(device) for image in images)
            target_part = [{k: v.to(device) for k, v in t.items()} for t in target_part]
            target_obj = [{k: v.to(device) for k, v in t.items()} for t in target_obj]
            target_mat = [{k: v.to(device) for k, v in t.items()} for t in target_mat]
            

            output = model(images)
            part_log = output['pred_logits_part'].reshape(-1, 100, num_cl_part)
            bb_pa_mat = output['pred_boxes_pa_ma'].reshape(-1,100,4)

            part = {'pred_boxes': bb_pa_mat , 'pred_logits':part_log }
            # mat = {'pred_boxes': bb_pa_mat, 'pred_logits':mat_log }
            # obj = {'pred_boxes':bb_obj , 'pred_logits':obj_log }

        
            loss_di_part = criterion_part(part, target_part)
            # loss_di_mat = criterion_mat(mat, target_mat)
            # loss_di_obj = criterion_obj(obj, target_obj)
            
            weight_dict = criterion_part.weight_dict
            
            losses_part = sum(loss_di_part[k] * weight_dict[k] for k in loss_di_part.keys() if k in weight_dict)
            # losses_obj = sum(loss_di_obj[k] * weight_dict[k] for k in loss_di_obj.keys() if k in weight_dict)
            # losses_mat = sum(loss_di_mat[k] * weight_dict[k] for k in loss_di_mat.keys() if k in weight_dict)

            losses = losses_part
            
            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
    
    return summary_loss


def collate_fn(batch):
    return tuple(zip(*batch))

def run(fold):
    
    train_dataset = PartDataset(data_dir,'train',norm_trans())
    valid_dataset = PartDataset(data_dir,'val',norm_trans())
    
    train_data_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE//4,shuffle=False,num_workers=4,collate_fn=collate_fn)
    
    device = torch.device('cuda')
    model = DETRModel(
                    num_cl_parts=num_cl_part,
                    num_que_parts=num_qu_part, 
                    num_cl_obj=num_cl_obj,
                    num_que_obj=num_qu_obj, 
                    num_cl_ma=num_cl_mat,
                    num_que_ma=num_qu_mat )
    model = model.to(device)

    criterion_part = SetCriterion(num_cl_part-1, matcher_part, weight_dict, eos_coef = null_class_coef, losses=losses).to(device)
    criterion_obj = SetCriterion(num_cl_obj-1, matcher_obj, weight_dict, eos_coef = null_class_coef, losses=losses).to(device)
    criterion_mat = SetCriterion(num_cl_mat-1, matcher_mat, weight_dict, eos_coef = null_class_coef, losses=losses).to(device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_loss = 10**5
    for epoch in range(EPOCHS):
        print("Traing about to start")
        train_loss = train_fn(train_data_loader, model,(criterion_part, criterion_obj, criterion_mat), optimizer,device,scheduler=None,epoch=epoch)
        # valid_loss = eval_fn(valid_data_loader, model,(criterion_part, criterion_obj, criterion_mat), device)
        
        print('|EPOCH {}| TRAIN_LOSS {}|'.format(epoch+1,train_loss.avg))
        
        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
            torch.save(model.state_dict(), f'model1b_best_{fold}.pth')

run(fold=0)