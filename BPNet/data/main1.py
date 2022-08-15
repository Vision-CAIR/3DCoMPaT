import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from utils import get_bounding_box_mat, unpickle_data, get_bounding_box_part, get_bounding_box_mat


########################################################
parts = ["rider head", "spice box", "eyes", "stand outer surface", "Base_gold-Mesh", "wine rack", "latch", "round table", "candle rop", "design rim", "Legs_1", "MDF-Mesh", "candle holder", "Cylinder005-Mesh", "METAL_1", "Skrews_1", "Frame2_1", "Handle_and_Botttom_1", "Metal_part_1", "bar set", "Mirror_1", "cutlery", "spoon_1", "board", "accessory", "Doorwood_1", "Drawer_1", "shade inside", "wheeloutside_1", "metal_1", "Inner_1", "hinge", "Nose_1", "weight", "coaster", "hanger", "Handle_And_Base_1", "rim", "pan", "Top_and_Handle_1", "wire frame", "bottom", "tub", "outer rings", "Front_1", "Small_outer_1", "rod", "_steel_1", "reindeer", "coat", "Wheels_1", "wall planter", "lamp", "Wood_Stool_small_1", "WheelMetal_1", "inner ring", "salad server", "wire cross", "Stand_1", "rabbit", "binding", "Top_1", "LargeLegs_1", "knife", "balls frame", "front curtain", "BULB_1", "Console-Mesh", "center", "leaf ball", "railing", "Small_spoon_1", "newspaper", "Wires1-Mesh", "suction pad", "Cake_plate_1", "ball", "side", "text", "bar cart", "Handle1_1", "leg", "wire flowers", "legs", "duvet", "Mdf_1", "Candle_top_1", "hanger_1", "vase", "stand", "foot", "stockpot", "tray", "jar", "seat-Mesh", "HOLDER_1", "mic stand", "brass_1", "wax", "shoes", "book border", "serving head", "legs_1", "wall divider", "lantern", "leaf ring", "bottle", "wood-Mesh", "Other_Metal_Part_1", "inner_part1_1", "moose head", "Bottom-Mesh", "pot", "rings", "rider bottom", "surface", "knob support", "shade", "tray outer surface", "tail", "suit", "Wire_1", "lip", "seat_1", "flask", "rack", "_tophandle_1", "Base_1", "stiches", "horse", "Gold_1", "Cupbord_1", "knob", "page", "office chair", "Candle_1", "legs2_1", "Big_legs_1", "basket", "Body_1", "rope", "cup", "beads", "tray inner surface", "wire urn", "engraving", "screw head", "Middle_1", "papers", "chain_1", "Outer_Ring_1", "outer casing", "head", "basin", "maintoppart_1", "tray outside surface", "spoon", "Bottoms_1", "grillpart_1", "bucket", "container", "ottoman", "Metal_Stool_Large_1", "mattress", "Spoon1_1", "pipe_1", "Inner_Ring_1", "soil", "bench", "wheel", "sculpture", "footrest", "wall mount", "slider", "Metal_Part_1", "socket", "paper holder", "knob post", "bird house", "Circle_1", "bird feeder", "drawer", "mug", "rivet", "shelf", "frame", "Skrews-Mesh", "wheelinside_1", "cabinet", "_leg_1", "trivet", "back curtain", "side rail", "Wires1_1", "drawers", "Skrew_1", "candle stand", "Bridge_1", "stick", "Metal_Stool_Small_1", "chopping board", "globe", "_wheel_1", "side design", "Border_1", "CirPattern4_1", "Caps-Mesh", "inside border", "bottom_1", "platter", "wall pocket", "grip", "black2_1", "LargeTop_1", "barbeque grill", "pillow", "finial", "Metal-Mesh", "Circles_1", "napkin holder", "bulb_1", "cutlery head", "engraved pattern", "planter", "sofa", "tshirt", "wood_part_1", "fork", "spray", "lotus", "Small_legs_1", "MetalGold2_1", "Middle_Marble_1", "prop", "Marble_1", "Table_1", "stole", "flower", "design", "plate", "media console", "book", "TOP1_1", "Top_Wood_1", "tray interior", "Back_1", "decorative", "bedsheet", "cap", "rung", "Base_feet_1", "lamp holder", "dining_room_Cusion_1", "Cupboard_1_0_1", "Metal1_1", "Other_metal_Part_1", "pineapple", "metal_frame_1", "stand inner surface", "screw_1", "Leg-Mesh", "Steel_1", "cloth", "border", "wire", "clip", "MetalGold_1", "bottom surface", "logo", "artichoke cachepot", "leg_wood_1", "connector", "Outer_1", "pizza blade", "eiffeltower", "shelf hanger", "Metal_1", "frog", "outer_part1_1", "steel_1", "stone", "ladder", "Bottom_1", "Cylinder007-Mesh", "Handels_And_Legs_1", "water cane", "grill", "Glass_1", "corner shelf", "bulb", "lever", "back", "WOOD_1", "Wood_Base_and_Back_support_1", "Trolly_1", "Wood_Stool_Large_1", "leg_wood-Mesh", "Cushion_1", "cheese set", "Wheel_1", "leaf", "zipper", "corner", "teapot", "knob holder", "Gril_1", "spike", "body", "details", "tube", "branch", "obelisk", "upper part", "Innerparts_1", "cactus", "dining_room_wood_1", "Medium_fork_1", "Inner_Wood_1", "lid", "candle", "metal", "underside", "neck", "carpet", "tea", "support", "jug", "handle", "backrest", "toy", "screw", "lady", "bag", "rider", "pouch", "trolley", "cake stand", "stopper_1", "Basewood_1", "pant", "Handle_1", "chair", "Base-Mesh", "Big_Top_1", "skirt", "Ring_1", "Glass-Mesh", "armrest", "top", "holder_1", "Fuse_1", "top tray", "Wood_1", "Cupboard_1_1_1", "Inlay_1_1_1", "bowl", "top_1", "Seat_1", "leveller", "Handles_and_Legs_1", "trigger", "MDF_1", "table", "_marble_1", "Top_and_Bottom_1", "rider body", "STEEL_1", "wreath", "Ineer_1", "wind chimes", "glass_1", "Pipes2_1", "cushion", "pumpkin", "trim", "GlassOuter_1", "book cover", "bottom plate", "glass", "Handles_And_Legs_1", "nightstand", "stool", "bottom shelf", "hook", "seat", "Strips-Mesh", "mesh wrapped sides", "Handlesmetal_1", "Weaves_1", "Pant_1", "Plane002-Mesh", "cover", "Seat-Mesh", "ring", "base", "handle_1", "side_1", "wood_1", "safe", "letter", "adjuster wheel", "TOP_1", "sieve", "candle wax", "bowl_1", "ChamferBox002_1", "MetalGold1_1", "star", "Sphere_1", "top surface", "door", "button", "Pattern-Mesh", "dresser", "SmallTop_1", "handle2_1", "holder", "Handles_1", "bookcase", "net_1", "desk", "middle_1", "crown", "desk cover", "wheel_1", "stretcher", "none"]
material = ['Ceramic', 'Fabric', 'Glass', 'Leather', 'Marble', 'Metal', 'Paper','Plastic', 'Wax', 'Wood', 'none', np.nan]
objs =  ['planters', 'bar cabinet', 'pouf', 'salad server', 'bench', 'finial', 'barbeque grill', 'kettle', 'Pumpkin', 'glass', 'bowl ', 'Stole', 'wall shelf', 'photoframe', 'eiffel tower', 'napkin holder', 'sieve', 'wall floral hook', 'finial wire object', 'suit', 'cushion', 'pine flower orn', 'tray', 'jar', 'maniquin', 'spotlight', 'media console', 'CAKE SERVER', 'pot', 'toy', 'wreath', 'CAKE STAND', 'fillet bench', 'knob', 'Medicine Box', 'decorative', 'obelisks', 'Finial', 'purse', 'pet feeder', 'Basket', 'dresser', 'bar', 'mic stand', 'christmas tree', 'rolling pin', 'platter', 'bed', 'mug', 'Cheese board', 'cheese set', 'Wine Rack', 'floral bunch orn', 'spray bottle', 'toys', 'water cane', 'serving table', 'knife', 'pizza cutter', 'belt', 'chopping board', 'DRESSER', 'bowl', 'tall unit', 'LAMP', 'plant finial', 'wine rack', 'Obelisk', 'caserole', 'strainer', 'ornament', 'clock', 'Candle Holder', 'utensils', 'wire cross', 'pumpkins', 'shade', 'cactus', 'CHEESE SET', 'trays', 'wall planter', 'lever', 'drawer', 'paper holder', 'Chair', 'Reindeer', 'pineapple', 'leaf t-lite holder', 'Napkin Holder', 't light holder', 'newspaper', 'waffle maker', 'fork', 'Table', 'round table', 'vase', 'cake stand', 'bird house', 'Carpet', 'sofa', 'plant', 'aviator', 'finials', 'door unit', 'jug', 'bar set', 'lantern', 'cabinet', 'plate', 'ladder', 'pouch', 'coaster', 'stand', 'obelisk', 'cloche plate', 'utensil holder', 'stool', 'trivet', 'ottoman', 'bongo', 'container', 'deer head', 'wind chimes', 'bird feeder', 'wall pockets', 'wall pocket', 'wall light', 'prop', 'bulb', 'farmstead finials', 'room', 'shelf', 'Spoon', 'Dining Table', 'table', 'vases', 'globe', 'Candle holder', 'Tray', 'tube', 'wall divider', 'basket', 'Cabinet', 'cutlery', 'artichoke cachepot', 'MEDIA UNIT', 'candle holder', 'bottle', 'Breakfast Trays', 'curtain', 'desk', 'safe', 'lamp', 'lotus', 'rabbit', 'tub', 'wine glass', 'Round table', 'chair', 'spoon', 'spice box', 'nightstand', 'bar cart', 'pillow', 'helf hanger', 'basin', 'urn', 'wine cooler', 'flask', 'Spotlight', 'crown', 'hanging light', 'pan', 'weight', 'wall rack', 'Artichoke Pots', 'planter', 'trolley', 'serving plate', 'corner shelf', 'Pine Flower Orn', 'baskets', 'drop lighting', 'tea set', 'bookcase', 'coffee table', 'stockpot', 'none']
data_dir = "/home/varshnt/tezuesh/data/rendered_img/"

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
BATCH_SIZE = 128
#######################################################
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_model_dict(path):
    a = json.load(open(path))
    dict_ = {}
    for key, value in a.items():
        dict_[key] = value['model_class']
    return dict_

id_name = get_model_dict('/home/varshnt/tezuesh/data/parts_annotation.json')


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

                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )




class PartDataset(Dataset):
    def __init__(self, data_dir, t='train', transforms=None):
        self.data_dir = data_dir

        # Read the splits
        splits = unpickle_data('/home/varshnt/tezuesh/data/test_train.pkl')

        model_id = json.load(open("id_model.json")) 
        train_model_ids = splits['train']['model_ids']
        test_model_ids = splits['test']['model_ids']
        # train_model_ids = splits['test']['model_ids']
        val_model_ids = splits['val']['model_ids']


        # Read the labels
        self.labels = unpickle_data('/home/varshnt/tezuesh/data/label.pkl')
        

        # Read the models data
        # self.models = pd.read_csv('v_{}_{}_dataset.csv'.format(view, t))

        # Get a list of all jpeg files in the data dir
        result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
        
        # use only models in the split
        self.models = []
        
        path_bb = "/home/varshnt/tezuesh/data/bounding_box/bb_mat_part/"
        # path_bb = "/home/varshnt/tezuesh/data/bounding_box/bb_mat_test/"
        self.part_bb = []

        for el in result:
            img_id =  os.path.basename(el).split('.jpg')[0]
            if img_id in self.labels:
                _id = img_id.split('_')[0]
                # return

                if t == 'train':
                    if _id in train_model_ids:
                        if os.path.exists(path_bb + _id + '_'+img_id.split('_')[-1]):
                            self.part_bb.append(path_bb + _id + '_'+img_id.split('_')[-1])
                            self.models.append(el)
                elif t == 'val':
                    if _id in val_model_ids:
                        if os.path.exists(path_bb + _id + '_'+img_id.split('_')[-1]):
                            self.part_bb.append(path_bb + _id + '_'+img_id.split('_')[-1])
                            self.models.append(el)
                else:
                    if _id in test_model_ids:
                        if os.path.exists(path_bb + _id + '_'+img_id.split('_')[-1]):
                            self.part_bb.append(path_bb + _id + '_'+img_id.split('_')[-1])
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
        # Read the image
        # record = self.models.iloc[item]
        record = self.models[item]
        
        p = record

        ori_image = cv2.imread(p, cv2.IMREAD_COLOR)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        ori_image /= 255.0
        image = ori_image.copy()

        # Apply transformation
        # image = self.transforms(image)

        partname_bb= get_bounding_box_part(self.part_bb[item])
        material_bb = get_bounding_box_mat(self.part_bb[item])


        # Part Param -------------------------------------------------------
        
        part_boxes = []
        label_part = []
        for key, value in partname_bb.items():
            part_boxes.append(value)
            if key not in parts:
                label_part.append(parts.index('none'))
            else:
                label_part.append(parts.index(key))
        
        part_boxes = torch.tensor(part_boxes)
        label_part =  torch.tensor(label_part)

        area = part_boxes[:,2]*part_boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms:
            sample = {
                'image': ori_image.copy(),
                'bboxes': part_boxes,
                'labels': label_part
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            label_part = sample['labels']
        
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target_part = {}
        target_part['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_part['labels'] = torch.as_tensor(label_part,dtype=torch.long)
        target_part['image_id'] = torch.tensor([item])
        target_part['area'] = area


        # Object Param -------------------------------------------------------
        w = (part_boxes[:,2] + part_boxes[:,0]).max() - part_boxes[:,0].min()
        h = (part_boxes[:,3] + part_boxes[:,1]).max() - part_boxes[:,1].min()

        obj_bb = torch.tensor([[part_boxes[:,0].min(), part_boxes[:1].min(), w, h]])
        obj_id = record.split("/")[-1].split("_")[0]
        if id_name[obj_id] in objs:
            label_obj = torch.tensor([objs.index(id_name[obj_id])])
        else:
            label_obj = torch.tensor([objs.index('none')])


        if self.transforms:
            sample = {
                'image': ori_image.copy(),
                'bboxes': obj_bb,
                'labels': label_obj
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            label_part = sample['labels']
        
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target_obj = {}
        target_obj['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_obj['labels'] = torch.as_tensor(label_part,dtype=torch.long)
        target_obj['image_id'] = torch.tensor([item])
        target_obj['area'] = area


        # Material Param -------------------------------------------------------

        boxes = []
        label_mat = []
        for key, value in material_bb.items():
            boxes.append(value)
            if key not in parts:
                label_mat.append(material.index('none'))
            else:
                label_mat.append(material.index(key))
            
        
        boxes = torch.tensor(boxes)        
        label_mat =  torch.tensor(label_mat)

        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms:
            sample = {
                'image': ori_image.copy(),
                'bboxes': boxes,
                'labels': label_mat
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            label_mat = sample['labels']
        
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target_mat = {}
        target_mat['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target_mat['labels'] = torch.as_tensor(label_mat,dtype=torch.long)
        target_mat['image_id'] = torch.tensor([item])
        target_mat['area'] = area
    

        img_id = os.path.basename(p).split('.jpg')[0]
        obj_class = np.array(self.labels[img_id])

        return image, target_part,target_obj, target_mat, obj_class, record



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
        self.bbox_part = MLP(self.in_features, self.in_features,4,3)
        self.num_que_part = num_que_parts

        self.model_obj = nn.Linear(in_features=self.in_features,out_features=num_cl_obj)
        self.bbox_obj = MLP(self.in_features, self.in_features,4,3) 
        # nn.Linear(in_features=self.in_features,out_features=4)
        self.num_que_obj = num_que_obj

        self.model_ma = nn.Linear(in_features=self.in_features,out_features=num_cl_ma)
        self.bbox_ma = MLP(self.in_features, self.in_features,4,3) 
        # nn.Linear(in_features=self.in_features,out_features=4)
        self.num_que_ma = num_que_ma
        
        
    def forward(self,images):

        activation = {}
        def getActivation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        
        h = self.model.transformer.decoder.register_forward_hook(getActivation('norm'))
        features = self.model(images)
        features = activation['norm']
        parts =  {'pred_logits': self.model_part(features)[-1], 
                'pred_boxes': self.bbox_part(features).sigmoid()[-1]}

        obj =  {'pred_logits': self.model_obj(features)[-1], 
                'pred_boxes': self.bbox_obj(features).sigmoid()[-1]}

        ma =  {'pred_logits': self.model_ma(features)[-1], 
                'pred_boxes': self.bbox_ma(features).sigmoid()[-1]}        
        return parts, obj, ma




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
    
    for step, (images, target_part, target_obj, target_mat, image_ids, obj_class) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        target_part = [{k: v.to(device) for k, v in t.items()} for t in target_part]
        target_obj = [{k: v.to(device) for k, v in t.items()} for t in target_obj]
        target_mat = [{k: v.to(device) for k, v in t.items()} for t in target_mat]
        

        part, obj, mat = model(images)

       
        part['pred_boxes'] = part['pred_boxes'].reshape(-1, 100,4)
        mat['pred_boxes'] = mat['pred_boxes'].reshape(-1, 100,4)
        obj['pred_boxes'] = obj['pred_boxes'].reshape(-1, 100,4)

        part['pred_logits'] = part['pred_logits'].reshape(BATCH_SIZE, 100,num_cl_part)
        mat['pred_logits'] = mat['pred_logits'].reshape(BATCH_SIZE, 100,num_cl_mat)
        obj['pred_logits'] = obj['pred_logits'].reshape(BATCH_SIZE, 100,num_cl_obj)

       
        loss_di_part = criterion_part(part, target_part)
        loss_di_obj = criterion_obj(obj, target_obj)
        loss_di_mat = criterion_mat(mat, target_mat)
        
        weight_dict = criterion_part.weight_dict
        
        losses_part = sum(loss_di_part[k] * weight_dict[k] for k in loss_di_part.keys() if k in weight_dict)
        losses_obj = sum(loss_di_obj[k] * weight_dict[k] for k in loss_di_obj.keys() if k in weight_dict)
        losses_mat = sum(loss_di_mat[k] * weight_dict[k] for k in loss_di_mat.keys() if k in weight_dict)

        losses = losses_part + losses_obj + losses_mat
        
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
        
        for step, (images, target_part, target_obj, target_mat, image_ids, obj_class) in enumerate(tk0):
            
            images = list(image.to(device) for image in images)
            target_part = [{k: v.to(device) for k, v in t.items()} for t in target_part]
            target_obj = [{k: v.to(device) for k, v in t.items()} for t in target_obj]
            target_mat = [{k: v.to(device) for k, v in t.items()} for t in target_mat]

            part, obj, mat = model(images)

       
            part['pred_boxes'] = part['pred_boxes'].reshape(-1, 100,4)
            mat['pred_boxes'] = mat['pred_boxes'].reshape(-1, 100,4)
            obj['pred_boxes'] = obj['pred_boxes'].reshape(-1, 100,4)

            part['pred_logits'] = part['pred_logits'].reshape(BATCH_SIZE//4, 100,num_cl_part)
            mat['pred_logits'] = mat['pred_logits'].reshape(BATCH_SIZE//4, 100,num_cl_mat)
            obj['pred_logits'] = obj['pred_logits'].reshape(BATCH_SIZE//4, 100,num_cl_obj)

        
            loss_di_part = criterion_part(part, target_part)
            loss_di_obj = criterion_obj(obj, target_obj)
            loss_di_mat = criterion_mat(mat, target_mat)
            
            weight_dict = criterion_part.weight_dict
            
            losses_part = sum(loss_di_part[k] * weight_dict[k] for k in loss_di_part.keys() if k in weight_dict)
            losses_obj = sum(loss_di_obj[k] * weight_dict[k] for k in loss_di_obj.keys() if k in weight_dict)
            losses_mat = sum(loss_di_mat[k] * weight_dict[k] for k in loss_di_mat.keys() if k in weight_dict)

            losses = losses_part + losses_obj + losses_mat

            
            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
    
    return summary_loss


def collate_fn(batch):
    return tuple(zip(*batch))

def run(fold):
    
    train_dataset = PartDataset(data_dir,'train',get_train_transforms())
    
  
    valid_dataset = PartDataset(
        data_dir,
        'val',
        get_valid_transforms())
    
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
        valid_loss = eval_fn(valid_data_loader, model,(criterion_part, criterion_obj, criterion_mat), device)
        
        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')

run(fold=0)