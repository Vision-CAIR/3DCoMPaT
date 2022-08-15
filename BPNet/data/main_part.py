import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from utils import unpickle_data, get_bounding_box_part


########################################################
BATCH_SIZE = 512
parts = ['Gold_Streak1_1', 'Candle_1', 'Leg_2', 'Top2_1', 'sm_spoon_handle_1', 'Inner_1', 'Big_Julep_Vase_1', 'benchLegs_1', 'Outer_part001_1', 'Wood_Part_1', 'benchFabric_1', 'Metal_CROSS_LEATHER_OTTOMAN_1', 'knob_1', 'decorative_bowl_1', 'Gold_1', 'Text1_1', 'Mattress_1', 'SmallLegs_1', 'Glass_Part_1', 'Glass2_1', 'Inner_part001_1', 'Spoon2_1', 'Metal__1', 'Middle_1', 'top_1', 'MetalGold2_1', 'Mirror_1', 'Outer_part_1', 'top_faces_1', 'fork_handle_1', 'Seat_HP_1', 'Metal1_1', 'Planter_1', 'LargeLegs_1', 'Wood_Top_1', 'Seat_1', 'Outer_1', 'Wood1_1', 'Leg_3', 'Back_HP_1', 'Top_03_1', 'LargeTop_1', 'Designwood_1', 'Drawer1_1', 'wire_1', 'Body_1', 'Corners_1', 'weawing_1', 'Fork2_1', 'Glass_1', 'Base3_1', 'door_1', 'wood_1', 'pipe_1', 'top_02_1', 'SmallTop_1', 'legs_1', 'Candle_Holder_1', 'Handle_1', 'Obj_000001_1', 'Obj_000002_1', 'Leather_CROSS_LEATHER_OTTOMAN_1', 'lg_spoon_handle_1', 'faces_1', 'Base_1', 'METAL_1', 'Drawer_1', 'Inner_part_1', 'Rop_1', 'Metal_1', 'Wood_1', 'RightPart_1', 'Marble_1', 'Wood2_1', 'Bolt_1', 'patti_1', 'Printing_Part_1', 'Table_1', 'Design_1', 'Metal_Steel2_1', 'Top_01_1', 'Marble1_1', 'Top_1', 'Inside_1', 'Base2_1', 'body_1', 'Top_02_1', 'base_1', 'SOFA_legs_1', 'Cabinet_1', 'Fabric_1', 'Leg_1', 'Wood3_1', 'Lg_spoon_handle_1', 'bolls_1', 'wood_CROSS_LEATHER_OTTOMAN_1', 'Box_1', 'Rubber_1', 'Mdf_1', 'Top_Base_1', 'Steel_Plate_1', 'Upper_Part2_1', 'Metal_Stand_1', 'SOFA_Fabric_1', 'Fork1_1', 'Stand_1', 'Sofa_1', 'Knife_1', 'Marble_Part_1', 'WOOD_1', 'Spoon_1', 'Leather_1', 'Bottom_1', 'Metal_Part_1', 'metal_1', 'knife_handle_1', 'Inner_Tray_1', 'Shape001_1', 'top_01_1', 'Pillow_1', 'Love1_1', 'support_1', 'Small_Julep_Vase_1', 'Box004_1', 'Wood4_1', 'Spoon1_1', 'FABRIC_1', 'bulb_1', 'space_1', 'handle_1', 'Caps_1', 'benchWood_1', 'LeftPart_1', 'Legs_1']
num_classes = len(parts)
data_dir = "/home/varshnt/tezuesh/data/rendered_img/"
num_queries = 30
seed = 42
null_class_coef = 0.5
BATCH_SIZE = 2
LR = 2e-5
EPOCHS = 50
#######################################################



def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),            
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      A.ToGray(p=0.01),                      
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.Resize(height=224, width=224, p=1),
                      A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,

                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=224, width=224, p=1.0),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )





class PartDataset(Dataset):
    def __init__(self, data_dir, t='train', transforms=None):
        self.data_dir = data_dir

        # Read the splits
        splits = unpickle_data('/home/varshnt/tezuesh/data/test_train.pkl')
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
        path_bb = "/home/varshnt/tezuesh/data/bounding_box/bb_mat_test/"
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

        image = cv2.imread(p, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Apply transformation
        # image = self.transforms(image)

        partname_bb= get_bounding_box_part(self.part_bb[item])

        boxes = []
        labels = []
        for key, value in partname_bb.items():
            boxes.append(value)
            labels.append(parts.index(key))
        
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        # labels =  np.zeros(len(boxes), dtype=np.int32)
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
        
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([item])
        target['area'] = area
    

        img_id = os.path.basename(p).split('.jpg')[0]
        obj_class = np.array(self.labels[img_id])

        return image, target, obj_class, record



class DETRModel(nn.Module):
    def __init__(self,num_classes, num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
        self.in_features = self.model.class_embed.in_features

        # self.model_part = nn.Linear(in_features=self.in_features,out_features=num_cl_parts)
        # self.num_que_part = num_que_parts

        # self.model_obj = nn.Linear(in_features=self.in_features,out_features=num_cl_obj)
        # self.num_que_obj = num_que_obj

        # self.model_ma = nn.Linear(in_features=self.in_features,out_features=num_cl_ma)
        # self.num_que_ma = num_que_ma
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)


matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

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
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for step, (images, targets, image_ids, obj_class) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images)
        print(type(output))
        print("target", targets[0]['boxes'].shape, output['pred_boxes'].shape)
        print(output['pred_logits'].shape)
        print(len(targets))
        exit(0)
        
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
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
    criterion.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids, obj_class) in enumerate(tk0):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
        
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
    
    return summary_loss


def collate_fn(batch):
    return tuple(zip(*batch))

def run(fold):


    
    train_dataset = PartDataset(
        data_dir,
        'train',
        get_train_transforms())
  
    valid_dataset = PartDataset(
        data_dir,
        'val',
        get_train_transforms())
    
    train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )
    
    device = torch.device('cuda')

    ngpus_per_node = torch.cuda.device_count()

    
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
    criterion = criterion.to(device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_loss = 10**5
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        valid_loss = eval_fn(valid_data_loader, model,criterion, device)
        
        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')

run(fold=0)