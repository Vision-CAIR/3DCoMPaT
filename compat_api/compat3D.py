# The following API functions are defined:
#  Compat3D       - 3DCompat api class that loads 3DCompat annotation file and prepare data structures.
# load_raw_models -  load raw shape without textures given the specified ids.
# load_stylized_3d - load stylized 3d shapes given the specified ids.

import os
import os.path as osp
import glob
import pandas as pd
import h5py
import numpy as np
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import trimesh

import torch
from torch.utils.data import Dataset

import pdb

_ALL_PARTS = ['access_panel', 'adjuster', 'aerator', 'arm', 'armrest', 'axle',
       'back', 'back_flap', 'back_horizontal_bar', 'back_panel', 'back_stretcher', 'back_support',
       'back_vertical_bar', 'backrest', 'bag_body',
       'ball_retrieving_pocket', 'base', 'beam', 'bed_post',
       'bed_surrounding_rail', 'bedsheet', 'bedskirt', 'bench', 'blade',
       'blade_bracket', 'body', 'border', 'bottom', 'bottom_panel', 'bow',
       'bowl', 'brace', 'bracket', 'brake', 'bulb', 'bush', 'button',
       'cabinet', 'candle', 'canopy', 'cap', 'cap_retainer', 'case',
       'caster', 'chain', 'chain_stay', 'channel', 'cleat', 'container',
       'containing_things', 'control', 'cooking', 'corner_pockets',
       'cue_stick', 'cushion', 'deck', 'decoration', 'design', 'dial',
       'disposer', 'door', 'downrod', 'drain', 'drawer',
       'duvet', 'enginestretcher', 'eyehook', 'fabric_design', 'fan',
       'faucet', 'feeder', 'fin', 'finial', 'flange', 'flapper',
       'flapper_support', 'floor', 'flush_handle', 'flush_push_button',
       'foot', 'foot_base', 'footboard', 'footrest', 'fork', 'frame',
       'front', 'front_flap', 'front_side_rail', 'gear_levers', 'glass',
       'grill', 'grip_tape', 'handle', 'handlebars', 'hanger', 'hardware',
       'harp', 'head', 'head_support', 'headboard', 'headlight',
       'headrest', 'headset', 'helm', 'hinge', 'hood', 'hook', 'hose',
       'hour_hand', 'hull', 'igniter', 'inner_surface', 'keel',
       'keyboard_tray', 'knob', 'lamp_surrounding_frame', 'leg',
       'leg_stretcher', 'level', 'leveller', 'lever', 'lid', 'light',
       'locks', 'long_ribs', 'lug', 'mast', 'mattress', 'mechanism',
       'minute_hand', 'mirror', 'motor_box', 'mouth', 'neck', 'net',
       'net_support', 'nozzle', 'number', 'number_plate',
       'open_close_button', 'outer_surface', 'paddle', 'pedal',
       'pendulum', 'perch', 'pillars', 'pillow', 'pipe', 'play_field',
       'plug', 'pocket', 'pole', 'propeller', 'propeller_blade', 'pulley',
       'rails', 'rear_side_rail', 'rear_view_mirror', 'rear_window',
       'rim', 'rocker', 'rod', 'rod_bracket', 'roof', 'rope', 'rudder',
       'runner', 'saddle', 'screen', 'screw', 'seat', 'seat_cover',
       'seat_cushion', 'seat_stay', 'second_hand', 'shade_cloth', 'shaft',
       'shelf', 'short_ribs', 'shoulder', 'shoulder_strap', 'shower_head',
       'shower_hose', 'side_panel', 'side_pockets', 'side_walls',
       'side_windows', 'sink', 'slab', 'socket', 'spokes', 'spout',
       'sprayer', 'spreader', 'stand', 'starboard', 'stem', 'step',
       'stopper', 'strap', 'stretcher', 'strut', 'support', 'surface',
       'switch', 'table', 'tabletop_frame', 'taillight', 'tank_cover',
       'throw_pillow', 'tie_wrap', 'top', 'top_cap', 'top_panel', 'trap',
       'tray_inner', 'truck', 'trunk', 'tube', 'tyre', 'unit', 'valve',
       'vertical_divider_panel', 'vertical_side_panel', 'wall_mount',
       'water', 'water_tank', 'wax_pan', 'wheel', 'windows', 'windshield',
       'wing', 'wiper', 'wire', 'zipper']


# parts index and reversed index
part_to_idx = dict(zip(_ALL_PARTS, range(len(_ALL_PARTS))))


class CompatLoader3D(Dataset):
    """
    Base class for 3D dataset loaders.

    Args:
        root_dir:    Base dataset URL containing data split shards
        split:       One of {train, valid}.
        n_comp:      Number of compositions to use
        cache_dir:   Cache directory to use
        view_type:   Filter by view type [0: canonical views, 1: random views]
    """
    def __init__(self, root_dir="./data/", split="train", n_comp=1, cache_dir=None, view_type=-1):
        if view_type not in [-1, 0, 1]:
            raise RuntimeError("Invalid argument: view_type can only be [-1, 0, 1]")
        if split not in ["train", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)

        self.root_dir = os.path.normpath(root_dir)

        self.cache_dir = cache_dir

        self.view_type = view_type

        df = pd.read_csv(osp.join(self.root_dir, "metadata/model.csv"))
        all_cats = list(set(df['model'].tolist()))
        all_classes = dict(zip(all_cats, range(len(all_cats))))
        id_to_cat = dict(zip(df['id'].tolist(), df['model'].tolist()))
        
        labels = []
        for key in df['id']:
            labels.append(all_classes[id_to_cat[key]])
    
        self.shape_ids = df['id'].tolist()
        self.labels = np.array(labels).astype('int64')


    def __getitem__(self, index, sample_point=True):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D unstylized models
        """
        shape_id = self.shape_ids[index]
        gltf_path = os.path.join(self.root_dir, 'raw_models', shape_id + '.glb')
        mesh = trimesh.load(gltf_path)

        if not sample_point:
            return shape_id, mesh
        else:
            v = []
            segment = []
            for g_name, g_mesh in mesh.geometry.items():
                g_name = g_name.lower()
                if g_name in part_to_idx:
                    # Glb name is same as defined
                    part_name = g_name
                elif g_name in part_index:
                    # Glb name is different from defined. We regulated the name.
                    part_name = part_index[g_name]
                else:
                    # If there are still some incorrect one.
                    part_name = g_name.split('_')[0]
                    if part_name not in classes:
                        part_name = difflib.get_close_matches(g_name, parts)[0]
                # Add the vertex
                v.append(g_mesh)
                # Add the segmentation Labels
                segment.append(np.full(g_mesh.faces.shape[0], part_to_idx[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id = trimesh.sample.sample_surface(combined, count=5000)
            # sample_xyz = pc_normalize(sample_xyz)
            # If there are no style models, color info set as zero
            sample_colors = np.zeros_like(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]

            return shape_id, sample_xyz, sample_colors, sample_segment

    def eval_3D_Shape_Cls(self, y_pred, y_true):
        """
        Evaluation function for 2D shape classification

        Args:
          y_pred: a numpy array, each line contains predicted classification label.
          y_true: a numpy array, each line contains GT classification label.
        """
        assert len(y_pred) == len(y_true)
        
        label_values = np.unique(y_true)
        cf_mat = confusion_matrix(y_true, y_pred)

        instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
        class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
        class_avg_acc = np.mean(class_acc)
        return instance_acc, class_avg_acc


    def eval_3D_Part_Seg(self, y_pred, y_true):
        """
        Evaluation function for 3D shape classification

        Args:
          pred_file: a numpy array, each line contains predicted part labels for all points.
          gt_file: a numpy array, each line contains GT part labels for all points.
        """
        assert len(y_pred) == len(y_true)
        
        label_values = np.unique(y_true)
        cf_mat = confusion_matrix(y_true, y_pred)

        instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
        class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
        return instance_acc, class_acc
        
    def eval_3D_Material_Seg(self, y_pred, y_true):
        """
        Evaluation function for 2D material segmentation

        Args:
          y_pred: a numpy array, each line contains predicted segmentation labels for all points.
          y_true: a numpy array, each line contains GT segmentation labels for all points.
        """
        assert len(y_pred) == len(y_true)

        f1 = metrics.f1_score(y_true, y_pred)
        prec = metrics.average_precision_score(y_true, y_pred)
        mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')

        return f1, prec, mIoU

    def eval_GCR_3D(pred_file, gt_file):
        """
        Evaluation function for 3D shape classification

        Args:
          pred_file: a txt file, each line contains shape_id, pred_cls for all points.
          gt_file: a txt file, each line contains shape_id, gt_cls for all points.
        """
        # To be updated

        return None

class CompatLoader_stylized3D(CompatLoader3D):
    """
      Stylized 3D dataset loaders.

      Args:
          root_dir:    Base dataset URL containing data split shards
          split:       One of {train, valid}.
          n_comp:      Number of compositions to use
          cache_dir:   Cache directory to use
          view_type:   Filter by view type [0: canonical views, 1: random views]
    """
    def __init__(self, root_dir="./data/", split="train", n_comp=1, cache_dir=None, view_type=-1):
        super().__init__(root_dir, split, n_comp, cache_dir, view_type)

    def __getitem__(self, index, style_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D stylized models
        """
        shape_id = self.shape_ids[index]
        gltf_path = os.path.join(self.root_dir, 'rendered_models/', shape_id,  shape_id + '_' + style_id + '.glb')
        mesh = trimesh.load(gltf_path)
        
        if not sample_point:
            return mesh
        else:
            v = []
            segment = []
            for g_name, g_mesh in mesh.geometry.items():
                g_name = g_name.lower()
                if g_name in part_to_idx:
                    # Glb name is same as defined
                    part_name = g_name
                elif g_name in part_index:
                    # Glb name is different from defined. We regulated the name.
                    part_name = part_index[g_name]
                else:
                    # If there are still some incorrect one.
                    part_name = g_name.split('_')[0]
                    if part_name not in classes:
                        part_name = difflib.get_close_matches(g_name, parts)[0]
                # Add the vertex
                v.append(g_mesh)
                # Add the segmentation Labels
                segment.append(np.full(g_mesh.faces.shape[0], part_to_idx[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id = trimesh.sample.sample_surface(combined, count=5000)
            # sample_xyz = pc_normalize(sample_xyz)
            # If there are no style models, color info set as zero
            sample_colors = np.zeros_like(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]

            return shape_id, sample_xyz, sample_colors, sample_segment

# Evaluation code for GCR task

from collections import defaultdict
import numpy as np
import pdb


class BboxEval:
    def __init__(self, shape='top1'):
        self.shape = shape
        self.per_obj_occ_bboxes = defaultdict(float)  # Number of object occured for each object, used for calculate gnd-value-all
        self.per_obj_all_correct_bboxes = defaultdict(float)  # Number of object occured for each object, used for calculate gnd-value-all
        
        self.per_obj_parts_bboxes = defaultdict(float)  # Number of GT parts per object, used for calculate gnd-value
        self.per_obj_parts_correct_bboxes = defaultdict(float)  # Number of (part, mat) pair predicted correctly and grounding all of them correctly for each object, used for calculate gnd-value

        self.per_obj_occ = defaultdict(float)  # Number of object occured for each object, used for calculate value_all
        self.per_obj_all_correct = defaultdict(float)  # All predicted (part, mat) pair for given object are correct, used for calculate value_all

        self.per_obj_parts = defaultdict(float)  # Number of parts per object, used for calculate value
        self.per_obj_parts_correct = defaultdict(
            float)  # Number of (part, mat) pair predicted correctly for a given object, used for calculate value

        self.all_objs = 0.0  # Number of objects
        self.correct_objs = 0.0  # Number of objects correctly predicted

    def obj(self):  # object accuracy
        return self.correct_objs / self.all_objs

    # accuracy of predicting both part category and the material of a given part correctly.
    def value(self): 
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts:
            sum_value += float(self.per_obj_parts_correct[obj]) / float(self.per_obj_parts[obj])
            total_value += 1.0
        return sum_value / total_value

    # accuracy of predicting all the (part, material) pairs of a shape
    def value_all(self): 
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ:
            sum_value_all += float(self.per_obj_all_correct[obj]) / float(self.per_obj_occ[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    # accuracy of predicting both part category and the material of a given part as well as correctly grounding it
    def value_bbox(self):
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts_bboxes:
            sum_value += float(self.per_obj_parts_correct_bboxes[obj]) / float(self.per_obj_parts_bboxes[obj])
            total_value += 1.0
        return sum_value / total_value

    # accuracy of predicting all the (part, material) pairs of a given shape correctly and grounding all of them correctly
    def value_all_bbox(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ_bboxes:
            sum_value_all += float(self.per_obj_all_correct_bboxes[obj]) / float(self.per_obj_occ_bboxes[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    # return all evaluation metrics
    def eval_all(self):
        acc, value, value_all, gnd_value, gnd_value_all = self.obj(), self.value(), self.value_all(), self.value_bbox(), self.value_all_bbox()
        return acc, value, value_all, gnd_value, gnd_value_all

    # Evaluation on all predictions and GTs
    def eval_GCR(pred_objs, pred_parts, pred_mats, gt_objs, gt_parts, gt_mats, part2mats, model_ids):
        '''
        Provide a list of predictions and GTs
        '''
        print('########## Start GRC Evaluation ##########')
        for pred_obj, pred_part, pred_mat, gt_obj, gt_part, gt_mat, part2mat, model_id in zip(pred_objs.cpu().numpy(), pred_parts.cpu().numpy(), pred_mats.cpu().numpy(), gt_objs.cpu().numpy(), gt_parts.cpu().numpy(), gt_mats.cpu().numpy(), part2mats.cpu().numpy(), model_ids.cpu().numpy()):

            pred_part_list = np.unique(pred_part)
            pred_mat_list = np.unique(pred_mat)
            gt_mat_list = np.unique(gt_part)
            gt_part_list = np.unique(gt_mat)

            self.update(pred_obj, pred_mat_list, pred_part_list, pred_part,  pred_mat, gt_obj, gt_mat_list, gt_part_list, gt_part, gt_mat, part2mat, model_id)
        print('########## End GCR Evaluation ##########')
        acc, value, value_all, gnd_value, gnd_value_all = self.eval_all()
        return acc, value, value_all, gnd_value, gnd_value_all
        

    # add data items
    def update(self, pred_obj, pred_mat, pred_part, pred_bboxes, pred_bboxesmat, \
            gt_obj, gt_mat, gt_part, gt_bboxes, gt_bboxesmat, part_2_mats, model_id):
        '''
        pred_obj: predicted objec category
        pred_mat: predicted list of material categories
        pred_part: predicted list of part categories
        pred_bboxes: predicted part segmetation labels for all points
        pred_bboxesmat: predicted material segmetation labels for all points
        part13: ground truth part to material mapping
        model_id: model id
        '''

        order = gt_part
        part_2_mat=part_2_mat2[0]
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0

        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        if len(pred_mat) == 0:
            pdb.set_trace()
        #         if gt_obj == pred_obj:
        #             self.correct_objs += 1
        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0

        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 236)
        part_id = np.where(area_intersection / (area_union + 1e-12) >= 0.5)[0]

        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxesmat, gt_bboxesmat, 14)
        mat_id = np.where(area_intersection / (area_union + 1e-12) >= 0.5)[0]

        # print(part13)
        for i, part in enumerate(gt_part):
            #compute matrix of value and value all
            if part in pred_part:
                correct_mat=part_2_mat[part]
                if correct_mat in pred_mat:
                    self.per_obj_parts_correct[gt_obj] += 1.0
                    value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            #compute matrix of gnd-value and gnd value all
            if part in pred_part:
                vit = True
                correct_mat = part_2_mat[part]
                if correct_mat in pred_mat:
                    if part in part_id and correct_mat in mat_id:
                        self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                        vit = False
                if vit:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0
                value_all = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value

    def intersectionAndUnion(self, output, target, K, ignore_index=0):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        #         print(output.shape)
        #         print(target.shape)
        assert (output.ndim in [1, 2, 3, 4])
        assert output.shape == target.shape
        output = output.reshape(output.size).copy()
        target = target.reshape(target.size)
        output[np.where(target == ignore_index)[0]] = ignore_index
        intersection = output[np.where(output == target)[0]]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
        area_output, _ = np.histogram(output, bins=np.arange(K + 1))
        area_target, _ = np.histogram(target, bins=np.arange(K + 1))
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target

# Eval when part predictions are assumed to be ground truth
class BboxEvalGTPart:
    def __init__(self, shape='top1'):
        super().__init__(shape)

    def update(self, pred_obj=None, pred_mat=None, pred_bboxes=None, gt_obj=None, gt_mat=None, gt_bboxes=None):
        order = gt_mat
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0
        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts (m,n,x,y)
        self.per_obj_parts_bboxes[gt_obj] += len(order)
        # if pred_obj == gt_obj:  # object shape
        #     self.correct_objs += 1
        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0
        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 14)
        cor_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        for i, gt in enumerate(gt_mat):
            if gt in pred_mat:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            if gt in pred_mat:
                if gt in cor_id:
                    self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                else:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value


# Eval when material predictions are assumed to be ground truth
class BboxEvalGTMat(BboxEval):
    def __init__(self, shape='top1'):
        super().__init__(shape)

    def update(self, pred_obj=None, pred_part=None, pred_bboxes=None, gt_obj=None, gt_part=None, gt_bboxes=None):
        order = gt_part
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0
        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        # if pred_obj == gt_obj:  # object shape
        #     self.correct_objs += 1

        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0
        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 236)
        cor_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        for i, gt in enumerate(gt_part):
            if gt in pred_part:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            if gt in pred_part:
                if gt in cor_id:
                    self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                else:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value
