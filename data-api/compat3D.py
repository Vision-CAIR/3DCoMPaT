# The following API functions are defined:
#  Compat3D       - 3DCompat api class that loads 3DCompat annotation file and prepare data structures.
# load_raw_models -  load raw shape without textures given the specified ids.
# show_raw_models -  show raw shape without textures given the specified ids.
# load_stylized_3d - load stylized 3d shapes given the specified ids.
# show_stylized_3d - show stylized 3d shapes given the specified ids.
# load_stylized_2d - load stylized 3d shapes given the specified ids.
# show_stylized_2d - show stylized 3d shapes given the specified ids.

_ALL_CLASSES = []
_ALL_PARTS = []

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


import os
import os.path as osp
import glob
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import trimesh

import torch
from torch.utils.data import Dataset
import pdb

class Compat3D:
    def __init__(self, meta_file=None, data_folder=None):
        """
        Constructor of 3DCompat helper class for reading and visualizing annotations.
        :param meta_file (str): location of meta file
        :param data_folder (str): location to the folder that hosts data.
        """

        df = pd.read_csv(osp.join(data_folder, "metadata/model.csv"))
        all_cats = list(set(df['model'].tolist()))
        all_classes = dict(zip(all_cats, range(len(all_cats))))
        id_to_cat = dict(zip(df['id'].tolist(), df['model'].tolist()))
        
        labels = []
        for key in df['id']:
            labels.append(all_classes[id_to_cat[key]])
    
        self.data_folder = data_folder
        self.shape_ids = df['id'].tolist()
        self.labels = np.array(labels).astype('int64')


    def load_raw_models(self, shape_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D unstylized models
        """
        gltf_path = os.path.join(self.data_folder, 'raw_models', shape_id + '.glb')
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

            return sample_xyz, sample_colors, sample_segment


    def load_stylized_3d(self, shape_id, style_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D stylized models
        """
        gltf_path = os.path.join(self.data_folder, 'rendered_models/', shape_id,  shape_id + '_' + style_id + '.glb')
        mesh = trimesh.load(gltf_path)
        
        
        # # TODO: 
        # 1. unable to load mesh
        # 2. load part labels, material colors and labels
        
        if not sample_point:
            return mesh
        else:
            pass


    def load_stylized_2d(self, shape_id, style_id, view_id):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               view_id  (int)     : camera view id
        :return: a 2d rendered image, class label, segmentation label
        """
        gltf_path = os.path.join(self.data_folder, 'canonical_views/', shape_id + '_' + style_id + '_' + view_id, 'Image0080.png')
        image = plt.imread(gltf_path)
        
        # # TODO: 
        # load part labels, material colors and labels
        
        return image


