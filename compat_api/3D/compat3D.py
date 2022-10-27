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
import matplotlib.pyplot as plt
import trimesh
import json
import torch
from torch.utils.data import Dataset

import pdb


class CompatLoader3D(Dataset):
    """
    Base class for 3D dataset loaders.

    Args:
        root_dir:    Base dataset URL containing data split shards
        split:       One of {train, valid}.
        n_comp:      Number of compositions to use
        n_points:    Number of sampled points
    """
    def __init__(self, root_dir="./data/", split="train", n_point=5000, n_comp=1):
        if split not in ["train", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)

        self.root_dir = os.path.normpath(root_dir)
        self.n_comp = n_comp
        self.n_point = n_point

        # parts index and reversed index
        f = open('./metadata/parts.json')
        _ALL_PARTS = json.load(f)
        self.part_to_idx = dict(zip(_ALL_PARTS, range(len(_ALL_PARTS))))

        df=pd.read_csv('./metadata/part_index.csv')
        self.part_rename=dict(zip(df['orgin'].tolist(),df['new'].tolist()))

        # read all object categories
        f = open('./metadata/labels.json')
        all_labels = json.load(f)

        # read splits
        df = pd.read_csv('./metadata/split.csv')
        df = df.loc[df['split'] == split]
        shape_ids = df.model_id.values

        labels = []
        for sid in shape_ids:
            labels.append(all_labels[sid])
        labels = np.array(labels)

        self.shape_ids = shape_ids
        self.labels = labels.astype('int64')

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
        part_to_idx = self.part_to_idx
        part_rename = self.part_rename
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
                elif g_name in part_rename:
                    # Glb name is different from defined. We regulated the name.
                    part_name = part_rename[g_name]
                else:
                    # If there are still some incorrect one.
                    part_name = g_name.split('_')[0]
                    if part_name not in part_to_idx:
                        part_name = difflib.get_close_matches(g_name, parts)[0]
                # Add the vertex
                v.append(g_mesh)
                # Add the segmentation Labels
                segment.append(np.full(g_mesh.faces.shape[0], part_to_idx[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id = trimesh.sample.sample_surface(combined, count=self.n_point)
            # sample_xyz = pc_normalize(sample_xyz)
            # If there are no style models, color info set as zero
            sample_colors = np.zeros_like(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]

            return shape_id, sample_xyz, sample_colors, sample_segment


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
    def __init__(self, root_dir="./data/", split="train", n_point=5000, n_comp=1):
        super().__init__(root_dir, split, n_point, n_comp)

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
        part_to_idx = self.part_to_idx
        part_rename = self.part_rename
        
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
                elif g_name in part_rename:
                    # Glb name is different from defined. We regulated the name.
                    part_name = part_rename[g_name]
                else:
                    # If there are still some incorrect one.
                    part_name = g_name.split('_')[0]
                    if part_name not in part_to_idx:
                        part_name = difflib.get_close_matches(g_name, parts)[0]
                # Add the vertex
                v.append(g_mesh)
                # Add the segmentation Labels
                segment.append(np.full(g_mesh.faces.shape[0], part_to_idx[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id, sample_colors = trimesh.sample.sample_surface(combined, count=5000, sample_color=True)
            # sample_xyz = pc_normalize(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]

            return shape_id, sample_xyz, sample_colors, sample_segment

