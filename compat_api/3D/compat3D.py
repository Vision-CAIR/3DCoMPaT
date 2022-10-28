""""
Dataloaders for the 3D 3DCoMPaT tasks.
"""
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
import difflib
import pdb


class CompatLoader3D(Dataset):
    """
    Base class for 3D dataset loaders.

    Args:
        meta_dir:    Metadata directory
        root_dir:    Base dataset URL containing data split shards
        split:       One of {train, valid}.
        n_comp:      Number of compositions to use
        n_points:    Number of sampled points. When n_points=0, returned original mesh.
    """
    def __init__(self, meta_dir ='./', root_dir="./data/", split="train", n_point=0, n_comp=1):
        if split not in ["train", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)

        self.root_dir = os.path.normpath(root_dir)
        self.n_comp = n_comp
        self.n_point = n_point

        # parts index and reversed index
        f = open(meta_dir + 'parts.json')
        _ALL_PARTS = json.load(f)
        self.part_to_idx = dict(zip(_ALL_PARTS, range(len(_ALL_PARTS))))

        df=pd.read_csv(meta_dir + 'part_index.csv')
        self.part_rename=dict(zip(df['orgin'].tolist(),df['new'].tolist()))

        # read all object categories
        f = open(meta_dir + 'labels.json')
        all_labels = json.load(f)

        # read splits
        df = pd.read_csv(meta_dir + '/split.csv')
        df = df.loc[df['split'] == split]
        shape_ids = df.model_id.values

        labels = []
        for sid in shape_ids:
            labels.append(all_labels[sid])
        labels = np.array(labels)

        self.shape_ids = shape_ids
        self.labels = labels.astype('int64')

    def __len__(self):
        return len(self.shape_ids)
        
    def __getitem__(self, index):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               sample_point  (bool)     : whether to sample points from 3D shape
        return: 
            if sample_point=False: a 3D unstylized mesh
            if sample_point=True, (shape_id, sample_xyz, object_cls, sample_colors, sample_segment) 
        """
        shape_id = self.shape_ids[index]
        gltf_path = os.path.join(self.root_dir, 'raw_models', shape_id + '.glb')
        mesh = trimesh.load(gltf_path)
        part_to_idx = self.part_to_idx
        part_rename = self.part_rename
        if self.n_point==0:
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
                        part_name = difflib.get_close_matches(g_name, list(part_to_idx.keys()))[0]
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

            sample_xyz = np.array(sample_xyz).astype('float32')
            sample_colors = np.array(sample_colors).astype('float32')
            sample_segment = np.array(sample_segment).astype('int32')
            return shape_id, sample_xyz, self.labels[index], sample_colors, sample_segment


class CompatLoader_stylized3D(CompatLoader3D):
    """
    Stylized 3D dataset loaders.

    Args:
        meta_dir:    Metadata directory
        root_dir:    Base dataset URL containing data split shards
        split:       One of {train, valid}.
        n_comp:      Number of compositions to use
        n_points:    Number of sampled points. When n_points=0, returned original mesh.
    """
    def __init__(self, meta_dir ='./', root_dir="./data/", split="train", n_point=0, n_comp=1):
        super().__init__(meta_dir, root_dir, split, n_point, n_comp)

    def __getitem__(self, index):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               sample_point  (bool)     : whether to sample points from 3D shape
        return: 
            if sample_point=False: a 3D stylized mesh
            if sample_point=True, (shape_id, sample_xyz, object_cls, sample_colors, sample_segment) 
        """
        shape_id = self.shape_ids[index]
        shape_id = '060dcf1e-f580-4b51-9769-4fba44152fcb'
        style_id = '1684500'
        
        # mesh = trimesh.load('rendered_models/060dcf1e-f580-4b51-9769-4fba44152fcb/060dcf1e-f580-4b51-9769-4fba44152fcb_1684500.glb')
        
        gltf_path = os.path.join(self.root_dir, 'rendered_models/', shape_id,  shape_id + '_' + style_id + '.glb')
        mesh = trimesh.load(gltf_path)
        part_to_idx = self.part_to_idx
        part_rename = self.part_rename
        
        if self.n_point==0:
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
                        part_name = difflib.get_close_matches(g_name, list(part_to_idx.keys()))[0]
                # Add the vertex
                v.append(g_mesh)
                # Add the segmentation Labels
                segment.append(np.full(g_mesh.faces.shape[0], part_to_idx[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id, sample_colors = trimesh.sample.sample_surface(combined, count=self.n_point, sample_color=True)
            # sample_xyz = pc_normalize(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]
            
            sample_xyz = np.array(sample_xyz).astype('float32')
            sample_colors = np.array(sample_colors).astype('float32')
            sample_segment = np.array(sample_segment).astype('int32')
            return shape_id, sample_xyz, self.labels[index], sample_colors, sample_segment

