# The following API functions are defined:
#  3DCompat       - 3DCompat api class that loads 3DCompat annotation file and prepare data structures.
#  load_stylized_3d - load stylized 3d shapes with the specified ids.

_ALL_CLASSES = []
_ALL_PARTS = []

import os
import os.path as osp
import glob
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

class Compat3D:
    def __init__(self, meta_file=None, data_folder=None):
        """
        Constructor of 3DCompat helper class for reading and visualizing annotations.
        :param meta_file (str): location of meta file
        :param data_folder (str): location to the folder that hosts data.
        """

        df = pd.read_csv(osp.join(data_folder, "model.csv"))
        all_cats = list(set(df['model'].tolist()))
        all_classes = dict(zip(all_cats, range(len(all_cats))))
        id_to_cat = dict(zip(df['id'].tolist(), df['model'].tolist()))
        
        labels = []
        for key in range(len(df['id'])):
            labels.append(all_classes[id_to_cat[key]])

        self.shape_ids = df['id'].tolist()
        self.labels = np.array(labels).astype('int64')


    def load_raw_models(self, shape_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D unstylized models
        """
        gltf_path = os.path.join(shape_id, '_', style_id)
        mesh = trimesh.load(gltf_path)

        if nont sample_point:
            return mesh
        else:
            v = []
            segment = []
            for g_name, g_mesh in mesh.geometry.items():
                g_name = g_name.lower()
                if g_name in classes:
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
                segment.append(np.full(g_mesh.faces.shape[0], classes[part_name]))
            combined = trimesh.util.concatenate(v)

            sample_xyz, sample_id = trimesh.sample.sample_surface(combined, count=5000)
            # sample_xyz = pc_normalize(sample_xyz)
            # If there are no style models, color info set as zero
            sample_colors = np.zeros_like(sample_xyz)
            sample_segment = np.concatenate(segment)[sample_id]

            return sample_xyz, sample_colors, sample_segment


    def show_raw_models(self, shape_id, sample_point=False):

    def load_stylized_3d(self, shape_id, style_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D stylized models
        """

    def show_stylized_3d(self, stylized_3d, sample_point=False):


    def load_stylized_2d(self, shape_id, style_id, view_id):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               view_id  (int)     : camera view id
        :return: a 2d rendered image, class label, segmentation label
        """

    def show_stylized_2d(self, stylized_2d):

    def load_3d_part_labels(self, shape_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D unstylized models with part labels
        """

    def load_3d_material_labels(self, shape_id, style_id, sample_point=False):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               sample_point  (bool)     : whether to sample points from 3D shape
        :return: a 3D stylized model with colors
        """
    def load_2d_material_labels(self, shape_id, style_id, view_id):
        """
        Get raw 3d shape given shape_id
        :param shape_id  (int)     : shape id
               style_id  (int)     : style id
               view_id  (int)     : camera view id
        :return: a 2d rendered image with colors
        """


