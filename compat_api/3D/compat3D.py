""""
Dataloaders for the 3D 3DCoMPaT tasks.
"""
import glob
import json
import logging
import os
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import trimesh

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


def map_meshes(obj, part_to_idx):
    """
    Baking each mesh in the model and mapping them from part names. 
    """
    def clean_part(part):
        """
        Removing the hexadecimal code added to mesh groups by trimesh.
        """
        def is_hex(s):
            try:
                int(s, 16)
                return True
            except ValueError:
                return False
        part = part.split('_')
        return "_".join([p for p in part if not is_hex(p)])

    # Generating mesh map
    mesh_map = defaultdict(list)
    for mesh_name, part_name in obj.graph.geometry_nodes.items():
        # Fetching part transform
        part_name = part_name[0]
        part_transform, _ = obj.graph[part_name]

        # Cleaning trimesh identifiers
        if part_name not in part_to_idx:
            part_name = clean_part(part_name)
        part_id = part_to_idx[part_name]

        # Fetching mesh and applying transform
        part_mesh = obj.geometry[mesh_name]
        part_mesh.apply_transform(part_transform)

        # Simplyfing PBR materials
        part_mesh.visual.material = part_mesh.visual.material.to_simple()

        mesh_map[part_id] += [part_mesh]

    # Concatenating duplicate nodes
    for part_id, mesh_list in mesh_map.items():
        if len(mesh_list) == 1:
            mesh_map[part_id] = mesh_list[0]
            continue
        # Fetching material node with PIL image
        for mesh in mesh_list:
            if mesh.visual.material.image:
                break

        new_mesh = trimesh.util.concatenate(mesh_list)
        new_mesh.visual = new_mesh.visual.to_texture()
        new_mesh.visual.material.image = mesh.visual.material.image

        mesh_map[part_id] = new_mesh

    return mesh_map


def to_z_up(p):
    p[:, [2, 1]] = p[:, [1, 2]]
    p[:,1] *= -1
    return p

def to_numpy(mat, type_n, conv_z_up=False):
    ret_mat = np.array(mat).astype(type_n)
    if conv_z_up:
        return to_z_up(ret_mat)
    return ret_mat


def sample_pointcloud_multiparts(n_points, mapped_meshes, sample_color):
    """
    Sample a pointcloud across multiple parts, without mesh fusing.
      (solves a bug in trimesh.concatenate creating visual artifacts)
    """
    # Computing the number of points to sample for each part
    mesh_areas = [mesh.area for _, mesh in mapped_meshes.items()]
    area_ratios = np.array(mesh_areas)/np.sum(mesh_areas)
    mesh_points = (area_ratios*n_points).astype("int32")
    mesh_points[-1] = n_points - np.sum(mesh_points[:-1])

    # Sampling points on each mesh
    p_xyz, p_seg, p_col = [], [], []
    for mesh_points, (part_id, part_mesh) in zip(mesh_points, mapped_meshes.items()):
        sampled = \
            trimesh.sample.sample_surface(part_mesh,
                                          count=mesh_points,
                                          sample_color=sample_color)

        p_xyz += [sampled[0]]
        p_seg += [np.full(mesh_points, part_id)]
        if sample_color:
            p_col += [sampled[2]]

    p_xyz = to_numpy(np.concatenate(p_xyz), 'float32', conv_z_up=True)
    p_seg = np.concatenate(p_seg)

    if sample_color:
        p_col = to_numpy(np.concatenate(p_col), 'float32')
        return p_xyz, p_seg, p_col
    else:
        return p_xyz, p_seg


def sample_pointcloud(in_mesh, n_points, sample_color=False, shape_only=False):
    """
    Sampling a pointcloud form a mesh map.
    """
    # Sampling shape-only point cloud
    if shape_only:
        if sample_color:
            return sample_pointcloud_multiparts(n_points,
                                                in_mesh,
                                                sample_color)[::2]

        p_xyz, _ = trimesh.sample.sample_surface(in_mesh, count=n_points)
        p_xyz = to_numpy(p_xyz, 'float32', conv_z_up=True)
        return p_xyz

    # Sampling colored pointcloud
    return sample_pointcloud_multiparts(n_points, in_mesh, sample_color)



class CompatLoader3D():
    """
    Base class for 3D dataset loaders.

    Args:
        meta_dir:    Metadata directory
        root_dir:    Base dataset URL containing data split shards
        split:       One of {train, valid}
        n_comp:      Number of compositions to use
        n_points:    Number of sampled points
        load_mesh:   Only load meshes
        shape_only:  Ignore part segments while sampling pointclouds
        seed:        Initial random seed
    """
    def __init__(self, meta_dir, root_dir, split="train", n_points=1024, load_mesh=False, shape_only=False, seed=None):
        if split not in ["train", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)

        self.n_points = n_points
        self.load_mesh = load_mesh
        self.shape_only = shape_only

        # Setting random seed
        if seed:
            np.random.seed(seed)

        # Parts index and reversed index
        all_parts = json.load(open(os.path.join(meta_dir, 'parts.json')))
        self.part_to_idx = dict(zip(all_parts, range(len(all_parts))))

        # Defining input paths
        root_dir = os.path.normpath(root_dir)

        # Read all object categories
        labels = json.load(open(os.path.join(meta_dir, 'labels.json')))

        # Indexing 3D models
        split_models = pd.read_csv(os.path.join(meta_dir, 'split.csv'))
        split_models = split_models.loc[split_models['split'] == split]

        self._list_models(root_dir, split_models, labels)

        self.index = -1

    def _get_split_list(self, split_models, shape_list):
        shape_ids = set(split_models.model_id.values) & shape_list
        shape_ids = list(shape_ids)
        shape_ids.sort()

        return shape_ids

    def _list_models(self, root_dir, split_models, labels):
        raise NotImplementedError()

    def __len__(self):
        return len(self.shape_ids)
        
    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        return self
 
    def __next__(self):
        if self.index == self.__len__() -1:
            raise StopIteration
        else:
            self.index += 1
            return self.__getitem__(self.index)


class ShapeLoader(CompatLoader3D):
    """
    Unsylized 3D shape loader.

        Args:
            meta_dir:    Metadata directory
            root_dir:    Base dataset URL containing data split shards
            split:       One of {train, valid}
            n_comp:      Number of compositions to use
            n_points:    Number of sampled points
            load_mesh:   Only load meshes
            shape_only:  Ignore part segments while sampling pointclouds
    """
    def __init__(self, meta_dir, root_dir, split="train", n_points=1024, load_mesh=False, shape_only=False):
        super().__init__(meta_dir, root_dir, split, n_points, load_mesh, shape_only)

    def __getitem__(self, index):
        """
        Get raw 3D shape given index.
        """
        shape_id = self.shape_ids[index]
        glb_f = self.zip_f.open(shape_id + ".glb", "r")
        obj = trimesh.load(glb_f, file_type=".glb",
                           force='mesh' if self.shape_only else 'scene')

        # Directly return the trimesh object
        if self.load_mesh:
            return shape_id, self.shape_labels[index], obj
        # Sample a pointcloud from the 3D shape  
        else:
            if not self.shape_only:
                obj = map_meshes(obj, self.part_to_idx)
            sample = sample_pointcloud(in_mesh=obj,
                                       n_points=self.n_points,
                                       sample_color=False,
                                       shape_only=self.shape_only)
            if self.shape_only:
                return shape_id, self.shape_labels[index], sample
            else:
                return shape_id, self.shape_labels[index], *sample

    def _list_models(self, root_dir, split_models, labels):
        """
        Indexing 3D models. 
        """
        raw_zip = os.path.join(root_dir, "raw_models.zip")
        if not os.path.exists(raw_zip):
            raise RuntimeError("Raw models zip not found: [%s]." % raw_zip)

        # Indexing the source zip
        self.zip_f = zipfile.ZipFile(raw_zip, mode="r")
        shape_list = set([f[:36] for f in self.zip_f.namelist()])

        # Fetching list of shapes
        self.shape_ids = self._get_split_list(split_models, shape_list)

        # Fetching shape labels
        labels = [labels[sid] for sid in self.shape_ids]
        labels = np.array(labels)

        self.shape_labels = labels.astype('int64')


class StylizedShapeLoader(CompatLoader3D):
    """
    Sylized 3D shape loader.

        Args:
            meta_dir:    Metadata directory
            root_dir:    Base dataset URL containing data split shards
            split:       One of {train, valid}
            n_comp:      Number of compositions to use
            n_points:    Number of sampled points
            load_mesh:   Only load meshes
            shape_only:  Ignore part segments while sampling pointclouds
    """
    def __init__(self, meta_dir, root_dir, split="train", n_points=1024, load_mesh=False, shape_only=False):
        super().__init__(meta_dir, root_dir, split, n_points, load_mesh, shape_only)

    def __getitem__(self, index):
        """
        Get raw 3D shape given index.
        """
        style_id, zip_idx, zip_file = self.file_list[index]
        shape_id = self.shape_ids[index]
        glb_f = self.zip_list[zip_idx].open(zip_file, "r")
        obj = trimesh.load(glb_f, file_type=".glb", force='scene')

        # Directly return the trimesh object
        if self.load_mesh:
            return shape_id, self.shape_labels[index], obj
        # Sample a pointcloud from the 3D shape  
        else:
            obj = map_meshes(obj, self.part_to_idx)
            sample = sample_pointcloud(in_mesh=obj,
                                       n_points=self.n_points,
                                       sample_color=True,
                                       shape_only=self.shape_only)
            return shape_id, self.shape_labels[index], *sample

    def _list_models(self, root_dir, split_models, labels):
        """
        Indexing 3D models from input zip files.
        """
        # Listing model zips
        all_zips = os.path.join(root_dir, "*.zip")
        all_zips = list(glob.glob(all_zips))
        all_zips = {os.path.basename(f)[:36]:f for f in all_zips}

        # Fetching list of shapes
        shape_list = all_zips.keys()
        model_ids  = self._get_split_list(split_models, shape_list)

        # Indexing all stylized models
        self.file_list = []
        self.zip_list  = []
        self.shape_labels = []
        self.shape_ids = []

        for shape_id in model_ids:
            # Loading zip handle
            zip_f = zipfile.ZipFile(all_zips[shape_id], mode="r")
            zip_idx = len(self.zip_list)
            self.zip_list += [zip_f]

            for zip_file in zip_f.namelist():
                style_id = zip_file.split('.')[0].split('_')[1]
                file_entry = (style_id, zip_idx, zip_file)
                self.file_list += [file_entry]

                self.shape_ids += [shape_id]
                self.shape_labels += [labels[shape_id]]

        self.shape_labels = np.array(self.shape_labels).astype('int64')
