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


def sample_pointcloud_multiparts(n_points, mapped_meshes,
                                 sample_color, get_normals):
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
    p_xyz, p_seg, p_col, p_norm = [], [], [], []
    for mesh_points, (part_id, part_mesh) in zip(mesh_points, mapped_meshes.items()):
        sampled = \
            trimesh.sample.sample_surface(part_mesh,
                                          count=mesh_points,
                                          sample_color=sample_color)

        p_xyz += [sampled[0]]
        p_seg += [np.full(mesh_points, part_id)]
        if sample_color:
            p_col += [sampled[2]]
        if get_normals:
            face_ids = sampled[1]
            p_norm += [part_mesh.face_normals[face_ids]]

    p_xyz = to_numpy(np.concatenate(p_xyz), 'float32', conv_z_up=True)
    p_seg = np.concatenate(p_seg)

    # Setting up return tuple
    ret = [p_xyz, p_seg]
    if sample_color:
        p_col = to_numpy(np.concatenate(p_col), 'float32')
        ret += [p_col]

    if get_normals:
        p_norm = to_numpy(np.concatenate(p_norm), 'float32', conv_z_up=True)
        ret += [p_norm]

    return ret


def sample_pointcloud(in_mesh, n_points, sample_color=False,
                      shape_only=False, get_normals=False):
    """
    Sampling a pointcloud form a mesh map.
    """
    # Sampling shape-only point cloud
    if shape_only:
        if sample_color:
            ret = sample_pointcloud_multiparts(n_points,
                                               in_mesh,
                                               sample_color,
                                               get_normals)
            return [ret[0]] + ret[2:]

        p_xyz, face_ids = trimesh.sample.sample_surface(in_mesh, count=n_points)
        p_xyz = to_numpy(p_xyz, 'float32', conv_z_up=True)

        if get_normals:
            p_norm = in_mesh.face_normals[face_ids]
            p_norm = to_numpy(np.concatenate(p_norm), 'float32', conv_z_up=True)

            return p_xyz, p_norm
        return p_xyz

    # Sampling colored pointcloud
    return sample_pointcloud_multiparts(n_points,
                                        in_mesh,
                                        sample_color,
                                        get_normals)
