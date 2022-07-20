#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-12-30 14:30
# @Author  : Xelawk
# @FileName: pcd_sampler.py

import numpy as np
import mathutils
import trimesh
import open3d as o3d

from .parser import MeshParser


class PcdSampler(object):
    """
    Sample the surface of a mesh, returning samples which are VERY approximately evenly spaced.
    This is accomplished by sampling and then rejecting pairs that are too close together.
    """
    def __init__(self, **kwargs):
        """
        Inputs:
            mesh_file: default None
            mesh_str: default None
            face_file: default None
            face_str: default None
            img_file: default None
            dds_file: default None
            dds_bin: default None
        """
        self.mesh = None
        try:
            self.mesh = MeshParser(**kwargs)
            self.mesh.parse_mesh()
            self.mesh.compute_uv_face_normal()
            self.mesh.create_trimesh()
            self.mesh.clear()
        except BaseException as e:
            if self.mesh:
                self.mesh.clear()
            raise e

    def sample_surface_even(self, num_points):
        pcd, face_idx = trimesh.sample.sample_surface_even(self.mesh.mesh, num_points)
        vertices = np.array(self.mesh.xyz)
        faces = self.mesh.faces
        uvs = self.mesh.uvs_faces
        texture = self.mesh.img_texture

        # get uvs corresponding to each point
        uvs_pcd = []
        for pt, idx in zip(pcd, face_idx):
            src_vers = vertices[faces[idx]].tolist()
            src_uvs = uvs[idx * 3:idx * 3 + 3]
            src_uvs = np.concatenate([src_uvs, np.ones([3, 1])], axis=1).tolist()
            pt = mathutils.Vector(pt)
            v1, v2, v3 = mathutils.Vector(src_vers[0]), mathutils.Vector(src_vers[1]), mathutils.Vector(src_vers[2])
            uv1, uv2, uv3 = mathutils.Vector(src_uvs[0]), mathutils.Vector(src_uvs[1]), mathutils.Vector(src_uvs[2])
            uv_get = list(mathutils.geometry.barycentric_transform(pt, v1, v2, v3, uv1, uv2, uv3))[0:2]
            uvs_pcd.append(uv_get)

        # get rgb feature corresponding to each point
        colors = []
        img = np.array(texture) / 255
        shape = np.array(img.shape[0:2])
        for uv in uvs_pcd:
            pix_idx = shape * [1 - uv[1], uv[0]]
            pix_idx = pix_idx.astype(int)
            colors.append(img[pix_idx[0], pix_idx[1]].tolist())

        return pcd, colors

    def show_mesh(self):
        try:
            self.mesh.open3d_show()
            self.mesh.clear()
        except BaseException as e:
            self.mesh.clear()
            raise e

    @staticmethod
    def show_points_cloud(points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], 'Points Cloud')

    @staticmethod
    def export_pcd(filename, points, colors=None):
        points = points.tolist()
        with open(filename, 'w') as tmp:
            if colors:
                for xyz, rgb in zip(points, colors):
                    content_list = xyz + rgb
                    content_list = list(map(lambda x: '{:.5f}'.format(x), content_list))
                    content_line = ','.join(content_list) + '\n'
                    tmp.write(content_line)
            else:
                for xyz in points:
                    content_list = xyz
                    content_list = list(map(lambda x: '{:.5f}'.format(x), content_list))
                    content_line = ','.join(content_list) + '\n'
                    tmp.write(content_line)
