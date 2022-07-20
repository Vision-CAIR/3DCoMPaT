#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-12-30 10:28
# @Author  : Xelawk
# @FileName: parser.py

import os
from io import BytesIO

import numpy as np
import trimesh
import open3d as o3d
import uuid

from wand import image
from PIL import Image


class MeshParser(object):
    def __init__(self, **kwargs):
        """
        Input:
            mesh_file: self-defined file about mesh xyz, and uv coordinate
            face_file: self-defined file about face index
        """
        # parse kwargs
        if 'mesh_file' in kwargs:
            self.mesh_file = kwargs['mesh_file']
        else:
            self.mesh_file = None
        if 'face_file' in kwargs:
            self.face_file = kwargs['face_file']
        else:
            self.face_file = None
        if 'img_file' in kwargs:  # jpg or png
            self.img_file = kwargs['img_file']
        else:
            self.img_file = None
        if 'dds_file' in kwargs:
            self.dds_file = kwargs['dds_file']
        else:
            self.dds_file = None
        if 'mesh_str' in kwargs:
            self.mesh_str = kwargs['mesh_str']
        else:
            self.mesh_str = None
        if 'face_str' in kwargs:
            self.face_str = kwargs['face_str']
        else:
            self.face_str = None
        if 'dds_bin' in kwargs:
            self.dds_bin = kwargs['dds_bin']
        else:
            self.dds_bin = None

        # Declare vars
        self.xyz = None
        self.nxyz = None
        self.uvs = None
        self.faces = None
        self.uvs_faces = None
        self.mesh = None
        self.img_texture = None

        # Register files paths temporarily created
        self.path_tmp = dict()

    def parse_mesh(self):
        # 解析得到xyz坐标，法向n_xyz, uv坐标
        xyz_list = []
        nxyz_list = []
        uv_list = []
        if self.mesh_file:
            with open(self.mesh_file, 'r') as tmp:
                content = tmp.read().strip()
        elif self.mesh_str:
            content = self.mesh_str.strip()
        else:
            raise Exception("Mesh info not found!")
        lines = content.split('\n')
        for line in lines:
            if line:
                xyz_str, nxyz_str, uv_str = line.split(',')
                x_str, y_str, z_str = xyz_str.split(' ')
                nx_str, ny_str, nz_str = nxyz_str.split(' ')
                u_str, v_str = uv_str.split(' ')
                xyz_list.append([float(x_str), float(y_str), float(z_str)])
                nxyz_list.append([float(nx_str), float(ny_str), float(nz_str)])
                uv_list.append([float(u_str), float(v_str)])

        # 解析得到face索引
        f_list = []
        if self.face_file:
            with open(self.face_file, 'r') as tmp:
                content = tmp.read().strip()
        elif self.face_str:
            content = self.face_str.strip()
        else:
            raise Exception("Faces info not found!")
        lines = content.split('\n')
        for line in lines:
            if line:
                idx1_str, idx2_str, idx3_str = line.split(' ')
                f_list.append([int(idx1_str), int(idx2_str), int(idx3_str)])

        # convert .dds to .png obj, then read the data
        if self.img_file:
            img_texture = Image.open(self.img_file)
            self.img_texture = img_texture.convert('RGB')
            self.xyz, self.nxyz, self.uvs, self.faces = xyz_list, nxyz_list, uv_list, f_list
            return True
        elif self.dds_file:
            img = image.Image(filename=self.dds_file)
        elif self.dds_bin:
            img = image.Image(file=BytesIO(self.dds_bin))
        else:
            raise Exception("Image info not found!")

        # saving to disk then read it
        img.compression = "no"
        path_tmp = 'cache/' + str(uuid.uuid1()) + '.png'
        img.save(filename=path_tmp)
        self.path_tmp["texture_png"] = path_tmp
        img_texture = Image.open(path_tmp)
        self.img_texture = img_texture.convert('RGB')
        self.xyz, self.nxyz, self.uvs, self.faces = xyz_list, nxyz_list, uv_list, f_list
        return True

    def compute_uv_face_normal(self):
        uv_list = np.array(self.uvs)
        u_list = uv_list[:, 0]
        v_list = uv_list[:, 1]
        u_list[:] = (u_list - u_list.min()) / (u_list.max() - u_list.min())
        v_list[:] = (v_list - v_list.min()) / (v_list.max() - v_list.min())
        uv_face = []
        for idx in self.faces:
            tmp = uv_list[idx].tolist()
            uv_face.extend(tmp)
        self.uvs_faces = uv_face

    def compute_uv_face(self):
        uv_list = np.array(self.uvs)
        uv_face = []
        for idx in self.faces:
            tmp = uv_list[idx].tolist()
            uv_face.extend(tmp)
        self.uvs_faces = uv_face

    def create_trimesh(self):
        if self.xyz and self.faces:
            mesh = trimesh.Trimesh(vertices=self.xyz, faces=self.faces)
            self.mesh = mesh
            return self.mesh
        else:
            raise Exception('Please parse mesh info before creating triangle mesh!')

    def trimesh_show(self):
        if self.mesh:
            self.mesh.show()
        else:
            raise Exception('Please create a triangle mesh before showing!')

    def open3d_show(self):
        if self.xyz and self.faces and self.uvs_faces:
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(self.xyz)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector(self.uvs_faces)
            if "texture_png" not in self.path_tmp:
                path_tmp = 'cache' + str(uuid.uuid1()) + '.jpg'
                self.img_texture.save(path_tmp)
                self.path_tmp["texture_png"] = path_tmp
            mesh_o3d.textures = [o3d.io.read_image(self.path_tmp["texture_png"])]
            mesh_o3d.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(self.faces), dtype=np.int))
            o3d.visualization.draw_geometries([mesh_o3d], 'Open3D')
        else:
            raise Exception('Please parse mesh info before creating triangle mesh!')

    def clear(self):
        """
        clear all files temporarily created
        """
        if self.path_tmp:
            keys_list = list(self.path_tmp.keys())
            for key in keys_list:
                try:
                    os.remove(self.path_tmp[key])
                except Exception as e:
                    print(e)
                self.path_tmp.pop(key)
