import json
import bpy
import mathutils
import os
import sys
import time
from mathutils import Vector
import numpy as np
import os.path as osp
import struct
from collections import defaultdict


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def check_gpu_found():
    print("DEVICES GOING TO BE USED")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
        if 'CPU' not in d['name'] or 'rtx' in d['name'].lower or 'v100' in d['name'].lower or 'gtx' in d['name'].lower:
            return True

    return False


def parse_style(style):
    ret = {'style_index': style['style_index'], 'part_styles': {}}

    for part_name, finish_data in style['part_styles'].items():
        ret['part_styles'][part_name] = finish_data['_id']

    return ret


def parse_json(styles_json):
    model_styles = defaultdict(list)

    for style in styles_json:
        # Get the model pid
        p_id = style['productStyleId']

        # Make the style blender friendly
        style_data = parse_style(style)

        # Append to model_styles
        model_styles[p_id].append(style_data)

    return model_styles


def select_objects(object_names):
    # De-select anything
    bpy.ops.object.select_all(action='DESELECT')

    # Select
    for n in object_names:
        bpy.ops.object.select_pattern(pattern=n)


def delete_objects(object_names):
    for mesh in bpy.data.meshes:
        if mesh.name in object_names:
            mesh.user_clear()
            bpy.data.meshes.remove(mesh)


def get_point_coordinates_stats(model_part_names):
    bbs = []
    for object_name, object in bpy.data.objects.items():
        if object_name not in model_part_names:
            continue

        # Get the object bbox in the world coordinates
        bbox_points = np.asarray(get_object_bbox(object))
        bbs.append(bbox_points)

    bbs = np.vstack(bbs)
    min_z = np.min(bbs[:, 2])
    max_z = np.max(bbs[:, 2])

    print('x min {}, x max {}'.format(np.min(bbs[:, 0]), np.max(bbs[:, 0])))
    print('y min {}, y max {}'.format(np.min(bbs[:, 1]), np.max(bbs[:, 1])))
    print('z min {}, z max {}'.format(np.sum(bbs[:, 2]), np.sum(bbs[:, 2])))

    c_x = np.min(bbs[:, 0]) + (np.max(bbs[:, 0]) - np.min(bbs[:, 0])) / 2
    c_y = np.min(bbs[:, 1]) + (np.max(bbs[:, 1]) - np.min(bbs[:, 1])) / 2
    c_z = np.min(bbs[:, 2]) + (np.max(bbs[:, 2]) - np.min(bbs[:, 2])) / 2
    bbox_center = np.asarray((c_x, c_y, c_z))

    return np.max(np.abs(bbs), 0), bbox_center, min_z, max_z


def get_object_bbox(object):
    """
    #  ________
    # |\       |\
    # |_\______|_\
    # \ |      \ |
    #  \|_______\|
    #
    # 0-3 = lower corners
    # 4-7 = upper corners
    #
    """
    # Get bounding Box
    object_BB = []
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[0][0], object.bound_box[0][1], object.bound_box[0][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[1][0], object.bound_box[1][1], object.bound_box[1][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[2][0], object.bound_box[2][1], object.bound_box[2][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[3][0], object.bound_box[3][1], object.bound_box[3][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[4][0], object.bound_box[4][1], object.bound_box[4][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[5][0], object.bound_box[5][1], object.bound_box[5][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[6][0], object.bound_box[6][1], object.bound_box[6][2])))
    object_BB.append(object.matrix_world @ mathutils.Vector(
        (object.bound_box[7][0], object.bound_box[7][1], object.bound_box[7][2])))

    return np.asarray(object_BB)


def scale_to_unit_cube(part_names):
    np.set_printoptions(suppress=True)

    # Get the biggest coordinate
    (biggest_coordinate, _, _, _) = get_point_coordinates_stats(part_names)
    biggest_coordinate = np.max(biggest_coordinate)
    # print("Before", biggest_coordinate)

    # Use it as a scale (assumeing the objects are centered
    scale_x = 1 / biggest_coordinate if biggest_coordinate > 0 else 1
    scale_y = 1 / biggest_coordinate if biggest_coordinate > 0 else 1
    scale_z = 1 / biggest_coordinate if biggest_coordinate > 0 else 1
    biggest_coordinate = np.asarray([scale_x, scale_y, scale_z])

    # print('After', biggest_coordinate)
    bpy.ops.transform.resize(value=biggest_coordinate, orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             mirror=True,
                             use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False)


def center_and_scale(part_names, all_in_z_positive=True):
    np.set_printoptions(suppress=True)

    # Get the biggest coordinate
    (_, point_mean, min_z, _) = get_point_coordinates_stats(part_names)
    #    point_mean[2] = 0
    # print("points mean", point_mean)

    bpy.ops.transform.translate(value=point_mean * -1, orient_type='GLOBAL',
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH',
                                proportional_size=1, use_proportional_connected=False,
                                use_proportional_projected=False)

    scale_to_unit_cube(part_names)

    if all_in_z_positive:
        (_, _, min_z, _) = get_point_coordinates_stats(part_names)
        bpy.ops.transform.translate(value=(0, 0, min_z * -1), orient_type='GLOBAL',
                                    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                    mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH',
                                    proportional_size=1, use_proportional_connected=False,
                                    use_proportional_projected=False)


class Shader:
    def __init__(self):
        pass

    @staticmethod
    def create_diffuse_node(diffuse_image, material_node, link_to_base=True):
        # Create the Image Texture node
        image_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")

        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Set up the location
        image_texture_node.location = principled_node.location - Vector(
            (image_texture_node.width * 3, image_texture_node.height * 2))

        # Load the image
        image_texture_node.image = diffuse_image

        # Set up the label
        image_texture_node.label = 'Diffuse'

        # link to base material if applicable
        if link_to_base:
            link_base_color = material_node.node_tree.links.new
            link_base_color(image_texture_node.outputs['Color'], principled_node.inputs['Base Color'])

        return True

    @staticmethod
    def create_normal_node(normal_image, material_node, link_to_base=True):
        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Normal Map
        normal_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")
        normal_texture_node.location = principled_node.location - Vector(
            (principled_node.width * 4, principled_node.height * 2))

        # Load the image
        normal_texture_node.image = normal_image

        # Set up the label
        normal_texture_node.label = 'Normal'

        # Create the normal map that will be connected to the principled node
        normal_map = material_node.node_tree.nodes.new("ShaderNodeNormalMap")
        normal_map.location = principled_node.location - Vector((normal_map.width * 3, normal_map.height * 2))
        normal_map.uv_map = "UVMap"

        # link to base material if applicable
        if link_to_base:
            link_normal_map = material_node.node_tree.links.new
            link_normal_map(normal_texture_node.outputs['Color'], normal_map.inputs['Color'])

            link_normal = material_node.node_tree.links.new
            link_normal(normal_map.outputs['Normal'], principled_node.inputs['Normal'])
        return True

    @staticmethod
    def create_clearcoat_normal_node(normal_image, material_node, link_to_base=True):
        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Normal Map
        normal_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")
        normal_texture_node.location = principled_node.location - Vector(
            (principled_node.width * 4, principled_node.height * 2))

        # Load the image
        normal_texture_node.image = normal_image

        # Set up the label
        normal_texture_node.label = 'Normal'

        # Create the normal map that will be connected to the principled node
        normal_map = material_node.node_tree.nodes.new("ShaderNodeNormalMap")
        normal_map.location = principled_node.location - Vector((normal_map.width * 3, normal_map.height * 2))
        normal_map.uv_map = "UVMap"

        # link to base material if applicable
        print(principled_node.inputs)
        if link_to_base:
            link_normal_map = material_node.node_tree.links.new
            link_normal_map(normal_texture_node.outputs['Color'], normal_map.inputs['Color'])

            link_normal = material_node.node_tree.links.new
            link_normal(normal_map.outputs['Normal'], principled_node.inputs['Clearcoat Normal'])
        return True

    @staticmethod
    def create_specular_node(specular_image, material_node, link_to_base=True):
        # Create the Image Texture node
        image_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")

        # Make it non color
        #        image_texture_node.colorspace_settings.name = 'Non-Color'

        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Set up the location
        image_texture_node.location = principled_node.location - Vector(
            (image_texture_node.width * 3, image_texture_node.height * 2))

        # Load the image
        image_texture_node.image = specular_image

        # Set up the label
        image_texture_node.label = 'Specular'

        # link to base material if applicable
        if link_to_base:
            link_base_color = material_node.node_tree.links.new
            link_base_color(image_texture_node.outputs['Color'], principled_node.inputs['Specular'])

        return True

    @staticmethod
    def create_metallic_node(metallic_image, material_node, link_to_base=True):
        # Create the Image Texture node
        image_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")

        # Make it non color
        #        image_texture_node.colorspace_settings.name = 'Non-Color'

        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Set up the location
        image_texture_node.location = principled_node.location - Vector(
            (image_texture_node.width * 3, image_texture_node.height * 2))

        # Load the image
        image_texture_node.image = metallic_image

        # Set up the label
        image_texture_node.label = 'Metallic'

        # link to base material if applicable
        if link_to_base:
            link_base_color = material_node.node_tree.links.new
            link_base_color(image_texture_node.outputs['Color'], principled_node.inputs['Metallic'])

        return True

    @staticmethod
    def create_roughness_node(roughness_image, material_node, link_to_base=True):
        # Create the Image Texture node
        image_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")

        # Make it non color
        #        image_texture_node.colorspace_settings.name = 'Non-Color'

        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']

        # Set up the location
        image_texture_node.location = principled_node.location - Vector(
            (image_texture_node.width * 3, image_texture_node.height * 2))

        # Load the image
        image_texture_node.image = roughness_image

        # Set up the label
        image_texture_node.label = 'Roughness'

        # link to base material if applicable
        if link_to_base:
            link_base_color = material_node.node_tree.links.new
            link_base_color(image_texture_node.outputs['Color'], principled_node.inputs['Roughness'])

        return True

    @staticmethod
    def create_metallic_roughness_node(metallic_roughness_image, material_node, link_to_base=True):
        # Get the principled node
        principled_node = material_node.node_tree.nodes['Principled BSDF']
        #
        # Create the image texture node
        #
        metallic_roughness_texture_node = material_node.node_tree.nodes.new("ShaderNodeTexImage")
        metallic_roughness_texture_node.location = principled_node.location - Vector(
            (principled_node.width * 4, principled_node.height * 2))
        # Set up the label
        metallic_roughness_texture_node.label = 'Metallic Roughness'
        # Load the image
        metallic_roughness_texture_node.image = metallic_roughness_image

        #
        # Create separate RGB
        #
        separate_rgb_node = material_node.node_tree.nodes.new("ShaderNodeSeparateRGB")
        separate_rgb_node.location = principled_node.location - Vector(
            (principled_node.width * 2, principled_node.height * 2))
        link_1 = material_node.node_tree.links.new
        link_1(metallic_roughness_texture_node.outputs['Color'], separate_rgb_node.inputs['Image'])

        #
        # Create the math node
        #
        math_node = material_node.node_tree.nodes.new("ShaderNodeMath")
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = 0
        math_node.label = "Metallic Factor"

        link_2 = material_node.node_tree.links.new
        link_2(separate_rgb_node.outputs['B'], math_node.inputs[0])

        if link_to_base:
            link_3 = material_node.node_tree.links.new
            link_3(separate_rgb_node.outputs['G'], principled_node.inputs['Roughness'])

            link_4 = material_node.node_tree.links.new
            link_4(math_node.outputs['Value'], principled_node.inputs['Metallic'])

    @staticmethod
    def apply_material(obj_name, materials_data):
        for obj_n, obj in bpy.data.objects.items():
            if obj_n != obj_name:
                continue

            if obj_name not in bpy.data.materials.keys():
                # Unlink un wanted materials
                while len(obj.data.materials) > 0:
                    obj.data.materials.pop()

                # Create a new Material
                mat = bpy.data.materials.new(name=obj_name)

                # Create a principled Node
                mat.use_nodes = True

                # Link this material to the object part
                obj.data.materials.append(mat)

            material_node = bpy.data.materials[obj_name]
            material_node.use_nodes = True
            principled_node = material_node.node_tree.nodes['Principled BSDF']

            if 'diffuse_image' in materials_data:
                diffuse_image = materials_data['diffuse_image']
                Shader.create_diffuse_node(diffuse_image, material_node, link_to_base=True)

            if 'normal_image' in materials_data:
                normal_image = materials_data['normal_image']
                Shader.create_normal_node(normal_image, material_node, link_to_base=True)

            if 'specular_image' in materials_data:
                normal_image = materials_data['specular_image']
                Shader.create_specular_node(normal_image, material_node, link_to_base=True)

            if 'metallic_image' in materials_data:
                normal_image = materials_data['metallic_image']
                Shader.create_metallic_node(normal_image, material_node, link_to_base=True)

            if 'roughness_image' in materials_data:
                roughness_image = materials_data['roughness_image']
                Shader.create_roughness_node(roughness_image, material_node, link_to_base=True)

            if 'clearcoat_normal_image' in materials_data:
                clearcoat_normal_image = materials_data['clearcoat_normal_image']
                Shader.create_clearcoat_normal_node(clearcoat_normal_image, material_node, link_to_base=True)

    @staticmethod
    def get_image_name(initial_file_path):
        if osp.isfile(initial_file_path):
            print(initial_file_path)
            return initial_file_path

        initial_file_path = initial_file_path.replace('png', 'jpg')
        if osp.isfile(initial_file_path):
            return initial_file_path

        initial_file_path = initial_file_path.replace('jpg', 'JPG')
        if osp.isfile(initial_file_path):
            return initial_file_path

        initial_file_path = initial_file_path.replace('JPG', 'jpeg')
        if osp.isfile(initial_file_path):
            return initial_file_path

        print("Can't find the texture with initial path", initial_file_path)
        raise ValueError

    @staticmethod
    def load_pbr_material(data_path, m_id):
        material_path = osp.join(data_path, 'materials', m_id, 'maps')
        res = {}

        try:
            #            diffuse_path = osp.join(material_path, material_name + '-base_color.png')
            diffuse_path = Shader.get_image_name(osp.join(material_path, 'diffuse.png'))
            #            print(diffuse_path)
            res['diffuse_image'] = bpy.data.images.load(diffuse_path)

            #            normal_path = osp.join(material_path, material_name + '-normal.png')
            normal_path = Shader.get_image_name(osp.join(material_path, 'normal.png'))
            #            print(normal_path)
            res['normal_image'] = bpy.data.images.load(normal_path)

            #            specular_path = osp.join(material_path, material_name + '-specular.png')
            #            if osp.isfile(specular_path):
            #                res['specular_image'] = bpy.data.images.load(specular_path)

            #            metallic_path = osp.join(material_path, material_name + '-metallic.png')
            metallic_path = Shader.get_image_name(osp.join(material_path, 'metalness.png'))
            #            print(metallic_path)
            res['metallic_image'] = bpy.data.images.load(metallic_path)

            #            roughness_path = osp.join(material_path, material_name + '-roughness.png')
            roughness_path = Shader.get_image_name(osp.join(material_path, 'roughness.png'))
            #            print(roughness_path)
            res['roughness_image'] = bpy.data.images.load(roughness_path)

        except:
            print("ERROR IN LOADING TEXTURES")
            return {}

        return res

    @staticmethod
    def load_exr_material(material_path ,material_cal,material_name):
#        material_path ,aterial_name,material_cal= m
        res = {}

#        try:
        diffuse_path = osp.join(material_path, material_cal, material_name, 'map_diffuse.exr')
        if osp.isfile(diffuse_path):
            res['diffuse_image'] = bpy.data.images.load(diffuse_path)
        print(diffuse_path)
        normal_path = osp.join(material_path, material_cal, material_name, 'map_normal.exr')
        if osp.isfile(normal_path):
            res['normal_image'] = bpy.data.images.load(normal_path)
        print(normal_path)
        specular_path = osp.join(material_path, material_cal, material_name, 'map_specular.exr')
        if osp.isfile(specular_path):
            res['specular_image'] = bpy.data.images.load(specular_path)

        # metallic_path = osp.join(material_path, material_cal, material_name, '-metallic.png')
        # if osp.isfile(metallic_path):
        #     res['metallic_image'] = bpy.data.images.load(metallic_path)
        print(specular_path)
        roughness_path = osp.join(material_path, material_cal, material_name, 'map_roughness.exr')
        if osp.isfile(roughness_path):
            res['roughness_image'] = bpy.data.images.load(roughness_path)
        print(roughness_path)
#            clearcoat_normal_path = osp.join(material_path, material_cal, material_name, '-clearcoat_normal.png')
#            if osp.isfile(clearcoat_normal_path):
#                res['clearcoat_normal_image'] = bpy.data.images.load(clearcoat_normal_path)
        print(res)
#        except:
#            print()
#            return {}

        return res


    def load_vray_material(m):
        material_name,material_cal, material_path = m
        res = {}

        try:
            diffuse_path = osp.join(material_path, material_cal, material_name, "output.RGB_color.0000.exr")
            if osp.isfile(diffuse_path):
                res['diffuse_image'] = bpy.data.images.load(diffuse_path)

            normal_path = osp.join(material_path, material_cal, material_name, 'map_normal.exr')
            if osp.isfile(normal_path):
                res['normal_image'] = bpy.data.images.load(normal_path)

            specular_path = osp.join(material_path, material_cal, material_name, 'map_specular.exr')
            if osp.isfile(specular_path):
                res['specular_image'] = bpy.data.images.load(specular_path)

            # metallic_path = osp.join(material_path, material_cal, material_name, '-metallic.png')
            # if osp.isfile(metallic_path):
            #     res['metallic_image'] = bpy.data.images.load(metallic_path)

            roughness_path = osp.join(material_path, material_cal, material_name, 'map_roughness.exr')
            if osp.isfile(roughness_path):
                res['roughness_image'] = bpy.data.images.load(roughness_path)

            clearcoat_normal_path = osp.join(material_path, material_cal, material_name, '-clearcoat_normal.png')
            if osp.isfile(clearcoat_normal_path):
                res['clearcoat_normal_image'] = bpy.data.images.load(clearcoat_normal_path)

        except:
            return {}

        return res

    @staticmethod
    def load_ue4_material(m):
        material_name, material_path = m
        res = {}

        try:
            diffuse_path = osp.join(material_path, material_name + '-base_color.png')
            if osp.isfile(diffuse_path):
                res['diffuse_image'] = bpy.data.images.load(diffuse_path)

            normal_path = osp.join(material_path, material_name + '-normal.png')
            if osp.isfile(normal_path):
                res['normal_image'] = bpy.data.images.load(normal_path)

            specular_path = osp.join(material_path, material_name + '-specular.png')
            if osp.isfile(specular_path):
                res['specular_image'] = bpy.data.images.load(specular_path)

            metallic_path = osp.join(material_path, material_name + '-metallic.png')
            if osp.isfile(metallic_path):
                res['metallic_image'] = bpy.data.images.load(metallic_path)

            roughness_path = osp.join(material_path, material_name + '-roughness.png')
            if osp.isfile(roughness_path):
                res['roughness_image'] = bpy.data.images.load(roughness_path)

            clearcoat_normal_path = osp.join(material_path, material_name + '-clearcoat_normal.png')
            if osp.isfile(clearcoat_normal_path):
                res['clearcoat_normal_image'] = bpy.data.images.load(clearcoat_normal_path)

        except:
            return {}

        return res

    @staticmethod
    def load_material(data_path, m,n, material_type):
        if material_type == 'pbr':
            ret = Shader.load_pbr_material(data_path, m)
        elif material_type == 'exr':
#            print('exr')
            ret = Shader.load_exr_material(data_path,m,n)
        else:
            ret = Shader.load_ue4_material(m)

        return ret


class BlenderIO():
    def __init__(self):
        pass

    @staticmethod
    def load_model(data_path, model_id):
        # Get the model path and extension
        model_path = osp.join(data_path, 'processed_models', model_id + '.glb')
        print(model_path)

        bpy.ops.import_scene.gltf(filepath=model_path, filter_glob="*.glb;*.gltf", files=[], loglevel=0,
                                  import_pack_images=True, import_shading='NORMALS', bone_heuristic='TEMPERANCE',
                                  guess_original_bind_pose=True)


class MassRenderer():
    def __init__(self,
                 data_path='../',
                 use_eevee=False,
                 loaded_json_file_name=None,
                 last_rendered_style_index=-1,
                 job_id=-1):
        self.data_path = data_path
        self.loaded_finishes = {}
        self.stats = {
            'last_rendered_style_index': last_rendered_style_index,
            'loaded_json_file_name': loaded_json_file_name,
            'job_id': job_id
        }
        self.use_eevee_renderer = use_eevee

        # read the styles json file
        with open(osp.join(data_path, 'style_jsons', loaded_json_file_name)) as fin:
            self.styles = json.load(fin)

        # Check Resume
        if last_rendered_style_index != -1:
            # Remove the already rendered stuff
            tmp = []

            for style in self.styles:
                if style['style_index'] <= last_rendered_style_index:
                    continue
                tmp.append(style)

            self.styles = tmp

        # Parse the json for easier rendering
        self.styles = parse_json(self.styles)

    @staticmethod
    def set_engine(type='CYCLES', use_gpu=False):
        bpy.context.scene.render.engine = type
        if use_gpu:
            bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.shading_system = True
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080

    def setup_renderer(self):
        renderer = 'CYCLES'

        if self.use_eevee_renderer:
            renderer = 'BLENDER_EEVEE'
        else:
            self.set_engine(renderer, check_gpu_found())

    def render(self, model_id, style_index):
        scene = bpy.context.scene

        t = time.time()
        for ob in scene.objects:
            if ob.type == 'CAMERA':
                bpy.context.scene.camera = ob

                # Get the camera index
                id = int(ob.name.split('_')[1])

                # Create output file if not created before
                create_dir(os.path.join(self.data_path, 'rendered_styles'))

                # print('Setting camera %s' % ob.name)
                file = os.path.join(self.data_path, 'rendered_styles',
                                    str(model_id) + '_' + str(style_index) + '_' + str(id))
                bpy.ops.view3d.camera_to_view_selected()
                bpy.context.scene.render.filepath = file
                bpy.ops.render.render(write_still=True)
                break

        print("Time taken to render 4 views", time.time() - t)

    def run(self):
        # Get main environment object names
        env_object_names = list(bpy.data.objects.keys())

        # Loop over the styles for the same model
        for model_id, model_styles in self.styles.items():
            # Load the model
            BlenderIO.load_model(self.data_path, model_id)

            # Get model part names
            object_names = list(bpy.data.objects.keys())
            part_names = []

            for n in object_names:
                if n not in env_object_names:
                    part_names.append(n)

            # Select model
            select_objects(part_names)

            # Center and scale the model
            center_and_scale(part_names=part_names)

            # Loop over styles and apply materials
            style_index = -2
            for model_style in model_styles:
                part_styles = model_style['part_styles']
                style_index = model_style['style_index']
                print(part_styles)

                # Load all material textures if not loaded before
                for finish_id in list(part_styles.values()):
                    if finish_id not in self.loaded_finishes:
                        print(finish_id)
                        self.data_path="/Users/liy0r/Downloads/data/aittala-beckmann"
#                        self.finish_id=
                        materials_cal="fabric"
                        materials_name="cse_chair_old"
                        self.loaded_finishes[0] = Shader.load_material(self.data_path, materials_cal, materials_name,'exr')
                        print(self.loaded_finishes[0])

                # Begin styling
                for part_name in part_names:
#                    st = self.loaded_finishes[part_styles[part_name]]
                    st = self.loaded_finishes[0]
                    print("Applying style {} to part name {} ".format(st, part_name))
                    Shader.apply_material(part_name, st)

                # Select the cube
                select_objects(['Cube'])

                # Render the views
                self.render(model_id, style_index)

            # Delete all model parts (objects)
            delete_objects(part_names)

            # Update the stats
            create_dir(osp.join(self.data_path, 'stats'))
            self.stats['last_rendered_style_index'] = style_index
            with open(osp.join(self.data_path, 'stats', str(self.stats['job_id']) + '.txt'), 'w') as fout:
                json.dump(self.stats, fout)


if __name__ == '__main__':
    #    argv = sys.argv
    #    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    #    argv = ['C:\\Users\\Windows\\Desktop\\styleNet\\data\\', False, 'C:\\Users\\Windows\\Desktop\\styleNet\\data\\style_jsons\\json_0_per_product.json', -1]
    argv = ['/Users/liy0r/Downloads/data/', False, '/Users/liy0r/Downloads/data/package.json', -1]

    print("Arguments", argv)  # --> ['example', 'args', '123']

    # Create the renderer
    renderer = MassRenderer(data_path=argv[0], use_eevee=argv[1], loaded_json_file_name=argv[2],
                            last_rendered_style_index=argv[3])

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    renderer.run()
