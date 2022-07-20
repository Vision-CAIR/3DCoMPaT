# WORKING_DIR="/ibex/scratch/liy0r/processed_models_v5"
WORKING_DIR="/var/remote/lustre/project/k1546/ujjwal/data/model_render_v0"
import numpy as np

from glob import glob
import h5py
import os
import trimesh
parts = ['arm', 'armrest', 'back', 'back_horizontal_bar', 'back_panel', 'back_vertical_bar', 'backrest', 'bag_body',
         'base', 'base1', 'body', 'bottom_panel', 'bulb', 'bush', 'cabinet', 'caster', 'channel', 'container',
         'containing_things', 'cushion', 'design', 'door', 'doorr', 'drawer', 'drawerr', 'foot_base', 'footrest',
         'glass', 'handle', 'harp', 'head', 'headboard', 'headrest', 'keyboard_tray', 'knob', 'lamp_surrounding_frame',
         'leg', 'legs', 'leveller', 'lever', 'lid', 'mattress', 'mechanism', 'neck', 'pillow', 'plug', 'pocket', 'pole',
         'rod', 'seat', 'seat_cushion', 'shade_cloth', 'shelf', 'socket', 'stand', 'stretcher', 'support', 'supports',
         'tabletop_frame', 'throw_pillow', 'top', 'top_panel', 'vertical_divider_panel', 'vertical_side_panel',
         'vertical_side_panel2', 'wall_mount', 'wheel', 'wire', 'none']
def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc

data_dir = glob(os.path.join(WORKING_DIR, "*",'*.glb'))
asciiList = [(n.split('/')[-1]).encode("ascii", "ignore") for n in data_dir]
f = h5py.File("style_compat.hdf5", mode='w')
xyz_shape = (173650, 5000, 3)
color_shape = (173650,5000, 4)
seg_shape = (173650,5000)
id_shape =(173650,)

f.create_dataset('pc', shape=xyz_shape, compression='gzip', chunks=True)
f.create_dataset('color',  shape=color_shape, compression='gzip', chunks=True)
f.create_dataset('seg', shape=seg_shape, compression='gzip', chunks=True)
f.create_dataset('class' ,data=asciiList, shape=id_shape, compression='gzip', chunks=True)
# the ".create_dataset" object is like a dictionary, the "train_labels" is the key.
N_POINTS_PER_PART = 5000
sample_xyzs = []
sample_colorss = []
sample_segments = []
valid_ids = []
errors_id = []
errors_part = []
errors_id1 = []
errors_part1 = []
for i, model_id in enumerate(data_dir):
    if i % 1000 == 0:
        print('Test data: {}/{}'.format(i, len(data_dir)))
    try:
        mesh = trimesh.load(model_id)
    except:
        print("Error in reading")
        continue
    if len(mesh.geometry.items()) <= 1:
        print("Error files{}: ".format(model_id))
        continue
    colors = []
    v = []
    segment = []
    for g_name, g_mesh in mesh.geometry.items():
        try:
            # new_part_name = self.new_part_names[model_id][g_name]
            g_name = g_name.lower()
            if g_name in parts:
                part_name = g_name
            elif g_name.split('.')[0] in parts:
                #                 errors_id.append(model_id)
                #                 errors_part.append(g_name)d
                part_name = g_name.split('.')[0]
            elif g_name.split('_')[0] in parts:
                #                 errors_id1.append(model_id)
                #                 errors_part1.append(g_name)
                part_name = g_name.split('_')[0]
            else:
                part_name = 'none'

            v.append(g_mesh)
            try:
                co = trimesh.visual.color.vertex_to_face_color(g_mesh.visual.to_color().vertex_colors, g_mesh.faces)
            except:
                co = np.random.randint(0, 255, (g_mesh.faces.shape[0], 4))
            colors.append(co)
            #             if co.shape[0]<g_mesh.faces.shape[0]:
            #                 colors.append(np.random.randint(0, 255, g_mesh.faces.shape))
            #             else:
            #                 colors.append(mesh.geometry[g_name].visual.to_color().vertex_colors)
            #             print(co.shape)
            segment.append(np.full(g_mesh.faces.shape[0], parts.index(part_name)))
        except:
            print("load model fails:", model_id)
            print(g_name)
    combined = trimesh.util.concatenate(v)
    #     print(model_id)
    sample_xyz, sample_id = trimesh.sample.sample_surface(combined, count=5000)
    sample_xyz = pc_normalize(sample_xyz)

    sample_colors = np.concatenate(colors)[sample_id]
    sample_segment = np.concatenate(segment)[sample_id]

    f['pc'][i, ...] = sample_xyz
    f['color'][i, ...] = sample_colors
    f['seg'][i, ...] = sample_segment
#     f['class'][i]=(model_id.split('/')[-1]).encode("ascii", "ignore")
#     sample_xyzs.append(sample_xyz)
#     sample_colorss.append(sample_colors)
#     sample_segments.append(sample_segment)
#     valid_ids.append(model_id)
f.close()