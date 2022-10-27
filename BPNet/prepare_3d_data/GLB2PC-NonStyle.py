import os
import trimesh
import numpy as np
from tqdm import tqdm
import h5py
import difflib
import pandas as pd
from collections import defaultdict

#required files
# put data in WORKING_DIR
# model.csv, contains all model_ids 
# split.txt, contains splits
# part_index, contains (model_id, part) pairs

# output dir
meta_dir = "../data/"
WORKING_DIR = "../data/processed_models_v5/" #TODO: Change to the glb folders
# number of sampled points
N_POINTS_PER_PART = 5000

def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc


# read all model ids
models=pd.read_csv(os.path.join(meta_dir, 'model.csv'))
model_ids = defaultdict(list)
all_ids=[]

# read split file
with open(os.path.join(meta_dir, "split.txt"), "r") as f:
    for line in f:
        ids, label = line.rstrip().split(',')
        model_ids[label].append(ids)
        all_ids.append(ids)

data_paths_train = list(set(model_ids['train']))
data_paths_vaild = list(set(model_ids['valid']))
data_paths_test = list(set(model_ids['test']))

# all data
dd=data_paths_train+data_paths_vaild+data_paths_test

parts=['access_panel', 'adjuster', 'aerator', 'arm', 'armrest', 'axle',
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
cat = parts
# parts index and reversed index
classes = dict(zip(cat, range(len(cat))))
r_classes = {}
for k, v in classes.items():
    r_classes[v] = k
len(parts)


df=pd.read_csv(os.path.join(meta_dir, 'part_index.csv'))
part_index=dict(zip(df['orgin'].tolist(),df['new'].tolist()))
def save_points(split):
    sample_xyzs = []
    sample_colorss = []
    sample_segments = []
    valid_ids = []
    data_paths = list(set(model_ids[split]))
    num_data=len(data_paths)
    for i, model_id in tqdm(enumerate(data_paths)):
        gltf_path = '{}/{}.glb'.format(WORKING_DIR, model_id)
        try:
            os.stat(gltf_path)
        except:
            gltf_path = '{}/{}.gltf'.format(WORKING_DIR, model_id)
            try:
                os.stat(gltf_path)
            except:
                print("model can't found {}".format(model_id))
                continue
        try:
            mesh = trimesh.load(gltf_path)
        except:
            print("Error in reading")
            continue
        if len(mesh.geometry.items()) <= 1:
            print("Error files{}: ".format(model_id))
            continue
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
        sample_xyz = pc_normalize(sample_xyz)
        # If there are no style models, color info set as zero
        sample_colors = np.zeros_like(sample_xyz)
        sample_segment = np.concatenate(segment)[sample_id]
        sample_xyzs.append(sample_xyz)
        sample_colorss.append(sample_colors)
        sample_segments.append(sample_segment)
        valid_ids.append(model_id)
    x = np.stack(sample_xyzs)
    y = np.stack(sample_colorss)
    z = np.stack(sample_segments)
    asciiList = [(n.split('/')[-1]).encode("ascii", "ignore") for n in valid_ids]
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(len(asciiList))
    num_data=len(x)# new
    with h5py.File('{}.hdf5'.format(split), 'w') as hf:
        hf.create_dataset('pc', data=x, shape=(num_data, 5000, 3), compression='gzip', chunks=True)
        hf.create_dataset('color', data=y, shape=(num_data, 5000, 3), compression='gzip', chunks=True)
        hf.create_dataset('seg', data=z, shape=(num_data, 5000), compression='gzip', chunks=True)
        hf.create_dataset('id', data=asciiList, shape=(num_data, 1), compression='gzip', chunks=True)

save_points('train')
save_points('test')
save_points('valid')
