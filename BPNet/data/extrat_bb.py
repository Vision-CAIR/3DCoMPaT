import trimesh, random, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


path_obj = "/project/k1546/ujjwal/data/processed_models_v5/"
objs = os.listdir(path_obj)
objs = [i.split('.')[0] for i in objs]

seg_file = "/project/k1546/ujjwal/data/tezuesh/seg_files.txt"
seg = open(seg_file)
seg = [i.strip() for i in seg]
seg_path = "/project/k1546/ujjwal/data/elevation_30_v5/"

save_dir = "/project/k1546/ujjwal/data/tezuesh/bounding_box/"


def get_values(f):
    glb_pa = path_obj + f + ".gltf"
    if not os.path.exists(glb_pa):
        glb_pa = path_obj + f+".glb"
    obj = trimesh.load(glb_pa)
    
    names = []
    for g_name, g_mesh in obj.geometry.items():
        names.append(g_name)
        
    pixel_vals = {}
    for i, key in enumerate(sorted(names)):
        pixel_vals[key] = 20*(i+1)
    return pixel_vals


def corr_obj_file(val):
    if val.split("_")[0] in objs:
        return val.split("_")[0]
    else:
        return None

def get_bounding_box(f):
    seg_mask = Image.open(seg_path + f + "/Segmentation0080.png")
    seg_mask = np.asarray(seg_mask)
    obj_file = corr_obj_file(f)
    if obj_file == None:
        return None
    else:
        pixel_vals = get_values(obj_file)
    
    bounding_box = {}
    for key, value in pixel_vals.items():
        a = np.where(seg_mask == value)
        if a[0].shape[0] == 0 or a[1].shape[0] == 0:
            continue
        x_max, x_min = a[0].max(), a[0].min()
        y_max, y_min = a[1].max(), a[1].min()
        # bounding_box[key] = [(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)]
        bounding_box[key] = [y_min,x_min,y_max,x_max]
    return bounding_box

count = 0
err = []
for i, seg_idx in enumerate(seg):
    print(i, len(seg))
    try:
        bb = get_bounding_box(seg_idx)
        save_file_name = seg_idx.split('_')[0] + '_' + seg_idx.split('_')[2]
        df = pd.DataFrame(columns=["name", "x_min","y_min","x_max","y_max"])
        for key, value in bb.items():
            df.loc[len(df)] = [key] + value
        df.to_csv(save_dir + save_file_name, header = True, index = False)
        count = count + 1
    except Exception as e:
        print(e, seg_idx)
        err.append((e, seg_idx))


print(err)
f = open("output.txt", "w")
for element in err:
    f.write(str(element[0]) + str(element[1]) + "\n" )
f.close()
        