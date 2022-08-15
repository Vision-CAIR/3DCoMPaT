import os
import re
import pandas as pd

bb_path = "/home/varshnt/tezuesh/test/tezuesh/bounding_box/"
bb_lis = os.listdir(bb_path)

mat_part_path = "/home/varshnt/tezuesh/test/tezuesh/mat_part/"
mat_part = os.listdir(mat_part_path)

save_dir = "/home/varshnt/tezuesh/test/tezuesh/bb_final/"

def merge(df, dic):
    new_df = pd.DataFrame({"part_name":[],"material_name":[],"x_min":[] ,"y_min":[] ,"x_max":[] ,"y_max":[]})
    for key, val in dic.items():
        r = re.compile(key+"*")
        matched = list(filter(r.match, list(df['name'])))
        # print(matched)
        if len(matched):
            temp = df[df['name'] == matched[0]]
            temp = temp.iloc[0]

            # print(temp.iloc[0]['name'][0], type(temp.iloc[0]['name'][0]))
            new_df.loc[len(new_df)] = [key, val, temp["x_min"], temp["y_min"], temp["x_max"], temp["y_max"]]
    return new_df


for i, m_s in enumerate(mat_part):
    print(i, len(mat_part))
    m,s = m_s.split('_')
    r = re.compile(m+"*")
    matched = list(filter(r.match, bb_lis))
    m_p = pd.read_csv(mat_part_path+m_s)
    m_p = dict(zip(list(m_p['part_name']),list(m_p['material_name'])))
    for bb in matched:
        df = pd.read_csv(bb_path + bb)
        v = bb.split("_")[-1]
        new_df = merge(df, m_p)
        new_df.to_csv(save_dir + m+"_"+s+"_"+v, index=False)
    
