import json, os
import pandas as pd

data_root = "/ibex/scratch/projects/c2090/tezuesh/"
models = open(os.path.join(data_root, "model_names.txt")).readlines()
models = [i.strip() for i in models]

styles = {i: [] for i in models}

json_path = '/lustre/project/k1546/ujjwal/data/style_jsons_v0/'
lis = os.listdir(json_path)


def check_fif_all(dic):
    for key, val in dic.items():
        if len(val) < 50:
            return False
    return True


def get_part_mat(dic):
    df = pd.DataFrame(columns=["part_name", "material_name"])
    for key, val in dic['part_styles'].items():
        df.loc[len(df)] = [key, val['name']]
    return str(dic['productStyleId']), str(dic['style_index']), df




save_dir = "/home/varshnt/tezuesh/test/tezuesh/mat_part/"
all_elements = []
fin_lis = []
for x, i in enumerate(lis):

    if check_fif_all(styles):
        break

    print(x + 1, "/", len(lis))
    lis_ = json.load(open(json_path + i))
    for j in lis_:
        try:
            mid, sid, df = get_part_mat(j)
            if (len(styles[mid]) == 50):
                continue
            else:
                styles[mid].append(sid)
                df.to_csv(save_dir + mid + '_' + sid, header=True, index=False)
        except Exception as e:
            print("error here ", x, e)
            continue
