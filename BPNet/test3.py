import json
with open("/home/liy0r/3d-few-shot/download/BPNet/data/parts_annotation.json") as f:
    new_part_names = json.load(f)
part_set = set()
for i in new_part_names.values():
    for j in i.values():
        part_set.add(j)
part_map = dict()
for i, j in enumerate(sorted(list(part_set))):
    part_map[j] = i
print(part_map)