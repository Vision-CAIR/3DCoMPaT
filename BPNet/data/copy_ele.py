import os
import shutil

lis = os.listdir("/home/varshnt/tezuesh/test/tezuesh/bb_final")
src = "/project/k1546/ujjwal/data/elevation_30_v5/"
tar = "/home/varshnt/tezuesh/test/tezuesh/elevation/"
li1 = os.listdir(src)

err = []

for x, i in enumerate(lis):
    print(x, len(lis))
    print(i)
    try:
        shutil.copytree(src + i + "/Image0080.png", tar + i +".png")
    except Exception as e:
        print(e)
        err.append(i)

