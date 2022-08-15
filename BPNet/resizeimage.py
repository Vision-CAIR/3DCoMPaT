import sys

from PIL import Image
from os import listdir
from os.path import isfile, join
import os.path as osp
import os
from glob import glob


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


if __name__ == '__main__':
    # Get from the arguments the directory path and output path

    # label_ads = os.path.join("/data/dataset/seg_maps_v0/", model_id + "_" + view_id, "segmentation0106.png")
    # result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    # input_path = "/ibex/scratch/liy0r/seg_maps_v0/"
    input_path = "/ibex/scratch/liy0r/depth_maps_v0/"
    # output_path = "/ibex/scratch/liy0r/cvpr/seg_maps_v4/"
    output_path = "/ibex/scratch/liy0r/cvpr/depth_maps_v4/"
    # file = sys.argv[3]
    # start = int(sys.argv[4])
    # end = int(sys.argv[5])
    create_dir(output_path)
    arr = [x for x in os.listdir(input_path)]

    print("input path", input_path)
    print("output path", output_path)

    # result = [y for x in os.walk(input_path) for y in glob(os.path.join(x[0], '*.jpg'))]
    # Read the files list
    # with open(file) as fin:
    #     onlyfiles = [f.strip() for f in fin.readlines()][:]

    create_dir(output_path)

    failed_images = []
    previously_done = 0
    ignored = 0

    # Loop over the files
    for i in arr:
        # only if it ends in .png
        # f = os.path.join(input_path, i, "segmentation0106.png")
        f = os.path.join(input_path, i, "depth0106.png")
        # print(f)
        # f=input_path+i
        if os.path.isfile(f):  # f[-4:].lower() == '.png':
            # Ignore previously converteed image
            # output_img = f.split("/")[-1][:-4]
            if os.path.isfile(osp.join(output_path, i + '.png')):
                previously_done += 1
                continue

            # print(f)

            try:
                # Read the image
                image = Image.open(osp.join(f))
                # image.show()
                # print(image.size)

                # Crop the image the internal 1064x1064 part
                # box = (414, 8, 1506, 1072)
                box = (428, 8, 1492, 1072)
                cropped_image = image.crop(box)

                # print(cropped_image.size)
                # cropped_image.show()

                resize_image = cropped_image.resize((400, 400), Image.NEAREST)

                resize_image.save(osp.join(output_path, i + '.png'))
            except:
                ignored += 1
                print("{} failed".format(f))
                failed_images.append(f)

    print("In total {} images are previously processd".format(previously_done))
    print("In total {} images are failed".format(ignored))
