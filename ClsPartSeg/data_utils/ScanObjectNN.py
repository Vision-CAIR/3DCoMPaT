"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# scanobjectnn: bag	bin	box	cabinet	chair	desk	display	door	shelf	table	bed	pillow	sink	sofa	toilet
# selected common classes: cabinet, chair, desk?, shelf, table, bed, sink, sofa, toilet

# label_9cls = np.array([3, 4, 6, 8, 9, 10, 12, 13, 14]) # picked class in scanobjectnn
# cla_mapping_40to9 = np.array(40*[-1])
# sel = np.array([14,38, 3,8,32, 12?, 4, 33, 2, 29, 30, 35]) # picked class in modelnet
# cla_mapping_40to9[sel] = np.array([3,3, 4,4,4, 6?, 8, 9, 10, 12, 13, 14])

# cla_mapping_40to9 = np.array(43*[-1])
# sel = np.array([37, 3, *, 8, 38, 36, 31, 5, 13]) # picked class in 3DCOMPAT
# cla_mapping_40to9[sel] = label_9cls


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training', class_choices=None, mapped_labels=None):
        self.data, self.label = load_scanobjectnn_data(partition)
        
        print('number of shapes', len(self.data), np.unique(self.label))
        # pdb.set_trace()
        if class_choices is not None:
            cls_idx = []
            label_new = []
            for i, c in enumerate(class_choices):
                c_idx = np.where(self.label==c)[0]
                cls_idx.extend(c_idx)
                label_new.extend(len(c_idx)*[mapped_labels[i]])
            cls_idx = np.array(cls_idx)
            self.data = self.data[cls_idx]
            self.label = np.array(label_new)
            print('selected number of shapes', len(self.data), np.unique(self.label))
        
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
