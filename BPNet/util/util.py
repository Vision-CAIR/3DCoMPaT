import os
import shutil
from os.path import join

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer
from six.moves import cPickle


def save_checkpoint(state, is_best, sav_path, filename='model_last.pth.tar'):
    filename = join(sav_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.shape == target.shape
    assert (output.dim() in [1, 2, 3, 4])
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def pickle_data(file_name, *args):
    """
    Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


from collections import defaultdict
import pdb
from PIL import Image
import numpy as np
import cv2


class Compat:
    def __init__(self):
        self.per_obj_occ_bboxes = defaultdict(float)  # Not done
        self.per_obj_all_correct_bboxes = defaultdict(float)  # Not done
        self.per_obj_parts_bboxes = defaultdict(float)  # Not done
        self.per_obj_parts_correct_bboxes = defaultdict(float)  # Not done

        self.per_obj_occ = defaultdict(float)  # How many times an object has occured in the dataset
        self.per_obj_all_correct = defaultdict(float)  # All predicted (part, mat) pair for given object are correct
        self.per_obj_parts = defaultdict(float)  # Number of objects per part
        self.per_obj_parts_correct = defaultdict(
            float)  # Number of (part, mat) pair predicted correctly for a given verb

        self.all_objs = 0.0  # Number of objects
        self.correct_objs = 0.0  # Number of objects correctly predicted

    def obj(self):  # object accuracy
        return self.correct_objs / self.all_objs

    def value_all(self):  # all predicted noun, role pair should match/exist in the ground thruth pair
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ:
            sum_value_all += float(self.per_obj_all_correct[obj]) / float(self.per_obj_occ[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    def value(self):
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts:
            sum_value += float(self.per_obj_parts_correct[obj]) / float(self.per_obj_parts[obj])
            total_value += 1.0
        return sum_value / total_value

    def value_all_bbox(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ_bboxes:
            sum_value_all += float(self.per_obj_all_correct_bboxes[obj]) / float(self.per_obj_occ_bboxes[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    def value_bbox(self):
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts_bboxes:
            sum_value += float(self.per_obj_parts_correct_bboxes[obj]) / float(self.per_obj_parts_bboxes[obj])
            total_value += 1.0
        return sum_value / total_value

    def update(self, pred_obj=None, pred_mat=None, pred_part=None, pred_bboxes=None, gt_obj=None, gt_mat=None,
               gt_part=None, gt_bboxes=None):
        order = gt_part
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0
        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts (m,n,x,y)
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        if pred_obj == gt_obj:  # object shape
            self.correct_objs += 1
        value_all_bbox = 1.0
        value_all = 1.0
        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 90)
        cor_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        def get_corresponding_box(pred, gt):
            gt = list(gt)
            return gt.index(pred)

        for i, gt in enumerate(gt_part):
            if gt in pred_part:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            if gt in pred_part:
                #                 j = get_corresponding_box(gt, pred_part)
                if gt in cor_id:
                    #                 if (self.intersectionAndUnion(pred_bboxes[i], gt_bboxes[j])):
                    self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                else:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value

    def intersectionAndUnion(self, output, target, K, ignore_index=0):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.ndim in [1, 2, 3, 4])
        assert output.shape == target.shape
        output = output.reshape(output.size).copy()
        target = target.reshape(target.size)
        output[np.where(target == ignore_index)[0]] = ignore_index
        intersection = output[np.where(output == target)[0]]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
        area_output, _ = np.histogram(output, bins=np.arange(K + 1))
        area_target, _ = np.histogram(target, bins=np.arange(K + 1))
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target


def MaxMatinPart(pred_part_logit, pred_mat_logit):
    from collections import Counter
    pred_part = pred_part_logit.detach().max(1)[1]
    pred_mat = pred_mat_logit.detach().max(1)[1]

    pred_mat_numpy = np.array(pred_mat)
    pred_part_numpy = np.array(pred_part)
    part2mat = dict()
    for i in range(len(pred_part_numpy)):
        unique_part = np.unique(pred_part_numpy[i])
        based_part = pred_part_numpy[i]
        fixed_mat = pred_mat_numpy[i].copy()
        for j in unique_part:
            position = np.where(based_part == j)
            pred_value = fixed_mat[position]
            pred_value_counter = Counter(pred_value)
            max_element = pred_value_counter.most_common(1)[0][0]
            part2mat.update({j, max_element}
                            )
            fixed_mat[position] = max_element
        pred_mat_numpy[i] = fixed_mat
    return pred_mat_numpy, part2mat

# def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
#     """
#     :param model: Pytorch Model which is nn.Module
#     :param conv:  'kaiming' or 'xavier'
#     :param batchnorm: 'normal' or 'constant'
#     :param linear: 'kaiming' or 'xavier'
#     :param lstm: 'kaiming' or 'xavier'
#     """
#     for m in model.modules():
#         if isinstance(m, (nn.modules.conv._ConvNd)):
#             if conv == 'kaiming':
#                 initer.kaiming_normal_(m.weight)
#             elif conv == 'xavier':
#                 initer.xavier_normal_(m.weight)
#             else:
#                 raise ValueError("init type of conv error.\n")
#             if m.bias is not None:
#                 initer.constant_(m.bias, 0)
#
#         elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
#             if batchnorm == 'normal':
#                 initer.normal_(m.weight, 1.0, 0.02)
#             elif batchnorm == 'constant':
#                 initer.constant_(m.weight, 1.0)
#             else:
#                 raise ValueError("init type of batchnorm error.\n")
#             initer.constant_(m.bias, 0.0)
#
#         elif isinstance(m, nn.Linear):
#             if linear == 'kaiming':
#                 initer.kaiming_normal_(m.weight)
#             elif linear == 'xavier':
#                 initer.xavier_normal_(m.weight)
#             else:
#                 raise ValueError("init type of linear error.\n")
#             if m.bias is not None:
#                 initer.constant_(m.bias, 0)
#
#         elif isinstance(m, nn.LSTM):
#             for name, param in m.named_parameters():
#                 if 'weight' in name:
#                     if lstm == 'kaiming':
#                         initer.kaiming_normal_(param)
#                     elif lstm == 'xavier':
#                         initer.xavier_normal_(param)
#                     else:
#                         raise ValueError("init type of lstm error.\n")
#                 elif 'bias' in name:
#                     initer.constant_(param, 0)
#
#
# def group_weight(weight_group, module, lr):
#     group_decay = []
#     group_no_decay = []
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.modules.conv._ConvNd):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.modules.batchnorm._BatchNorm):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#     assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
#     weight_group.append(dict(params=group_decay, lr=lr))
#     weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
#     return weight_group
#
#
# def convert_to_syncbn(model):
#     def recursive_set(cur_module, name, module):
#         if len(name.split('.')) > 1:
#             recursive_set(getattr(cur_module, name[:name.find('.')]), name[name.find('.') + 1:], module)
#         else:
#             setattr(cur_module, name, module)
#
#     from lib.sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
#     for name, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm1d):
#             recursive_set(model, name, SynchronizedBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine))
#         elif isinstance(m, nn.BatchNorm2d):
#             recursive_set(model, name, SynchronizedBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine))
#         elif isinstance(m, nn.BatchNorm3d):
#             recursive_set(model, name, SynchronizedBatchNorm3d(m.num_features, m.eps, m.momentum, m.affine))
#
#
# def colorize(gray, palette):
#     # gray: numpy array of the label and 1*3N size list palette
#     color = Image.fromarray(gray.astype(np.uint8)).convert('P')
#     color.putpalette(palette)
#     return color
