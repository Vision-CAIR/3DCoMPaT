from collections import defaultdict
import pdb
from PIL import Image
import numpy as np
import cv2


class BboxEval:
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

    def update(self, pred_obj, pred_mat, pred_part, pred_bboxes, gt_obj, gt_mat, gt_part, gt_bboxes):
        # print(gt_obj, pred_obj)
        order = gt_part

        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0

        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts (m,n,x,y)
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        if len(pred_mat) == 0:
            pdb.set_trace()

        if pred_obj == gt_obj:  # object shape
            self.correct_objs += 1
        value_all_bbox = 1.0
        value_all = 1.0

        value = []

        def get_corresponding_box(pred, gt):
            gt = list(gt)
            return gt.index(pred)

        for i, gt in enumerate(zip(gt_part, gt_mat)):
            if gt in zip(pred_part, pred_mat):
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)

            if gt in zip(pred_part, pred_mat):
                j = get_corresponding_box(gt, zip(pred_part, pred_mat))
                if (self.bb_intersection_over_union(pred_bboxes[i], gt_bboxes[j])):
                    self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                else:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value

    def bb_intersection_over_union(self, boxA, boxB):
        if boxA is None and boxB is None:
            return True
        if boxA is None or boxB is None:
            return False
        # boxB = [b / 2.0 for b in boxB]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 0.5:
            return True
        else:
            return False