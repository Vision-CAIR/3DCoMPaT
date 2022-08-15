from collections import defaultdict
import numpy as np
import pdb


class BboxEvalGTPart:
    def __init__(self, shape='top1'):
        self.shape = shape
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

        return self.correct_objs / self.all_objs  # To avoid 0

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

    def update(self, pred_obj=None, pred_mat=None, pred_bboxes=None, gt_obj=None, gt_mat=None, gt_bboxes=None):
        order = gt_mat
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0
        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts (m,n,x,y)
        self.per_obj_parts_bboxes[gt_obj] += len(order)
        # if pred_obj == gt_obj:  # object shape
        #     self.correct_objs += 1
        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0
        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 14)
        cor_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        for i, gt in enumerate(gt_mat):
            if gt in pred_mat:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            if gt in pred_mat:
                if gt in cor_id:
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


class BboxEval:
    def __init__(self, shape='top1'):
        self.shape = shape
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

    def update(self, pred_obj, pred_mat, pred_part, pred_bboxes, pred_bboxesmat, gt_obj, gt_mat, gt_part, gt_bboxes,
               gt_bboxesmat, part13, model_id):
        # print(gt_obj, pred_obj)
        order = gt_part
        part_mat=part13[0]
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0

        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        if len(pred_mat) == 0:
            pdb.set_trace()
        #         if gt_obj == pred_obj:
        #             self.correct_objs += 1
        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0

        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 236)
        part_id = np.where(area_intersection / (area_union + 1e-12) >= 0.5)[0]

        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxesmat, gt_bboxesmat, 14)
        mat_id = np.where(area_intersection / (area_union + 1e-12) >= 0.5)[0]

        # print(part13)
        for i, part in enumerate(gt_part):
            #compute matrix of value and value all
            if part in pred_part:
                correct_mat=part_mat[part]
                if correct_mat in pred_mat:
                    self.per_obj_parts_correct[gt_obj] += 1.0
                    value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            #compute matrix of gnd-valud and gnd value all
            if part in pred_part:
                vit = True
                correct_mat = part_mat[part]
                if correct_mat in pred_mat:
                    if part in part_id and correct_mat in mat_id:
                        self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                        vit = False
                if vit:
                    value_all_bbox = 0.0
            else:
                value_all_bbox = 0.0
                value_all = 0.0

        self.per_obj_all_correct_bboxes[gt_obj] += value_all_bbox
        self.per_obj_all_correct[gt_obj] += value_all

        return value

    def intersectionAndUnion(self, output, target, K, ignore_index=0):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        #         print(output.shape)
        #         print(target.shape)
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


# Eval when material is the ground truch
class BboxEvalGTMat:
    def __init__(self, shape='top1'):
        self.shape = shape
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

    def update(self, pred_obj=None, pred_part=None, pred_bboxes=None, gt_obj=None, gt_part=None, gt_bboxes=None):
        order = gt_part
        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0
        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        # if pred_obj == gt_obj:  # object shape
        #     self.correct_objs += 1

        if self.shape == 'top5':
            if gt_obj in pred_obj:  # object shape
                self.correct_objs += 1
            else:
                return
        elif self.shape == 'top1':
            if gt_obj == pred_obj[0]:  # object shape
                self.correct_objs += 1
            else:
                return
        else:
            self.correct_objs += 1

        value_all_bbox = 1.0
        value_all = 1.0
        value = []
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 236)
        cor_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        for i, gt in enumerate(gt_part):
            if gt in pred_part:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            if gt in pred_part:
                if gt in cor_id:
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
