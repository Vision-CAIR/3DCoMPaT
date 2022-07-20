from collections import defaultdict
import pdb
import numpy as np


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
        order = gt_part

        self.all_objs += 1.0  # total number of obj
        self.per_obj_occ[gt_obj] += 1.0  # how is obj distributed in the dataset
        self.per_obj_occ_bboxes[gt_obj] += 1.0

        self.per_obj_parts[gt_obj] += len(order)  # # every object will have some parts (m,n,x,y)
        self.per_obj_parts_bboxes[gt_obj] += len(order)

        if len(pred_mat) == 0:
            pdb.set_trace()

        #         if pred_obj == gt_obj: # object shape
        #             self.correct_objs += 1
        if gt_obj in pred_obj:
            self.correct_objs += 1
        # else:
        #     return

        value_all_bbox = 1.0
        value_all = 1.0

        value = []
        flag = 1
        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 90)
        part_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        area_intersection, area_union, area_target = self.intersectionAndUnion(pred_bboxes, gt_bboxes, 90)
        mat_id = np.where(area_intersection / (area_union + 1e-12) > 0.5)[0]

        def get_corresponding_box(pred, gt):
            gt = list(gt)
            return gt.index(pred)

        for i, (part, mat) in enumerate(zip(gt_part, gt_mat)):
            #             if gt in zip(pred_part, pred_mat):
            if part in pred_part and mat in pred_mat:
                self.per_obj_parts_correct[gt_obj] += 1.0
                value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            try:
                if part in pred_part and mat in pred_mat:
                    #                 j = get_corresponding_box(gt, zip(pred_part, pred_mat))
                    if part in part_id or mat in mat_id:
                        #                 if (self.bb_intersection_over_union(pred_bboxes[i], gt_bboxes[j])):
                        self.per_obj_parts_correct_bboxes[gt_obj] += 1.0
                    else:
                        value_all_bbox = 0.0
                else:
                    value_all_bbox = 0.0
            except:
                flag = 1
                print('error')
        if flag:
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


bbox2 = BboxEval()
gtcls = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/gtcls.npy")
gtmat = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/gtmat.npy")
gtseg = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/gt.npy")
outcls = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/cls.npy")
outmat = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/mats.npy")
outseg = np.load("/ibex/scratch/liy0r/cvpr/BPNet/Exp/scannet/ibex2dmat1/result/best/pred.npy")
for a, b, c, d, e, f in zip(outseg, outmat, outcls, gtseg, gtmat, gtcls):
    #     bbox.update()
    #     print(a.shape)
    pred_part = list(set(a.reshape(a.size)))
    pred_mat = list(set(b.reshape(b.size)))
    gt_mat = list(set(e.reshape(e.size)))
    gt_part = list(set(d.reshape(d.size)))
    #     print(gt_mat)
    #     print(gt_part)
    #     print(a.ndim)
    bbox2.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
print(bbox2.obj())
print(bbox2.value())
print(bbox2.value_all())
print(bbox2.value_bbox())
print(bbox2.value_all_bbox())
print(bbox2.value_all())
