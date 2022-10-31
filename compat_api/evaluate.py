
# Evaluation code for GCR and non-GCR tasks.

from collections import defaultdict
import numpy as np
from sklearn import metrics
import pdb

class BboxEval:
    """
    Base class for evaluating GRC task.

    Args:
        shape:    choosen from 'top1', 'top5'
    """

    def __init__(self, shape='top1'):
        self.shape = shape
        self.per_obj_occ_bboxes = defaultdict(float)  # Number of object occured for each object, used for calculate gnd-value-all
        self.per_obj_all_correct_bboxes = defaultdict(float)  # Number of object occured for each object, used for calculate gnd-value-all
        
        self.per_obj_parts_bboxes = defaultdict(float)  # Number of GT parts per object, used for calculate gnd-value
        self.per_obj_parts_correct_bboxes = defaultdict(float)  # Number of (part, mat) pair predicted correctly and grounding all of them correctly for each object, used for calculate gnd-value

        self.per_obj_occ = defaultdict(float)  # Number of object occured for each object, used for calculate value_all
        self.per_obj_all_correct = defaultdict(float)  # All predicted (part, mat) pair for given object are correct, used for calculate value_all

        self.per_obj_parts = defaultdict(float)  # Number of parts per object, used for calculate value
        self.per_obj_parts_correct = defaultdict(
            float)  # Number of (part, mat) pair predicted correctly for a given object, used for calculate value

        self.all_objs = 0.0  # Number of objects
        self.correct_objs = 0.0  # Number of objects correctly predicted

    def obj(self):  # object accuracy
        return self.correct_objs / self.all_objs

    # accuracy of predicting both part category and the material of a given part correctly.
    def value(self): 
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts:
            sum_value += float(self.per_obj_parts_correct[obj]) / float(self.per_obj_parts[obj])
            total_value += 1.0
        return sum_value / total_value

    # accuracy of predicting all the (part, material) pairs of a shape
    def value_all(self): 
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ:
            sum_value_all += float(self.per_obj_all_correct[obj]) / float(self.per_obj_occ[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    # accuracy of predicting both part category and the material of a given part as well as correctly grounding it
    def value_bbox(self):
        sum_value = 0.0
        total_value = 0.0
        for obj in self.per_obj_parts_bboxes:
            sum_value += float(self.per_obj_parts_correct_bboxes[obj]) / float(self.per_obj_parts_bboxes[obj])
            total_value += 1.0
        return sum_value / total_value

    # accuracy of predicting all the (part, material) pairs of a given shape correctly and grounding all of them correctly
    def value_all_bbox(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for obj in self.per_obj_occ_bboxes:
            sum_value_all += float(self.per_obj_all_correct_bboxes[obj]) / float(self.per_obj_occ_bboxes[obj])
            total_value_all += 1.0
        return sum_value_all / total_value_all

    # return all evaluation metrics
    def eval_all(self):
        acc, value, value_all, gnd_value, gnd_value_all = self.obj(), self.value(), self.value_all(), self.value_bbox(), self.value_all_bbox()
        return acc, value, value_all, gnd_value, gnd_value_all

    # Evaluation on all predictions and GTs
    def eval_GCR(pred_objs, pred_parts, pred_mats, gt_objs, gt_parts, gt_mats, part2mats, model_ids):
        '''
        Provide a list of predictions and GTs
        '''
        print('########## Start GRC Evaluation ##########')
        for pred_obj, pred_part, pred_mat, gt_obj, gt_part, gt_mat, part2mat, model_id in zip(pred_objs.cpu().numpy(), pred_parts.cpu().numpy(), pred_mats.cpu().numpy(), gt_objs.cpu().numpy(), gt_parts.cpu().numpy(), gt_mats.cpu().numpy(), part2mats.cpu().numpy(), model_ids.cpu().numpy()):

            pred_part_list = np.unique(pred_part)
            pred_mat_list = np.unique(pred_mat)
            gt_mat_list = np.unique(gt_part)
            gt_part_list = np.unique(gt_mat)

            self.update(pred_obj, pred_mat_list, pred_part_list, pred_part,  pred_mat, gt_obj, gt_mat_list, gt_part_list, gt_part, gt_mat, part2mat, model_id)
        print('########## End GCR Evaluation ##########')
        acc, value, value_all, gnd_value, gnd_value_all = self.eval_all()
        return acc, value, value_all, gnd_value, gnd_value_all
        

    # add data items
    def update(self, pred_obj, pred_mat, pred_part, pred_bboxes, pred_bboxesmat, \
            gt_obj, gt_mat, gt_part, gt_bboxes, gt_bboxesmat, part_2_mats, model_id):
        '''
        pred_obj: predicted objec category
        pred_mat: predicted list of material categories
        pred_part: predicted list of part categories
        pred_bboxes: predicted part segmetation labels for all points
        pred_bboxesmat: predicted material segmetation labels for all points
        part13: ground truth part to material mapping
        model_id: model id
        '''

        order = gt_part
        part_2_mat=part_2_mat2[0]
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
                correct_mat=part_2_mat[part]
                if correct_mat in pred_mat:
                    self.per_obj_parts_correct[gt_obj] += 1.0
                    value.append(1)
            else:
                value_all = 0.0
                value.append(0)
            #compute matrix of gnd-value and gnd value all
            if part in pred_part:
                vit = True
                correct_mat = part_2_mat[part]
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


class BboxEvalGTPart:
    """
    Evaluting GRC task when part predictions are assumed to be ground truth.

    Args:
        shape:    choosen from 'top1', 'top5'
    """
    def __init__(self, shape='top1'):
        super().__init__(shape)

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

class BboxEvalGTMat(BboxEval):
    """
    Evaluting GRC task when material predictions are assumed to be ground truth.

    Args:
        shape:    choosen from 'top1', 'top5'
    """
    def __init__(self, shape='top1'):
        super().__init__(shape)

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


class Eval_NonGCR:
    """
    Evaluting non-GRC tasks.

    Args:
        shape:    choosen from 'top1', 'top5'
    """
	def __init__(self):
		pass

	def eval_2D_Shape_Cls(self, y_pred, y_true):
        """
        Evaluation function for 2D shape classification

        Args:
          y_pred: a numpy array, each line contains predicted classification label.
          y_true: a numpy array, each line contains GT classification label.
        """
        assert len(y_pred) == len(y_true)

        label_values = np.unique(y_true)
        cf_mat = confusion_matrix(y_true, y_pred)

        instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
        class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
        return instance_acc, class_acc

    def eval_2D_Material_Tagging(self, y_pred, y_true):
        """
        Evaluation function for 3D shape classification

        Args:
          pred_file: a numpy array, each line contains a one-hot vector of predicted material classes.
          gt_file: a numpy array, each line contains a one-hot vector of GT material classes.
        """
        assert len(y_pred) == len(y_true)

        f1 = metrics.f1_score(y_true, y_pred)
        prec = metrics.average_precision_score(y_true, y_pred)
        return f1, prec

    def eval_2D_Material_Seg(self, pred_file, gt_file):
        """
        Evaluation function for 2D material segmentation

        Args:
          y_pred: a numpy array, each line contains predicted segmentation labels for all pixels.
          y_true: a numpy array, each line contains GT segmentation labels for all pixels.
        """
        assert len(y_pred) == len(y_true)

        f1 = metrics.f1_score(y_true, y_pred)
        prec = metrics.average_precision_score(y_true, y_pred)

        mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')

        return f1, prec, mIoU


    def eval_3D_Shape_Cls(self, y_pred, y_true):
        """
        Evaluation function for 2D shape classification

        Args:
          y_pred: a numpy array, each line contains predicted classification label.
          y_true: a numpy array, each line contains GT classification label.
        """
        assert len(y_pred) == len(y_true)
        
        label_values = np.unique(y_true)
        cf_mat = metrics.confusion_matrix(y_true, y_pred)

        instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
        class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
        class_avg_acc = np.mean(class_acc)
        return instance_acc, class_avg_acc


    def eval_3D_Part_Seg(self, y_pred, y_true):
        """
        Evaluation function for 3D shape classification

        Args:
          pred_file: a numpy array, each line contains predicted part labels for all points.
          gt_file: a numpy array, each line contains GT part labels for all points.
        """
        assert len(y_pred) == len(y_true)
        
        label_values = np.unique(y_true)
        cf_mat = metrics.confusion_matrix(y_true, y_pred)

        instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
        class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
        return instance_acc, class_acc
        
    def eval_3D_Material_Seg(self, y_pred, y_true):
        """
        Evaluation function for 2D material segmentation

        Args:
          y_pred: a numpy array, each line contains predicted segmentation labels for all points.
          y_true: a numpy array, each line contains GT segmentation labels for all points.
        """
        assert len(y_pred) == len(y_true)

        f1 = metrics.f1_score(y_true, y_pred)
        prec = metrics.average_precision_score(y_true, y_pred)
        mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')

        return f1, prec, mIoU


