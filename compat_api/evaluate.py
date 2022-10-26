""""
Evaluation tools.
"""

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def eval_3D_Shape_Cls(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls.
      gt_file: a txt file, each line contains shape_id, pred_cls.
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1]
    y_pred = preds[:,1]
    
    label_values = np.unique(y_true)
    cf_mat = confusion_matrix(y_true, y_pred)

    instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
    class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
    class_avg_acc = np.mean(class_acc)
    return instance_acc, class_avg_acc


def eval_3D_Part_Seg(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls for all points.
      gt_file: a txt file, each line contains shape_id, gt_cls for all points.
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1:]
    y_pred = preds[:,1:]
    
    label_values = np.unique(y_true)
    cf_mat = confusion_matrix(y_true, y_pred)

    instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
    class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
    return instance_acc, class_acc

def eval_2D_Shape_Cls(pred_file, gt_file):
    """
    Evaluation function for 2D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls.
      gt_file: a txt file, each line contains shape_id, pred_cls.
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1]
    y_pred = preds[:,1]

    label_values = np.unique(y_true)
    cf_mat = confusion_matrix(y_true, y_pred)

    instance_acc = sum([cf_mat[i,i] for i in range(len(label_values))])/len(y_true)
    class_acc = np.array([cf_mat[i,i]/cf_mat[i,:].sum() for i in range(len(label_values))])
    return instance_acc, class_acc

def eval_2D_Material_Tagging(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, one-hot vector of predicted material classes.
      gt_file: a txt file, each line contains shape_id, one-hot vector of gt material classes.
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1:]
    y_pred = preds[:,1:]

    f1 = metrics.f1_score(y_true, y_pred)
    prec = metrics.average_precision_score(y_true, y_pred)
    return f1, prec

def eval_2D_Material_Seg(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls for all pixels.
      gt_file: a txt file, each line contains shape_id, gt_cls for all pixels..
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1:]
    y_pred = preds[:,1:]

    f1 = metrics.f1_score(y_true, y_pred)
    prec = metrics.average_precision_score(y_true, y_pred)

    mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')

    return f1, prec, mIoU

def eval_3D_Material_Seg(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls for all points.
      gt_file: a txt file, each line contains shape_id, gt_cls for all points.
    """
    preds = np.loadtxt(pred_file)
    gts = np.loadtxt(gt_file)
    assert len(preds) == len(gts)

    y_true = gts[:,1:]
    y_pred = preds[:,1:]

    f1 = metrics.f1_score(y_true, y_pred)
    prec = metrics.average_precision_score(y_true, y_pred)
    mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')

    return f1, prec, mIoU

def eval_GCR_3D(pred_file, gt_file):
    """
    Evaluation function for 3D shape classification

    Args:
      pred_file: a txt file, each line contains shape_id, pred_cls for all points.
      gt_file: a txt file, each line contains shape_id, gt_cls for all points.
    """
    # To be updated

    return None
