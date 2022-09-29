"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
# from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.compatseg.compatseg_loader import CompatSeg
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'bench': [4, 6, 8, 12, 13, 16, 146, 35, 170, 172, 177, 54, 57, 63, 202, 211, 116, 119, 120], 'tray': [202, 203, 140, 111, 84, 116, 215, 56, 27, 92], 'vase': [97, 35, 133, 202, 48, 16, 49, 179, 80, 57], 'chair': [128, 3, 4, 6, 8, 9, 12, 13, 16, 146, 34, 35, 170, 172, 54, 57, 189, 202, 82, 92, 223, 97, 98, 228, 101, 114, 115, 116, 119, 120], 'trolley': [228, 9, 202, 60, 92, 16, 177, 211, 116, 213, 183, 28, 222, 63], 'sofa': [4, 6, 8, 12, 13, 16, 146, 35, 170, 172, 177, 54, 57, 63, 202, 206, 209, 82, 211, 116, 119], 'planter': [97, 9, 170, 48, 49, 80, 116, 121, 92, 223], 'ladder': [193, 197, 79, 212, 87, 156], 'shelf': [223, 6, 9, 202, 28, 60, 16, 177, 116, 213, 89, 92, 222, 63], 'car': [163, 100, 228, 230, 90, 232, 105, 170, 138, 92, 207, 217, 186, 27, 60, 157, 158], 'bbq grill': [37, 114, 51, 116, 92], 'candle holder': [227, 196, 3, 38, 136, 202, 106, 16, 179, 84], 'love seat': [35, 4, 6, 8, 202, 170, 12, 13, 172, 206, 16, 177, 146, 82, 116, 54], 'toilet': [226, 170, 171, 77, 78, 208, 30], 'faucets': [2, 40, 41, 169, 107, 25, 92, 191], 'boat': [141, 143, 152, 25, 29, 165, 168, 170, 44, 47, 55, 58, 195, 68, 199, 204, 89, 103, 233, 106, 109, 112, 122, 125, 126], 'jug': [225, 121, 132, 16, 118, 25, 92], 'fans': [131, 39, 23, 24, 122, 61], 'clock': [89, 129, 6, 137, 42, 108, 141, 174, 84, 116, 25, 58], 'dishwasher': [0, 36, 9, 60, 16, 50, 211, 183, 92], 'garbage bin': [121, 228, 5, 74, 202, 75, 142, 116, 25, 92], 'lamp': [3, 16, 149, 151, 25, 161, 34, 35, 169, 175, 57, 189, 194, 202, 224, 96, 233, 115, 119], 'sun loungers': [160, 4, 6, 104, 170, 10, 13, 116, 117, 84, 119], 'sports table': [37, 134, 15, 116, 148, 54, 52, 184, 53], 'airplane': [65, 228, 197, 230, 231, 7, 71, 229, 201, 86, 153, 60, 25], 'parasol': [166, 39, 72, 139, 92, 176, 16, 178, 210, 154, 124], 'bird house': [224, 66, 163, 164, 70, 199, 16, 144, 185], 'gazebos': [76, 145, 17, 22, 155, 188, 31], 'bags': [1, 228, 234, 11, 14, 180, 116, 150, 92], 'ottoman': [35, 4, 170, 202, 172, 16, 116, 54, 119, 57, 28, 223], 'shower': [32, 3, 136, 73, 107, 16, 147, 181, 182], 'sinks': [192, 69, 198, 107, 187, 16, 214, 59, 220, 221, 62, 191], 'basket': [121, 228, 202, 74, 25, 123, 92], 'dresser': [130, 35, 9, 202, 28, 57, 92, 16, 177, 116, 213, 119, 89, 60, 63, 222, 223], 'rug': [57, 26, 85, 6], 'skateboard': [228, 55, 216, 91, 95], 'bed': [16, 146, 18, 20, 21, 19, 35, 177, 54, 57, 63, 64, 202, 81, 99, 101, 116, 119, 127], 'cabinet': [6, 9, 16, 28, 35, 37, 43, 177, 57, 60, 63, 202, 206, 211, 213, 89, 92, 222, 223, 116, 119], 'table': [9, 16, 28, 34, 35, 37, 170, 46, 177, 54, 57, 60, 189, 63, 200, 202, 206, 211, 213, 92, 222, 223, 113, 115, 116, 119], 'stool': [35, 4, 6, 8, 170, 202, 12, 13, 16, 82, 211, 116, 114, 54, 119, 120, 57], 'coat racks': [194, 202, 16, 151, 25, 94, 57], 'curtains': [162, 67, 106, 114, 151], 'bicycle': [33, 102, 167, 44, 45, 173, 142, 83, 88, 218, 219, 93, 190, 159]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    TEST_DATASET = CompatSeg(data_root='data/compat', num_points=args.num_point, split='test', transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 43
    num_part = 235

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            # seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            seg_pred, _ = classifier(points)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, -1)
            target = target.cpu().data.numpy()

            # cur_pred_val_logits = cur_pred_val
            # cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            # for i in range(cur_batch_size):
            #     cat = seg_label_to_cat[target[i, 0]]
            #     logits = cur_pred_val_logits[i, :, :]
            #     cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            # for i in range(cur_batch_size):
            #     segp = cur_pred_val[i, :]
            #     segl = target[i, :]
            #     cat = seg_label_to_cat[segl[0]]
            #     part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            #     for l in seg_classes[cat]:
            #         if (np.sum(segl == l) == 0) and (
            #                 np.sum(segp == l) == 0):  # part is not present, no prediction as well
            #             part_ious[l - seg_classes[cat][0]] = 1.0
            #         else:
            #             part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
            #                 np.sum((segl == l) | (segp == l)))
            #     shape_ious[cat].append(np.mean(part_ious))

        # all_shape_ious = []
        # for cat in shape_ious.keys():
        #     for iou in shape_ious[cat]:
        #         all_shape_ious.append(iou)
        #     shape_ious[cat] = np.mean(shape_ious[cat])
        # mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        # for cat in sorted(shape_ious.keys()):
        #     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        # test_metrics['class_avg_iou'] = mean_shape_ious
        # test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    # log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    # log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
