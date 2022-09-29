"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import pdb

from pathlib import Path
from tqdm import tqdm
# from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.compatseg.compatseg_loader import CompatSeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'bench': [4, 6, 8, 12, 13, 16, 146, 35, 170, 172, 177, 54, 57, 63, 202, 211, 116, 119, 120], 'tray': [202, 203, 140, 111, 84, 116, 215, 56, 27, 92], 'vase': [97, 35, 133, 202, 48, 16, 49, 179, 80, 57], 'chair': [128, 3, 4, 6, 8, 9, 12, 13, 16, 146, 34, 35, 170, 172, 54, 57, 189, 202, 82, 92, 223, 97, 98, 228, 101, 114, 115, 116, 119, 120], 'trolley': [228, 9, 202, 60, 92, 16, 177, 211, 116, 213, 183, 28, 222, 63], 'sofa': [4, 6, 8, 12, 13, 16, 146, 35, 170, 172, 177, 54, 57, 63, 202, 206, 209, 82, 211, 116, 119], 'planter': [97, 9, 170, 48, 49, 80, 116, 121, 92, 223], 'ladder': [193, 197, 79, 212, 87, 156], 'shelf': [223, 6, 9, 202, 28, 60, 16, 177, 116, 213, 89, 92, 222, 63], 'car': [163, 100, 228, 230, 90, 232, 105, 170, 138, 92, 207, 217, 186, 27, 60, 157, 158], 'bbq grill': [37, 114, 51, 116, 92], 'candle holder': [227, 196, 3, 38, 136, 202, 106, 16, 179, 84], 'love seat': [35, 4, 6, 8, 202, 170, 12, 13, 172, 206, 16, 177, 146, 82, 116, 54], 'toilet': [226, 170, 171, 77, 78, 208, 30], 'faucets': [2, 40, 41, 169, 107, 25, 92, 191], 'boat': [141, 143, 152, 25, 29, 165, 168, 170, 44, 47, 55, 58, 195, 68, 199, 204, 89, 103, 233, 106, 109, 112, 122, 125, 126], 'jug': [225, 121, 132, 16, 118, 25, 92], 'fans': [131, 39, 23, 24, 122, 61], 'clock': [89, 129, 6, 137, 42, 108, 141, 174, 84, 116, 25, 58], 'dishwasher': [0, 36, 9, 60, 16, 50, 211, 183, 92], 'garbage bin': [121, 228, 5, 74, 202, 75, 142, 116, 25, 92], 'lamp': [3, 16, 149, 151, 25, 161, 34, 35, 169, 175, 57, 189, 194, 202, 224, 96, 233, 115, 119], 'sun loungers': [160, 4, 6, 104, 170, 10, 13, 116, 117, 84, 119], 'sports table': [37, 134, 15, 116, 148, 54, 52, 184, 53], 'airplane': [65, 228, 197, 230, 231, 7, 71, 229, 201, 86, 153, 60, 25], 'parasol': [166, 39, 72, 139, 92, 176, 16, 178, 210, 154, 124], 'bird house': [224, 66, 163, 164, 70, 199, 16, 144, 185], 'gazebos': [76, 145, 17, 22, 155, 188, 31], 'bags': [1, 228, 234, 11, 14, 180, 116, 150, 92], 'ottoman': [35, 4, 170, 202, 172, 16, 116, 54, 119, 57, 28, 223], 'shower': [32, 3, 136, 73, 107, 16, 147, 181, 182], 'sinks': [192, 69, 198, 107, 187, 16, 214, 59, 220, 221, 62, 191], 'basket': [121, 228, 202, 74, 25, 123, 92], 'dresser': [130, 35, 9, 202, 28, 57, 92, 16, 177, 116, 213, 119, 89, 60, 63, 222, 223], 'rug': [57, 26, 85, 6], 'skateboard': [228, 55, 216, 91, 95], 'bed': [16, 146, 18, 20, 21, 19, 35, 177, 54, 57, 63, 64, 202, 81, 99, 101, 116, 119, 127], 'cabinet': [6, 9, 16, 28, 35, 37, 43, 177, 57, 60, 63, 202, 206, 211, 213, 89, 92, 222, 223, 116, 119], 'table': [9, 16, 28, 34, 35, 37, 170, 46, 177, 54, 57, 60, 189, 63, 200, 202, 206, 211, 213, 92, 222, 223, 113, 115, 116, 119], 'stool': [35, 4, 6, 8, 170, 202, 12, 13, 16, 82, 211, 116, 114, 54, 119, 120, 57], 'coat racks': [194, 202, 16, 151, 25, 94, 57], 'curtains': [162, 67, 106, 114, 151], 'bicycle': [33, 102, 167, 44, 45, 173, 142, 83, 88, 218, 219, 93, 190, 159]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=1024, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    TRAIN_DATASET = CompatSeg(data_root='data/compat', num_points=args.npoint, split='train', transform=None)
    TEST_DATASET = CompatSeg(data_root='data/compat', num_points=args.npoint, split='test', transform=None)
    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 43
    num_part = 235

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    
    if args.optimizer == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    elif args.optimizer == 'SGD':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate)
        
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        # lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # pdb.set_trace()
        log_string('Learning rate:%f' % scheduler.get_lr()[0])
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()
        
        scheduler.step()
        
        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            # pdb.set_trace()
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

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

        log_string('Epoch %d test Accuracy: %f ' % (
            epoch + 1, test_metrics['accuracy']))
        if (test_metrics['accuracy'] >= best_acc):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                # 'class_avg_iou': test_metrics['class_avg_iou'],
                # 'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        # if test_metrics['class_avg_iou'] > best_class_avg_iou:
        #     best_class_avg_iou = test_metrics['class_avg_iou']
        # if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
        #     best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        # log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        # log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
