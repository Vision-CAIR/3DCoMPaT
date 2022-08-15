import os
import random
import numpy as np
import logging
import argparse
from metrics.compat import BboxEval, BboxEvalGTMat, BboxEvalGTPart
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join
from metrics import iou

from MinkowskiEngine import SparseTensor, CoordsManager
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU
from tqdm import tqdm
from tool.trainstylenew import get_model

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def worker_init_fn(worker_id):
    random.seed(1463 + worker_id)
    np.random.seed(1463 + worker_id)
    torch.manual_seed(1463 + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/scannet/bpnet_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/bpnet_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    # https://github.com/Microsoft/human-pose-estimation.pytorch/issues/8
    # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/7
    # torch.backends.cudnn.enabled = False

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    # Following code is for caching dataset into memory
    if args.data_name == 'scannet_3d_mink':
        from dataset.scanNet3DClsNew import ScanNet3D, collation_fn_eval_all
        _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='test', aug=False,
                      memCacheInit=True, eval_all=True, identifier=6797)
    elif args.data_name == 'scannet_cross':
        from dataset.scanNetCrossStyleNew import ScanNetCross, collation_fn, collation_fn_eval_all
        _ = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='test', aug=False,
                         memCacheInit=True, eval_all=True, identifier=6797, val_benchmark=args.val_benchmark)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if args.data_name == 'scannet_3d_mink':
        from dataset.scanNet3DClsNew import ScanNet3D, collation_fn_eval_all
        val_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='test', aug=False,
                             memCacheInit=True, eval_all=True, identifier=6797)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                 drop_last=False, collate_fn=collation_fn_eval_all,
                                                 sampler=val_sampler)
    elif args.data_name == 'scannet_cross':
        from dataset.scanNetCrossStyleNew import ScanNetCross, collation_fn_eval_all
        val_data = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='test', aug=False,
                                memCacheInit=True, eval_all=True, identifier=6797, val_benchmark=args.val_benchmark)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                 drop_last=False, collate_fn=collation_fn_eval_all,
                                                 sampler=val_sampler)
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Test ####################### #
    if args.data_name == 'scannet_3d_mink':
        validate(model, val_loader)
    elif args.data_name == 'scannet_cross':
        # bbox = BboxEval()
        # bboxmat = BboxEvalMat()
        validate_cross(model, val_loader)


def validate(model, val_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    model.eval()
    with torch.no_grad():
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds = []
            gts = []
            for i, (coords, feat, label, inds_reverse) in enumerate(tqdm(val_loader)):
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                predictions = model(sinput)
                predictions_enlarge = predictions[inds_reverse, :]
                if args.multiprocessing_distributed:
                    dist.all_reduce(predictions_enlarge)
                preds.append(predictions_enlarge.detach_().cpu())
                gts.append(label.cpu())
            gt = torch.cat(gts)
            pred = torch.cat(preds)
            current_iou = iou.evaluate(pred.max(1)[1].numpy(), gt.numpy())
            if rep_i == 0 and main_process():
                np.save(join(args.save_folder, 'gt.npy'), gt.numpy())
            store = pred + store
            accumu_iou = iou.evaluate(store.max(1)[1].numpy(), gt.numpy())
            if main_process():
                np.save(join(args.save_folder, 'pred.npy'), store.max(1)[1].numpy())


# def validate_cross(model, val_loader, bbox, bboxmat):
#     torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
#     intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
#     union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
#     target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
#     target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
#     target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
#     acc = 0
#     total = 0
#     model.eval()
#     bbox3d = BboxEval()
#     bboxmat3d = BboxEvalMat()
#     with torch.no_grad():
#         outseg, outcls, outmat, gtcls, gtseg, gtmat, gtseg3d, outseg3d = [], [], [], [], [], [], [], []
#         pts = []
#         gtmat3d, outmat3d = [], []
#         for i, batch_data in enumerate(tqdm(val_loader)):
#             if args.data_name == 'scannet_cross':
#                 (coords, feat, label_3d, color, label_2d, link, inds_reverse, cls, mat, mat3d) = batch_data
#                 sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
#                 color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
#                 label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
#                 cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
#                 mat3d = mat3d.cuda(non_blocking=True)
#                 output_3d, output_2d, output_cls, output_mat, output_3dmat = model(sinput, color, link)
#             else:
#                 raise NotImplemented
#             # ############ 3D ############ #
#             # o3d=output_3d.detach().topk(5)[1]
#             output_3d = output_3d.detach().max(1)[1]
#             intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
#                                                                   args.ignore_label)
#             if args.multiprocessing_distributed:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
#             intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter_3d.update(intersection)
#             union_meter_3d.update(union)
#             target_meter_3d.update(target)
#             accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)
#
#             # ############ 2D ############ #
#             # print(output_2d.shape)
#             # o2d=output_2d.detach().topk(5, dim=1)[1]
#             output_2d = output_2d.detach().max(1)[1]
#             intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
#                                                                   args.ignore_label)
#             if args.multiprocessing_distributed:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
#             intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter_2d.update(intersection)
#             union_meter_2d.update(union)
#             target_meter_2d.update(target)
#             accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)
#             # ############ mat_3d ############ #
#             # o3dmat = output_3dmat.detach().topk(5)[1]
#             output_3dmat = output_3dmat.detach().max(1)[1]
#             intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
#                                                                   args.ignore_label)
#             intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter_3dmat.update(intersection)
#             union_meter_3dmat.update(union)
#             target_meter_3dmat.update(target)
#             accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)
#             # ############ mat ############ #
#             # omat=output_mat.detach().topk(5, dim=1)[1]
#             output_mat = output_mat.detach().max(1)[1]
#             intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
#                                                                   args.ignore_label)
#             if args.multiprocessing_distributed:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
#             intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter_mat.update(intersection)
#             union_meter_mat.update(union)
#             target_meter_mat.update(target)
#             accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)
#
#             # ############ cls ############ #
#             ocls = output_cls.detach().topk(5)[1]
#             output_cls = output_cls.detach().max(1)[1]
#             correct_guessed = output_cls == cls
#             cls_b_acc = torch.sum(correct_guessed.double()).item()
#             acc += cls_b_acc
#             total += output_cls.size(0)
#             print(total)
#             outseg.append(output_2d.cpu())
#             outmat.append(output_mat.cpu())
#             outcls.append(ocls.cpu())
#             gtcls.append(cls.cpu())
#             gtseg.append(label_2d.cpu())
#             gtmat.append(mat.cpu())
#             outseg3d.append(output_3d.cpu())
#             gtseg3d.append(label_3d.cpu())
#             gtmat3d.append(mat3d.cpu())
#             outmat3d.append(output_3dmat.cpu())
#             # for a, b, c, d, e, f in zip(output_2d.detach_().cpu().numpy(), output_mat.detach_().cpu().numpy(),
#             #                             output_cls.detach_().cpu().numpy(), label_2d.cpu().numpy(), mat.cpu().numpy(),
#             #                             cls.cpu().numpy()):
#             #     pred_part = list(set(a.reshape(a.size)))
#             #     pred_mat = list(set(b.reshape(b.size)))
#             #     gt_mat = list(set(e.reshape(e.size)))
#             #     gt_part = list(set(d.reshape(d.size)))
#             #     bboxmat.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
#             #     bbox.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
#             #
#             # for a, b, c, d, e, f in zip(output_3d.detach_().cpu().numpy(), output_3dmat.detach_().cpu().numpy(),
#             #                             output_cls.detach_().cpu().numpy(), label_3d.cpu().numpy(), mat3d.cpu().numpy(),
#             #                             cls.cpu().numpy()):
#             #     pred_part = list(set(a.reshape(a.size)))
#             #     pred_mat = list(set(b.reshape(b.size)))
#             #     gt_mat = list(set(e.reshape(e.size)))
#             #     gt_part = list(set(d.reshape(d.size)))
#             #     bboxmat3d.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
#             #     bbox3d.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
#
#     # offpts  =[]
#     # offset=0
#     # for i in range(pts:
#     #     offpts.append(offset)
#     #     offset+=len(i)
#
#     offoutseg3d = []
#     offset = 0
#     for i in outseg3d:
#         offoutseg3d.append(offset)
#         offset += len(i)
#
#     offoutmat3d = []
#     offset = 0
#     for i in outmat3d:
#         offoutmat3d.append(offset)
#         offset += len(i)
#     #
#     # np.save(join(args.save_folder, 'offpts.npy'), np.array(offpts)
#     #         )
#     np.save(join(args.save_folder, 'offoutseg3d.npy'), np.array(offoutseg3d)
#             )
#     np.save(join(args.save_folder, 'offoutmat3d.npy'), np.array(offoutmat3d)
#             )
#     # pts1 = torch.cat(pts)
#     outcls1 = torch.cat(outcls)
#     outseg1 = torch.cat(outseg)
#     outseg3d1 = torch.cat(outseg3d)
#     outmat1 = torch.cat(outmat)
#     gtcls1 = torch.cat(gtcls)
#     gtmat1 = torch.cat(gtmat)
#     gtseg1 = torch.cat(gtseg)
#     gtseg3d1 = torch.cat(gtseg3d)
#     outmat3d1 = torch.cat(outmat3d)
#     gtmat3d1 = torch.cat(gtmat3d)
#     # np.save(join(args.save_folder, 'pts.npy'), pts1.numpy()
#     #         )
#     np.save(join(args.save_folder, 'outcls.npy'), outcls1.numpy()
#             )
#     np.save(join(args.save_folder, 'outseg.npy'), outseg1.numpy()
#             )
#     np.save(join(args.save_folder, 'outseg3d.npy'), outseg3d1.numpy()
#             )
#     np.save(join(args.save_folder, 'outmat.npy'), outmat1.numpy()
#             )
#     # mIou_2d = iou.evaluate(store.max(1)[1].numpy(), gt.numpy())
#     np.save(join(args.save_folder, 'gtcls.npy'), gtcls1.numpy())
#     np.save(join(args.save_folder, 'gtseg.npy'), gtseg1.numpy())
#     np.save(join(args.save_folder, 'gtseg3d.npy'), gtseg3d1.numpy())
#     np.save(join(args.save_folder, 'gtmat.npy'), gtmat1.numpy())
#     np.save(join(args.save_folder, 'gtmat3d.npy'), gtmat3d1.numpy())
#     np.save(join(args.save_folder, 'outmat3d.npy'), outmat3d1.numpy())
#
#     iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
#     accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
#     mIoU_3d = np.mean(iou_class_3d)
#     mAcc_3d = np.mean(accuracy_class_3d)
#     allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)
#
#     iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
#     accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
#     mIoU_2d = np.mean(iou_class_2d)
#     mAcc_2d = np.mean(accuracy_class_2d)
#     allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
#     # acc_cls = acc / total
#
#     iou_class_3dmat = intersection_meter_3dmat.sum / (union_meter_3dmat.sum + 1e-10)
#     accuracy_class_3dmat = intersection_meter_3dmat.sum / (target_meter_3dmat.sum + 1e-10)
#     mIoU_3dmat = np.mean(iou_class_3dmat)
#     mAcc_3dmat = np.mean(accuracy_class_3dmat)
#     allAcc_3dmat = sum(intersection_meter_3dmat.sum) / (sum(target_meter_3dmat.sum) + 1e-10)
#
#     # allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
#     acc_cls = acc / total
#
#     iou_class_mat = intersection_meter_mat.sum / (union_meter_mat.sum + 1e-10)
#     accuracy_class_mat = intersection_meter_mat.sum / (target_meter_mat.sum + 1e-10)
#     mIoU_mat = np.mean(iou_class_mat)
#     mAcc_mat = np.mean(accuracy_class_mat)
#     allAcc_mat = sum(intersection_meter_mat.sum) / (sum(target_meter_mat.sum) + 1e-10)
#
#     if main_process():
#         logger.info(
#             'Val result 3d: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
#         logger.info(
#             'Val result 2d : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_2d, mAcc_2d, allAcc_2d))
#         logger.info(
#             'Val result 2dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_mat, mAcc_mat, allAcc_mat))
#         logger.info('Class ACC{:.4f}'.format(acc_cls))
#         logger.info(
#             'Val result 3dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3dmat, mAcc_3dmat, allAcc_3dmat))
#         # logger.info('Class ACC{:.4f}'.format(acc_cls))
#     return mIoU_3d, mAcc_3d, allAcc_3d, \
#            mIoU_2d, mAcc_2d, allAcc_2d


def validate_cross(model, val_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    acc = 0
    total = 0
    model.eval()
    print('start validation on cross')
    bbox2d = BboxEval()
    bboxmat2d = BboxEvalGTMat()
    bboxpart2d = BboxEvalGTPart()

    bbox3d = BboxEval()
    bboxmat3d = BboxEvalGTMat()
    bboxpart3d = BboxEvalGTPart()

    bbox2dtop5 = BboxEval(shape='top5')
    bboxmat2dtop5 = BboxEvalGTMat(shape='top5')
    bboxpart2dtop5 = BboxEvalGTPart(shape='top5')

    bbox3dtop5 = BboxEval(shape='top5')
    bboxmat3dtop5 = BboxEvalGTMat(shape='top5')
    bboxpart3dtop5 = BboxEvalGTPart(shape='top5')

    bbox2dgt = BboxEval(shape='gt')
    bboxmat2dgt = BboxEvalGTMat(shape='gt')
    bboxpart2dgt = BboxEvalGTPart(shape='gt')

    bbox3dgt = BboxEval(shape='gt')
    bboxmat3dgt = BboxEvalGTMat(shape='gt')
    bboxpart3dgt = BboxEvalGTPart(shape='gt')

    bbox2dcls = BboxEval()
    bboxmat2dcls = BboxEvalGTMat()
    bboxpart2dcls = BboxEvalGTPart()

    bbox3dcls = BboxEval()
    bboxmat3dcls = BboxEvalGTMat()
    bboxpart3dcls = BboxEvalGTPart()

    bbox2dtop5cls = BboxEval(shape='top5')
    bboxmat2dtop5cls = BboxEvalGTMat(shape='top5')
    bboxpart2dtop5cls = BboxEvalGTPart(shape='top5')

    bbox3dtop5cls = BboxEval(shape='top5')
    bboxmat3dtop5cls = BboxEvalGTMat(shape='top5')
    bboxpart3dtop5cls = BboxEvalGTPart(shape='top5')

    bbox2dgtcls = BboxEval(shape='gt')
    bboxmat2dgtcls = BboxEvalGTMat(shape='gt')
    bboxpart2dgtcls = BboxEvalGTPart(shape='gt')

    bbox3dgtcls = BboxEval(shape='gt')
    bboxmat3dgtcls = BboxEvalGTMat(shape='gt')
    bboxpart3dgtcls = BboxEvalGTPart(shape='gt')

    import pickle
    import json
    with open('data/resnet50_preds.pickle', 'rb') as handle:
        resnet_predictions = pickle.load(handle)
    with open('data/test_output_top5.json', 'rb') as handle:
        point_predictions = json.load(handle)
    yc2uj={0: 0,
     1: 1,
     2: 2,
     3: 3,
     4: 4,
     5: 5,
     6: 6,
     7: 7,
     8: 8,
     9: 9,
     10: 10,
     12: 11,
     13: 12,
     14: 13,
     15: 14,
     16: 15,
     17: 16,
     18: 17,
     19: 18,
     20: 19,
     21: 20,
     22: 21,
     23: 22,
     24: 23,
     25: 24,
     26: 25,
     27: 26,
     28: 27,
     29: 28,
     30: 29,
     31: 30,
     32: 31,
     33: 32,
     34: 33,
     35: 34,
     36: 35,
     37: 36,
     38: 37,
     39: 38,
     40: 39,
     41: 40,
     42: 41}
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(val_loader)):
            if args.data_name == 'scannet_cross':
                (coords, feat, label_3d, color, label_2d, link, inds_reverse, cls, mat, mat3d, part13,
                 model_id) = batch_data
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
                label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
                cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
                mat3d = mat3d.cuda(non_blocking=True)
                output_3d, output_2d, output_cls, output_mat, output_3dmat = model(sinput, color, link)
            else:
                raise NotImplemented
            # ############ 3D ############ #
            # o3d=output_3d.detach().topk(5)[1]
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)
            # accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

            # ############ 2D ############ #
            # print(output_2d.shape)
            # o2d=output_2d.detach().topk(5, dim=1)[1]
            output_2d = output_2d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_2d.update(intersection)
            union_meter_2d.update(union)
            target_meter_2d.update(target)
            accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)
            # ############ mat_3d ############ #
            # o3dmat = output_3dmat.detach().topk(5)[1]
            output_3dmat = output_3dmat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
                                                                  args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3dmat.update(intersection)
            union_meter_3dmat.update(union)
            target_meter_3dmat.update(target)
            # accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)
            # ############ mat ############ #
            # omat=output_mat.detach().topk(5, dim=1)[1]
            output_mat = output_mat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_mat.update(intersection)
            union_meter_mat.update(union)
            target_meter_mat.update(target)
            # accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

            # ############ cls ############ #
            ocls = output_cls.detach().topk(5)[1]
            output_cls = output_cls.detach().max(1)[1]
            correct_guessed = output_cls == cls
            cls_b_acc = torch.sum(correct_guessed.double()).item()
            acc += cls_b_acc
            total += output_cls.size(0)

            for a, b, c, d, e, f in zip(output_2d.cpu().numpy(), output_mat.cpu().numpy(), ocls.cpu().numpy(),
                                        label_2d.cpu().numpy(), mat.cpu().numpy(), cls.cpu().numpy()):
                pred_part = np.unique(a)
                pred_mat = np.unique(b)
                gt_mat = np.unique(e)
                gt_part = np.unique(d)
                # Top1

                # if model_id[0] in resnet_predictions:
                #     com_id=model_id[0]
                #     top1 = resnet_predictions[com_id][0].item()
                #     top5 = resnet_predictions[com_id][1].tolist()
                #     gt = resnet_predictions[com_id][2].item()
                #     c = top5
                #     f = gt
                # mid=model_id[0].split('_')[0]
                # if mid in point_predictions:
                #     top5 = point_predictions[mid]
                #     c = top5

                bbox2d.update(c, pred_mat, pred_part, a, b,
                              f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2d.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2d.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # Top5
                bbox2dtop5.update(c, pred_mat, pred_part, a, b,
                                  f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2dtop5.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2dtop5.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # GT
                bbox2dgt.update(c, pred_mat, pred_part, a, b,
                                f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2dgt.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2dgt.update(c, pred_mat, b, f, gt_mat, e)  # GT Part

            if main_process() and i % 50 == 0:
                logger.info(
                    'Val TOP 1 result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2d.obj(), bbox2d.value(), bbox2d.value_all(), bbox2d.value_bbox(), bbox2d.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2d.obj(), bboxmat2d.value(), bboxmat2d.value_all(), bboxmat2d.value_bbox(),
                        bboxmat2d.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2d.obj(), bboxpart2d.value(), bboxpart2d.value_all(), bboxpart2d.value_bbox(),
                        bboxpart2d.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2dtop5.obj(), bbox2dtop5.value(), bbox2dtop5.value_all(), bbox2dtop5.value_bbox(),
                        bbox2dtop5.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2dtop5.obj(), bboxmat2dtop5.value(), bboxmat2dtop5.value_all(),
                        bboxmat2dtop5.value_bbox(), bboxmat2dtop5.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2dtop5.obj(), bboxpart2dtop5.value(), bboxpart2dtop5.value_all(),
                        bboxpart2dtop5.value_bbox(), bboxpart2dtop5.value_all_bbox()))
                logger.info(
                    'Val GT result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2dgt.obj(), bbox2dgt.value(), bbox2dgt.value_all(), bbox2dgt.value_bbox(),
                        bbox2dgt.value_all_bbox()))
                logger.info(
                    'Val GT result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2dgt.obj(), bboxmat2dgt.value(), bboxmat2dgt.value_all(), bboxmat2dgt.value_bbox(),
                        bboxmat2dgt.value_all_bbox()))
                logger.info(
                    'Val GT result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2dgt.obj(), bboxpart2dgt.value(), bboxpart2dgt.value_all(),
                        bboxpart2dgt.value_bbox(), bboxpart2dgt.value_all_bbox()))
############# 2d with seperate shape classifier
            for a, b, c, d, e, f in zip(output_2d.cpu().numpy(), output_mat.cpu().numpy(), ocls.cpu().numpy(),
                                        label_2d.cpu().numpy(), mat.cpu().numpy(), cls.cpu().numpy()):
                pred_part = np.unique(a)
                pred_mat = np.unique(b)
                gt_mat = np.unique(e)
                gt_part = np.unique(d)
                # Top1

                if model_id[0] in resnet_predictions:
                    com_id=model_id[0]
                    top1 = resnet_predictions[com_id][0].item()
                    top5 = resnet_predictions[com_id][1].tolist()
                    gt = resnet_predictions[com_id][2].item()
                    c = top5
                    f = gt
                # mid=model_id[0].split('_')[0]
                # if mid in point_predictions:
                #     top5 = point_predictions[mid]
                #     c = top5

                bbox2dcls.update(c, pred_mat, pred_part, a, b,
                              f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2dcls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2dcls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # Top5
                bbox2dtop5cls.update(c, pred_mat, pred_part, a, b,
                                  f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2dtop5cls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2dtop5cls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # GT
                bbox2dgtcls.update(c, pred_mat, pred_part, a, b,
                                f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat2dgtcls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart2dgtcls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part

            if main_process() and i % 50 == 0:
                logger.info(
                    'Val TOP 1 result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2dcls.obj(), bbox2dcls.value(), bbox2dcls.value_all(), bbox2dcls.value_bbox(), bbox2dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2dcls.obj(), bboxmat2dcls.value(), bboxmat2dcls.value_all(), bboxmat2dcls.value_bbox(),
                        bboxmat2dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2dcls.obj(), bboxpart2dcls.value(), bboxpart2dcls.value_all(), bboxpart2dcls.value_bbox(),
                        bboxpart2dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2dtop5cls.obj(), bbox2dtop5cls.value(), bbox2dtop5cls.value_all(), bbox2dtop5cls.value_bbox(),
                        bbox2dtop5cls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2dtop5cls.obj(), bboxmat2dtop5cls.value(), bboxmat2dtop5cls.value_all(),
                        bboxmat2dtop5cls.value_bbox(), bboxmat2dtop5cls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2dtop5cls.obj(), bboxpart2dtop5cls.value(), bboxpart2dtop5cls.value_all(),
                        bboxpart2dtop5cls.value_bbox(), bboxpart2dtop5cls.value_all_bbox()))
                logger.info(
                    'Val GT result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox2dgt.obj(), bbox2dgt.value(), bbox2dgt.value_all(), bbox2dgt.value_bbox(),
                        bbox2dgt.value_all_bbox()))
                logger.info(
                    'Val GT result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat2dgt.obj(), bboxmat2dgt.value(), bboxmat2dgt.value_all(), bboxmat2dgt.value_bbox(),
                        bboxmat2dgt.value_all_bbox()))
                logger.info(
                    'Val GT result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart2dgt.obj(), bboxpart2dgt.value(), bboxpart2dgt.value_all(),
                        bboxpart2dgt.value_bbox(), bboxpart2dgt.value_all_bbox()))


            for a, b, c, d, e, f in zip(np.expand_dims(output_3d.cpu().numpy(), axis=0),
                                        np.expand_dims(output_3dmat.cpu().numpy(), axis=0), ocls.cpu().numpy(),
                                        np.expand_dims(label_3d.cpu().numpy(), axis=0),
                                        np.expand_dims(mat3d.cpu().numpy(), axis=0), cls.cpu().numpy()):
                pred_part = np.unique(a)
                pred_mat = np.unique(b)
                gt_part = np.unique(d)
                gt_mat = np.unique(e)
                # mid=model_id[0].split('_')[0]
                # if mid in point_predictions:
                #     top5 = point_predictions[mid]
                #     c = top5
                #     f=yc2uj[f]
                # Top1
                bbox3d.update(c, pred_mat, pred_part, a, b,
                              f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3d.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3d.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # Top5
                bbox3dtop5.update(c, pred_mat, pred_part, a, b,
                                  f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3dtop5.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3dtop5.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # GT
                bbox3dgt.update(c, pred_mat, pred_part, a, b,
                                f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3dgt.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3dgt.update(c, pred_mat, b, f, gt_mat, e)  # GT Part

            if main_process() and i % 50 == 0:
                logger.info(
                    'Val TOP 1 result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3d.obj(), bbox3d.value(), bbox3d.value_all(), bbox3d.value_bbox(), bbox3d.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3d.obj(), bboxmat3d.value(), bboxmat3d.value_all(), bboxmat3d.value_bbox(),
                        bboxmat3d.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3d.obj(), bboxpart3d.value(), bboxpart3d.value_all(), bboxpart3d.value_bbox(),
                        bboxpart3d.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3dtop5.obj(), bbox3dtop5.value(), bbox3dtop5.value_all(), bbox3dtop5.value_bbox(),
                        bbox3dtop5.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3dtop5.obj(), bboxmat3dtop5.value(), bboxmat3dtop5.value_all(),
                        bboxmat3dtop5.value_bbox(), bboxmat3dtop5.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3dtop5.obj(), bboxpart3dtop5.value(), bboxpart3dtop5.value_all(),
                        bboxpart3dtop5.value_bbox(), bboxpart3dtop5.value_all_bbox()))
                logger.info(
                    'Val GT result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3dgt.obj(), bbox3dgt.value(), bbox3dgt.value_all(), bbox3dgt.value_bbox(),
                        bbox3dgt.value_all_bbox()))
                logger.info(
                    'Val GT result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3dgt.obj(), bboxmat3dgt.value(), bboxmat3dgt.value_all(), bboxmat3dgt.value_bbox(),
                        bboxmat3dgt.value_all_bbox()))
                logger.info(
                    'Val GT result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3dgt.obj(), bboxpart3dgt.value(), bboxpart3dgt.value_all(),
                        bboxpart3dgt.value_bbox(), bboxpart3dgt.value_all_bbox()))
########### Classifer 3d
            for a, b, c, d, e, f in zip(np.expand_dims(output_3d.cpu().numpy(), axis=0),
                                        np.expand_dims(output_3dmat.cpu().numpy(), axis=0), ocls.cpu().numpy(),
                                        np.expand_dims(label_3d.cpu().numpy(), axis=0),
                                        np.expand_dims(mat3d.cpu().numpy(), axis=0), cls.cpu().numpy()):
                pred_part = np.unique(a)
                pred_mat = np.unique(b)
                gt_part = np.unique(d)
                gt_mat = np.unique(e)
                mid = model_id[0].split('_')[0]
                if mid in point_predictions:
                    top5 = point_predictions[mid]
                    c = top5
                    f = yc2uj[f]
                # Top1
                bbox3dcls.update(c, pred_mat, pred_part, a, b,
                              f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3dcls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3dcls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # Top5
                bbox3dtop5cls.update(c, pred_mat, pred_part, a, b,
                                  f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3dtop5cls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3dtop5cls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part
                # GT
                bbox3dgtcls.update(c, pred_mat, pred_part, a, b,
                                f, gt_mat, gt_part, d, e, part13, model_id)
                bboxmat3dgtcls.update(c, pred_part, a, f, gt_part, d)  # GT MAT
                bboxpart3dgtcls.update(c, pred_mat, b, f, gt_mat, e)  # GT Part

            if main_process() and i % 50 == 0:
                logger.info(
                    'Val TOP 1 result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3dcls.obj(), bbox3dcls.value(), bbox3dcls.value_all(), bbox3dcls.value_bbox(), bbox3dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result separate classifier 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3dcls.obj(), bboxmat3dcls.value(), bboxmat3dcls.value_all(), bboxmat3dcls.value_bbox(),
                        bboxmat3dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 1 result separate classifier 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3dcls.obj(), bboxpart3dcls.value(), bboxpart3dcls.value_all(), bboxpart3dcls.value_bbox(),
                        bboxpart3dcls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3dtop5cls.obj(), bbox3dtop5cls.value(), bbox3dtop5cls.value_all(), bbox3dtop5cls.value_bbox(),
                        bbox3dtop5cls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3dtop5cls.obj(), bboxmat3dtop5cls.value(), bboxmat3dtop5cls.value_all(),
                        bboxmat3dtop5cls.value_bbox(), bboxmat3dtop5cls.value_all_bbox()))
                logger.info(
                    'Val TOP 5 result separate classifier 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3dtop5cls.obj(), bboxpart3dtop5cls.value(), bboxpart3dtop5cls.value_all(),
                        bboxpart3dtop5cls.value_bbox(), bboxpart3dtop5cls.value_all_bbox()))
                logger.info(
                    'Val GT result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bbox3dgtcls.obj(), bbox3dgtcls.value(), bbox3dgtcls.value_all(), bbox3dgtcls.value_bbox(),
                        bbox3dgtcls.value_all_bbox()))
                logger.info(
                    'Val gtcls result separate classifier 3d gtcls mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxmat3dgtcls.obj(), bboxmat3dgtcls.value(), bboxmat3dgtcls.value_all(), bboxmat3dgtcls.value_bbox(),
                        bboxmat3dgtcls.value_all_bbox()))
                logger.info(
                    'Val gtcls result separate classifier 3d gtcls part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                        bboxpart3dgtcls.obj(), bboxpart3dgtcls.value(), bboxpart3dgtcls.value_all(),
                        bboxpart3dgtcls.value_bbox(), bboxpart3dgtcls.value_all_bbox()))
            # for a, b, c, d, e, f in zip(output_2dcls.cpu().numpy(), output_mat.cpu().numpy(), ocls.cpu().numpy(), label_2d.cpu().numpy(),  mat.cpu().numpy(), cls.cpu().numpy()):
            #     pred_part = np.unique(a)
            #     pred_mat = np.unique(b)
            #     gt_mat = np.unique(e)
            #     gt_part = np.unique(d)
            #     bbox2d.update(c, pred_mat, pred_part, a,b,
            #                   f, gt_mat, gt_part, d,e)
            #     bboxmat2d.update(c,pred_part,b,f, gt_part,e) # GT MAT
            #     bboxpart2d.update(c, pred_mat, b, f, gt_mat, e) #GT Part

            # bboxmat2d.update()
            # bboxpart2d.update()

            # for a, b, c, d, e, f in zip(output_2d.detach_().cpu().numpy(), output_mat.detach_().cpu().numpy(),
            #                             output_cls.detach_().cpu().numpy(), label_2d.cpu().numpy(), mat.cpu().numpy(),
            #                             cls.cpu().numpy()):
            #     pred_part = list(set(a.reshape(a.size)))
            #     pred_mat = list(set(b.reshape(b.size)))
            #     gt_mat = list(set(e.reshape(e.size)))
            #     gt_part = list(set(d.reshape(d.size)))
            #     bboxmat.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
            #     bbox.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
            #
            # for a, b, c, d, e, f in zip(output_3d.detach_().cpu().numpy(), output_3dmat.detach_().cpu().numpy(),
            #                             output_cls.detach_().cpu().numpy(), label_3d.cpu().numpy(), mat3d.cpu().numpy(),
            #                             cls.cpu().numpy()):
            #     pred_part = list(set(a.reshape(a.size)))
            #     pred_mat = list(set(b.reshape(b.size)))
            #     gt_mat = list(set(e.reshape(e.size)))
            #     gt_part = list(set(d.reshape(d.size)))
            #     bboxmat3d.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)
            #     bbox3d.update(c, pred_mat, pred_part, a, f, gt_mat, gt_part, d)

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    # acc_cls = acc / total

    iou_class_3dmat = intersection_meter_3dmat.sum / (union_meter_3dmat.sum + 1e-10)
    accuracy_class_3dmat = intersection_meter_3dmat.sum / (target_meter_3dmat.sum + 1e-10)
    mIoU_3dmat = np.mean(iou_class_3dmat)
    mAcc_3dmat = np.mean(accuracy_class_3dmat)
    allAcc_3dmat = sum(intersection_meter_3dmat.sum) / (sum(target_meter_3dmat.sum) + 1e-10)

    # allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    acc_cls = acc / total

    iou_class_mat = intersection_meter_mat.sum / (union_meter_mat.sum + 1e-10)
    accuracy_class_mat = intersection_meter_mat.sum / (target_meter_mat.sum + 1e-10)
    mIoU_mat = np.mean(iou_class_mat)
    mAcc_mat = np.mean(accuracy_class_mat)
    allAcc_mat = sum(intersection_meter_mat.sum) / (sum(target_meter_mat.sum) + 1e-10)

    if main_process():
        logger.info("##########This is the Final###########")
        logger.info(
            'Val TOP 1 result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2d.obj(), bbox2d.value(), bbox2d.value_all(), bbox2d.value_bbox(), bbox2d.value_all_bbox()))
        logger.info(
            'Val TOP 1 result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2d.obj(), bboxmat2d.value(), bboxmat2d.value_all(), bboxmat2d.value_bbox(),
                bboxmat2d.value_all_bbox()))
        logger.info(
            'Val TOP 1 result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2d.obj(), bboxpart2d.value(), bboxpart2d.value_all(), bboxpart2d.value_bbox(),
                bboxpart2d.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2dtop5.obj(), bbox2dtop5.value(), bbox2dtop5.value_all(), bbox2dtop5.value_bbox(),
                bbox2dtop5.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2dtop5.obj(), bboxmat2dtop5.value(), bboxmat2dtop5.value_all(),
                bboxmat2dtop5.value_bbox(), bboxmat2dtop5.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2dtop5.obj(), bboxpart2dtop5.value(), bboxpart2dtop5.value_all(),
                bboxpart2dtop5.value_bbox(), bboxpart2dtop5.value_all_bbox()))
        logger.info(
            'Val GT result 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2dgtcls.obj(), bbox2dgtcls.value(), bbox2dgtcls.value_all(), bbox2dgtcls.value_bbox(),
                bbox2dgtcls.value_all_bbox()))
        logger.info(
            'Val GT result 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2dgtcls.obj(), bboxmat2dgtcls.value(), bboxmat2dgtcls.value_all(), bboxmat2dgtcls.value_bbox(),
                bboxmat2dgtcls.value_all_bbox()))
        logger.info(
            'Val GT result 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2dgtcls.obj(), bboxpart2dgtcls.value(), bboxpart2dgtcls.value_all(),
                bboxpart2dgtcls.value_bbox(), bboxpart2dgtcls.value_all_bbox()))

    if main_process():
        logger.info(
            'Val TOP 1 result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2dcls.obj(), bbox2dcls.value(), bbox2dcls.value_all(), bbox2dcls.value_bbox(),
                bbox2dcls.value_all_bbox()))
        logger.info(
            'Val TOP 1 result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2dcls.obj(), bboxmat2dcls.value(), bboxmat2dcls.value_all(), bboxmat2dcls.value_bbox(),
                bboxmat2dcls.value_all_bbox()))
        logger.info(
            'Val TOP 1 result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2dcls.obj(), bboxpart2dcls.value(), bboxpart2dcls.value_all(), bboxpart2dcls.value_bbox(),
                bboxpart2dcls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2dtop5cls.obj(), bbox2dtop5cls.value(), bbox2dtop5cls.value_all(), bbox2dtop5cls.value_bbox(),
                bbox2dtop5cls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2dtop5cls.obj(), bboxmat2dtop5cls.value(), bboxmat2dtop5cls.value_all(),
                bboxmat2dtop5cls.value_bbox(), bboxmat2dtop5cls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2dtop5cls.obj(), bboxpart2dtop5cls.value(), bboxpart2dtop5cls.value_all(),
                bboxpart2dtop5cls.value_bbox(), bboxpart2dtop5cls.value_all_bbox()))
        logger.info(
            'Val GT result separate classifier 2d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox2dgt.obj(), bbox2dgt.value(), bbox2dgt.value_all(), bbox2dgt.value_bbox(),
                bbox2dgt.value_all_bbox()))
        logger.info(
            'Val GT result separate classifier 2d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat2dgt.obj(), bboxmat2dgt.value(), bboxmat2dgt.value_all(), bboxmat2dgt.value_bbox(),
                bboxmat2dgt.value_all_bbox()))
        logger.info(
            'Val GT result separate classifier 2d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart2dgt.obj(), bboxpart2dgt.value(), bboxpart2dgt.value_all(),
                bboxpart2dgt.value_bbox(), bboxpart2dgt.value_all_bbox()))

    if main_process():
        logger.info(
            'Val TOP 1 result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3d.obj(), bbox3d.value(), bbox3d.value_all(), bbox3d.value_bbox(), bbox3d.value_all_bbox()))
        logger.info(
            'Val TOP 1 result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3d.obj(), bboxmat3d.value(), bboxmat3d.value_all(), bboxmat3d.value_bbox(),
                bboxmat3d.value_all_bbox()))
        logger.info(
            'Val TOP 1 result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3d.obj(), bboxpart3d.value(), bboxpart3d.value_all(), bboxpart3d.value_bbox(),
                bboxpart3d.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3dtop5.obj(), bbox3dtop5.value(), bbox3dtop5.value_all(), bbox3dtop5.value_bbox(),
                bbox3dtop5.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3dtop5.obj(), bboxmat3dtop5.value(), bboxmat3dtop5.value_all(),
                bboxmat3dtop5.value_bbox(), bboxmat3dtop5.value_all_bbox()))
        logger.info(
            'Val TOP 5 result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3dtop5.obj(), bboxpart3dtop5.value(), bboxpart3dtop5.value_all(),
                bboxpart3dtop5.value_bbox(), bboxpart3dtop5.value_all_bbox()))
        logger.info(
            'Val GT result 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3dgt.obj(), bbox3dgt.value(), bbox3dgt.value_all(), bbox3dgt.value_bbox(),
                bbox3dgt.value_all_bbox()))
        logger.info(
            'Val GT result 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3dgt.obj(), bboxmat3dgt.value(), bboxmat3dgt.value_all(), bboxmat3dgt.value_bbox(),
                bboxmat3dgt.value_all_bbox()))
        logger.info(
            'Val GT result 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3dgt.obj(), bboxpart3dgt.value(), bboxpart3dgt.value_all(),
                bboxpart3dgt.value_bbox(), bboxpart3dgt.value_all_bbox()))

    if main_process():
        logger.info(
            'Val TOP 1 result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3dcls.obj(), bbox3dcls.value(), bbox3dcls.value_all(), bbox3dcls.value_bbox(),
                bbox3dcls.value_all_bbox()))
        logger.info(
            'Val TOP 1 result separate classifier 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3dcls.obj(), bboxmat3dcls.value(), bboxmat3dcls.value_all(), bboxmat3dcls.value_bbox(),
                bboxmat3dcls.value_all_bbox()))
        logger.info(
            'Val TOP 1 result separate classifier 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3dcls.obj(), bboxpart3dcls.value(), bboxpart3dcls.value_all(), bboxpart3dcls.value_bbox(),
                bboxpart3dcls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3dtop5cls.obj(), bbox3dtop5cls.value(), bbox3dtop5cls.value_all(), bbox3dtop5cls.value_bbox(),
                bbox3dtop5cls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 3d gt mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3dtop5cls.obj(), bboxmat3dtop5cls.value(), bboxmat3dtop5cls.value_all(),
                bboxmat3dtop5cls.value_bbox(), bboxmat3dtop5cls.value_all_bbox()))
        logger.info(
            'Val TOP 5 result separate classifier 3d gt part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3dtop5cls.obj(), bboxpart3dtop5cls.value(), bboxpart3dtop5cls.value_all(),
                bboxpart3dtop5cls.value_bbox(), bboxpart3dtop5cls.value_all_bbox()))
        logger.info(
            'Val GT result separate classifier 3d: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bbox3dgtcls.obj(), bbox3dgtcls.value(), bbox3dgtcls.value_all(), bbox3dgtcls.value_bbox(),
                bbox3dgtcls.value_all_bbox()))
        logger.info(
            'Val gtcls result separate classifier 3d gtcls mat: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxmat3dgtcls.obj(), bboxmat3dgtcls.value(), bboxmat3dgtcls.value_all(), bboxmat3dgtcls.value_bbox(),
                bboxmat3dgtcls.value_all_bbox()))
        logger.info(
            'Val gtcls result separate classifier 3d gtcls part: object acc/value/value_all/value_bbox/value_bbox_all {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                bboxpart3dgtcls.obj(), bboxpart3dgtcls.value(), bboxpart3dgtcls.value_all(),
                bboxpart3dgtcls.value_bbox(), bboxpart3dgtcls.value_all_bbox()))

    if main_process():
        logger.info(
            'Val result 3d: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
        logger.info(
            'Val result 2d : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_2d, mAcc_2d, allAcc_2d))
        logger.info(
            'Val result 2dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_mat, mAcc_mat, allAcc_mat))
        logger.info('Class ACC{:.4f}'.format(acc_cls))
        logger.info(
            'Val result 3dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3dmat, mAcc_3dmat, allAcc_3dmat))
        # logger.info('Class ACC{:.4f}'.format(acc_cls))
    return mIoU_3d, mAcc_3d, allAcc_3d, \
           mIoU_2d, mAcc_2d, allAcc_2d


def test_cross_3d(model, val_data_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with torch.no_grad():
        model.eval()
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                pbar = tqdm(total=len(val_data_loader))
            for i, (coords, feat, label_3d, color, label_2d, link, inds_reverse) in enumerate(val_data_loader):
                if main_process():
                    pbar.update(1)
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
                label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
                output_3d, output_2d = model(sinput, color, link)
                output_2d = output_2d.contiguous()
                output_3d = output_3d[inds_reverse, :]
                if args.multiprocessing_distributed:
                    dist.all_reduce(output_3d)
                    dist.all_reduce(output_2d)

                output_2d = output_2d.detach().max(1)[1]
                intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                                      args.ignore_label)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

                preds.append(output_3d.detach_().cpu())
                gts.append(label_3d.cpu())
            if main_process():
                pbar.close()
            gt = torch.cat(gts)
            pred = torch.cat(preds)
            if rep_i == 0:
                np.save(join(args.save_folder, 'gt.npy'), gt.numpy())
            store = pred + store
            mIou_3d = iou.evaluate(store.max(1)[1].numpy(), gt.numpy())
            np.save(join(args.save_folder, 'pred.npy'), store.max(1)[1].numpy())

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU_2d = np.mean(iou_class)
            # mAcc = np.mean(accuracy_class)
            # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
            if main_process():
                print("2D: ", mIoU_2d, ", 3D: ", mIou_3d)


if __name__ == '__main__':
    main()
