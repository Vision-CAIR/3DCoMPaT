import os
import time
import random
import numpy as np
import logging
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from os.path import join
import torch.nn.functional as F
from MinkowskiEngine import SparseTensor, CoordsManager
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, poly_learning_rate, save_checkpoint

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
best_iou = 0.0


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


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
    # Log for check version
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False
    #
    # if args.data_name == 'scannet_3d_mink':
    #     from dataset.scanNet3DClsNew import ScanNet3D, collation_fn
    #     _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
    #                   memCacheInit=True, loop=5)
    #     if args.evaluate:
    #         _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='valid', aug=args.aug,
    #                       memCacheInit=True)
    # elif args.data_name == 'scannet_cross':
    #     from dataset.scanNetCrossStyleNew import ScanNetCross, collation_fn, collation_fn_eval_all
    #     _ = ScanNetCross(dataPathPrefix=args.data_root, dataPathPrefix2D=args.data_root2d, voxelSize=args.voxelSize,
    #                      split='train', aug=args.aug,
    #                      memCacheInit=True, loop=args.loop)
    #     if args.evaluate:
    #         _ = ScanNetCross(dataPathPrefix=args.data_root, dataPathPrefix2D=args.data_root2d, voxelSize=args.voxelSize,
    #                          split='valid', aug=False,
    #                          memCacheInit=True, eval_all=True)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if args.sync_bn_2d:
        print("using DDP synced BN for 2D")
        try:
            model.layer0_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer0_2d)
            model.layer1_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer1_2d)
            model.layer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer2_2d)
            model.layer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer3_2d)
            model.layer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer4_2d)
            model.up4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up4_2d)
            model.delayer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer4_2d)
            model.up3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up3_2d)
            model.delayer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer3_2d)
            model.up2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up2_2d)
            model.delayer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer2_2d)
            model.cls_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cls_2d)
        except:
            print("3d modeling")

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info("categories: {}".format(args.categories))
        logger.info(model)

    # ####################### Optimizer ####################### #
    if args.arch == 'bpnet':
        modules_ori = [model.layer0_2d, model.layer1_2d, model.layer2_2d, model.layer3_2d, model.layer4_2d]
        modules_new = [
            model.up4_2d, model.delayer4_2d, model.up3_2d, model.delayer3_2d, model.up2_2d, model.delayer2_2d,
            model.cls_2d,
            model.layer0_3d, model.layer1_3d, model.layer2_3d, model.layer3_3d, model.layer4_3d, model.layer5_3d,
            model.layer6_3d, model.layer7_3d, model.layer8_3d, model.layer9_3d, model.cls_3d,
            model.linker_p2, model.linker_p3, model.linker_p4, model.linker_p5
        ]
        params_list = []
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
        args.index_split = len(modules_ori)
        optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #

    if args.data_name == 'scannet_3d_mink':

        from dataset.scanNet3DClsNew import ScanNet3D, collation_fn, collation_fn_eval_all
        train_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                               memCacheInit=True, loop=args.loop)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn)
        if args.evaluate:
            val_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='valid', aug=False,
                                 memCacheInit=True, eval_all=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all,
                                                     sampler=val_sampler)
    elif args.data_name == 'scannet_cross':
        if args.profile:
            if args.profile:
                # model.eval()
                print("Profile111: True")

                class ScanNetCross(torch.utils.data.Dataset):
                    def __init__(self):
                        self.coords = torch.rand(100, 2000, 4)
                        self.feats = torch.rand(100, 2000, 3)
                        self.labels = torch.rand(100, 2000)
                        self.colors = torch.rand(100, 2000, 4)
                        self.labels_2d = torch.rand(100, 4, 224, 224, 4)
                        self.links = torch.rand(100, 2000, 4, 5)
                        self.category = torch.rand(100, 4)
                        self.materials = torch.rand(100, 4, 224, 224, 4)

                    def __len__(self):
                        return len(self.coords)

                    def __getitem__(self, idx):
                        return self.coords[idx], self.feats[idx], self.labels[idx], self.colors[idx], self.labels_2d[
                            idx], \
                               self.links[idx], self.category[idx], self.materials[idx]

                def collation_fn(batch):
                    """
                    :param batch:
                    :return:    coords: N x 4 (batch,x,y,z)
                                feats:  N x 3
                                labels: N
                                colors: B x C x H x W x V
                                labels_2d:  B x H x W x V
                                links:  N x 4 x V (B,H,W,mask)

                    """
                    coords, feats, labels, colors, labels_2d, links, cls, mat = list(zip(*batch))
                    # pdb.set_trace()

                    for i in range(len(coords)):
                        coords[i][:, 0] *= i
                        links[i][:, 0, :] *= i
                    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
                           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.stack(cls), torch.stack(
                        mat)

                train_data = ScanNetCross()
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_data) if args.distributed else None
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                           shuffle=(train_sampler is None),
                                                           num_workers=args.workers, pin_memory=True,
                                                           sampler=train_sampler,
                                                           drop_last=True, collate_fn=collation_fn,
                                                           worker_init_fn=worker_init_fn)
                if args.evaluate:
                    val_data = ScanNetCross()
                    val_sampler = torch.utils.data.distributed.DistributedSampler(
                        val_data) if args.distributed else None
                    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                             shuffle=False, num_workers=args.workers, pin_memory=True,
                                                             drop_last=False, collate_fn=collation_fn,
                                                             sampler=val_sampler)
        else:
            from dataset.scanNetCrossStyleNew import ScanNetCross, collation_fn, collation_fn_eval_all
            train_data = ScanNetCross(dataPathPrefix=args.data_root, dataPathPrefix2D=args.data_root2d,
                                      voxelSize=args.voxelSize, split='train', aug=args.aug,
                                      memCacheInit=True, loop=args.loop, args=args)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=(train_sampler is None),
                                                       num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                       drop_last=True, collate_fn=collation_fn,
                                                       worker_init_fn=worker_init_fn)
            if args.evaluate:
                val_data = ScanNetCross(dataPathPrefix=args.data_root, dataPathPrefix2D=args.data_root2d,
                                        voxelSize=args.voxelSize, split='test', aug=False,
                                        memCacheInit=True, eval_all=True, args=args)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
                val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                         shuffle=False, num_workers=args.workers, pin_memory=True,
                                                         drop_last=False, collate_fn=collation_fn_eval_all,
                                                         sampler=val_sampler)
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        if args.data_name == 'scannet_cross':
            loss_train, mIoU_train, mAcc_train, allAcc_train, \
            loss_train_2d, mIoU_train_2d, mAcc_train_2d, allAcc_train_2d \
                = train_cross(train_loader, model, criterion, optimizer, epoch)
        else:
            loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            if args.data_name == 'scannet_cross':
                writer.add_scalar('loss_train_2d', loss_train_2d, epoch_log)
                writer.add_scalar('mIoU_train_2d', mIoU_train_2d, epoch_log)
                writer.add_scalar('mAcc_train_2d', mAcc_train_2d, epoch_log)
                writer.add_scalar('allAcc_train_2d', allAcc_train_2d, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'scannet_cross':
                loss_val, mIoU_val, mAcc_val, allAcc_val, \
                loss_val_2d, mIoU_val_2d, mAcc_val_2d, allAcc_val_2d \
                    = validate_cross(val_loader, model, criterion)
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                if args.data_name == 'scannet_cross':
                    writer.add_scalar('loss_val_2d', loss_val_2d, epoch_log)
                    writer.add_scalar('mIoU_val_2d', mIoU_val_2d, epoch_log)
                    writer.add_scalar('mAcc_val_2d', mAcc_val_2d, epoch_log)
                    writer.add_scalar('allAcc_val_2d', allAcc_val_2d, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    if cfg.arch == 'mink_18A':
        from models.unet_3d import MinkUNet18A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3)
    elif cfg.arch == 'mink_34C':
        from models.unet_3d import MinkUNet34C as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3)
    elif cfg.arch == 'bpnet':
        from models.bpnetClsNew import BPNet as Model
        model = Model(cfg=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model


def train(train_loader, model, criterion, optimizer, epoch):
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, label) = batch_data
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, :3] += (torch.rand(3) * 100).type_as(coords)

        sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
        label = label.cuda(non_blocking=True)
        output = model(sinput)
        # pdb.set_trace()
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        if args.arch == 'psp50':
            for index in range(0, args.index_split):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(args.index_split, len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
            label = label.cuda(non_blocking=True)
            output = model(sinput)
            # pdb.set_trace()
            output = output[inds_reverse, :]
            loss = criterion(output, label)

            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def train_cross(train_loader, model, criterion, optimizer, epoch):
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter, loss_meter_3d, loss_meter_2d, loss_meter_cls = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter_mat, loss_meter_3dmat = AverageMeter(), AverageMeter()
    acc = 0
    total = 0
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.data_name == 'scannet_cross':
            (coords, feat, label_3d, color, label_2d, link, cls, mat, mat_3d) = batch_data
            # For some networks, making the network invariant to even, odd coords is important
            coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
            # 2d colors, labels_2d, links
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)

            label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
            cls, mat, mat_3d = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True), mat_3d.cuda(non_blocking=True)


            output_3d, output_2d, output_cls, output_mat, output_3dmat = model(sinput, color, link)

            loss_2d = F.nll_loss(F.log_softmax(output_2d, dim=1), label_2d)
            loss_mat = F.nll_loss(F.log_softmax(output_mat, dim=1), mat)

            loss_cls = criterion(output_cls, cls)
            loss_3d = criterion(output_3d, label_3d)
            loss_3dmat = criterion(output_3dmat, mat_3d)

            # loss_2d = criterion(output_2d, label_2d)
            # loss_mat = criterion(output_mat, mat)

            loss = loss_3d + loss_2d + loss_cls + loss_mat + loss_3dmat
        else:
            raise NotImplemented
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ############ 3D ############ #
        output_3d = output_3d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_3d.update(intersection)
        union_meter_3d.update(union)
        target_meter_3d.update(target)
        accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

        # ############ mat_3d ############ #
        output_3dmat = output_3dmat.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat_3d.detach(), args.mat,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_3dmat.update(intersection)
        union_meter_3dmat.update(union)
        target_meter_3dmat.update(target)
        accuracy_3dmat = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

        # ############ 2D ############ #
        output_2d = output_2d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_2d.update(intersection)
        union_meter_2d.update(union)
        target_meter_2d.update(target)
        accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)
        # ############ cls ############ #
        output_cls = output_cls.detach().max(1)[1]
        correct_guessed = output_cls == cls
        cls_b_acc = torch.sum(correct_guessed.double()).item()
        acc += cls_b_acc
        total += output_cls.size(0)
        # ############ materials ############ #
        output_mat = output_mat.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_mat.update(intersection)
        union_meter_mat.update(union)
        target_meter_mat.update(target)
        accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_2d.update(loss_2d.item(), args.batch_size)
        loss_meter_3d.update(loss_3d.item(), args.batch_size)
        loss_meter_cls.update(loss_cls.item(), args.batch_size)
        loss_meter_mat.update(loss_mat.item(), args.batch_size)
        loss_meter_3dmat.update(loss_3dmat.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        # 2d compat result:

        # 3d compat resutl: Todo
        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # if args.arch == 'cross_p5' or args.arch == 'cross_p2':
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}'
                        'Loss {loss_meter_2d.val:.4f}'
                        'Accuracy2D {accuracy_2d:.4f}'
                        'Acc Cls {acc:.4f}'
                        'Loss_mat {loss_meter_mat.val:.4f} '
                        'Materials ACC {acc_mat:.4f}. '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                               batch_time=batch_time, data_time=data_time,
                                                               remain_time=remain_time,
                                                               loss_meter=loss_meter_3d,
                                                               accuracy=accuracy_3d,
                                                               loss_meter_2d=loss_meter_2d,
                                                               accuracy_2d=accuracy_2d,
                                                               acc=acc / total, loss_meter_mat=loss_meter_mat,
                                                               acc_mat=accuracy_mat))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss3d_train_batch', loss_meter_3d.val, current_iter)
            writer.add_scalar('loss2d_train_batch', loss_meter_2d.val, current_iter)
            writer.add_scalar('mIoU3d_train_batch', np.mean(intersection_meter_3d.val / (union_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc3d_train_batch', np.mean(intersection_meter_3d.val / (target_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc3d_train_batch', accuracy_3d, current_iter)

            writer.add_scalar('mIoU2d_train_batch', np.mean(intersection_meter_2d.val / (union_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc2d_train_batch', np.mean(intersection_meter_2d.val / (target_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc2d_train_batch', accuracy_2d, current_iter)

            writer.add_scalar('learning_rate', current_lr, current_iter)

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

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU_3d, mAcc_3d, allAcc_3d))
        logger.info(
            'Train result 2d at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                              mIoU_2d, mAcc_2d,
                                                                                              allAcc_2d))
        logger.info('Train result cls at at epoch [{}/{}]: acc:{:.4f}/total:{}'.format(epoch + 1, args.epochs,
                                                                                       acc / total, total))

    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d


# def validate_cross(val_loader, model, criterion):
#     torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
#     loss_meter, loss_meter_3d, loss_meter_2d, loss_meter_cls = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
#     intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
#     union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
#     target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
#     loss_meter_mat = AverageMeter()
#     target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
#     acc = 0
#     total = 0
#     model.eval()
#     with torch.no_grad():
#         for i, batch_data in enumerate(val_loader):
#             if args.data_name == 'scannet_cross':
#                 (coords, feat, label_3d, color, label_2d, link, inds_reverse, cls, mat) = batch_data
#                 sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
#                 color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
#                 label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
#                 cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
#                 output_3d, output_2d, output_cls, output_mat = model(sinput, color, link)
#                 output_3d = output_3d[inds_reverse, :]
#                 loss_mat = criterion(output_mat, mat)
#                 loss_cls = criterion(output_cls, cls)
#                 loss_3d = criterion(output_3d, label_3d)
#                 loss_2d = criterion(output_2d, label_2d)
#                 loss = loss_3d + args.weight_2d * loss_2d + 0.1 * loss_cls + 0.1 * loss_mat
#             else:
#                 raise NotImplemented
#             # ############ 3D ############ #
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
#             # ############ 2D ############ #
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
#
#             # ############ mat ############ #
#             output_mat = output_mat.detach().max(1)[1]
#             intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.classes,
#                                                                   args.ignore_label)
#             if args.multiprocessing_distributed:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
#             intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter_mat.update(intersection)
#             union_meter_mat.update(union)
#             target_meter_mat.update(target)
#             accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)
#
#
#             # ############ cls ############ #
#             output_cls = output_cls.detach().max(1)[1]
#             correct_guessed = output_cls == cls
#             cls_b_acc = torch.sum(correct_guessed.double()).item()
#             acc += cls_b_acc
#             total += output_cls.size(0)
#
#             loss_meter.update(loss.item(), args.batch_size)
#             loss_meter_cls.update(loss_cls.item(), args.batch_size)
#             loss_meter_2d.update(loss_2d.item(), args.batch_size)
#             loss_meter_3d.update(loss_3d.item(), args.batch_size)
#             loss_meter_mat.update(loss_mat.item(), args.batch_size)
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
#             'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
#         logger.info(
#             'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_2d, mAcc_2d, allAcc_2d))
#         logger.info(
#             'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_mat, mAcc_mat, allAcc_mat))
#         logger.info('Class ACC'.format(acc_cls))
#
#     return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
#            loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d
def validate_cross(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter, loss_meter_3d, loss_meter_2d, loss_meter_cls = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    loss_meter_mat = AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter_3dmat = AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        # outseg, outcls, outmat, gtcls, gtseg, gtmat, gtseg3d, outseg3d = [], [], [], [], [], [], [], []
        for i, batch_data in enumerate(val_loader):
            if args.data_name == 'scannet_cross':
                (coords, feat, label_3d, color, label_2d, link, inds_reverse, cls, mat, mat3d) = batch_data
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
                label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
                cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
                mat3d = mat3d.cuda(non_blocking=True)
                output_3d, output_2d, output_cls, output_mat, output_3dmat = model(sinput, color, link)
                # output_3d = output_3d[inds_reverse, :]
            else:
                raise NotImplemented

            loss_mat = criterion(output_mat, mat)
            loss_cls = criterion(output_cls, cls)
            loss_3d = criterion(output_3d, label_3d)
            loss_2d = criterion(output_2d, label_2d)
            loss_3dmat = criterion(output_3dmat, mat3d)

            loss = loss_3d + args.weight_2d * loss_2d + 0.1 * loss_cls + 0.1 * loss_mat + loss_3dmat

            # ############ 3D ############ #
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)
            accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

            # ############ 2D ############ #
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
            output_3dmat = output_3dmat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
                                                                  args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3dmat.update(intersection)
            union_meter_3dmat.update(union)
            target_meter_3dmat.update(target)
            accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)

            # ############ mat ############ #
            output_mat = output_mat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_mat.update(intersection)
            union_meter_mat.update(union)
            target_meter_mat.update(target)
            accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

            loss_meter.update(loss.item(), args.batch_size)
            loss_meter_cls.update(loss_cls.item(), args.batch_size)
            loss_meter_2d.update(loss_2d.item(), args.batch_size)
            loss_meter_3d.update(loss_3d.item(), args.batch_size)
            loss_meter_mat.update(loss_mat.item(), args.batch_size)
            loss_meter_3dmat.update(loss_3dmat.item(), args.batch_size)
            # ############ cls ############ #
            output_cls = output_cls.detach().max(1)[1]
            correct_guessed = output_cls == cls
            cls_b_acc = torch.sum(correct_guessed.double()).item()
            acc += cls_b_acc
            total += output_cls.size(0)

    # outcls1 = torch.cat(outcls)
    # outseg1 = torch.cat(outseg)
    # outseg3d1 = torch.cat(outseg3d)
    # outmat1 = torch.cat(outmat)
    # gtcls1 = torch.cat(gtcls)
    # gtmat1 = torch.cat(gtmat)
    # gtseg1 = torch.cat(gtseg)
    # gtseg3d1 = torch.cat(gtseg3d)
    #
    # np.save(join(args.save_folder, 'outcls.npy'), outcls1.numpy()
    #         )
    # np.save(join(args.save_folder, 'outseg.npy'), outseg1.numpy()
    #         )
    # np.save(join(args.save_folder, 'outseg3d.npy'), outseg3d1.numpy()
    #         )
    # np.save(join(args.save_folder, 'outmat.npy'), outmat1.numpy()
    #         )
    # # mIou_2d = iou.evaluate(store.max(1)[1].numpy(), gt.numpy())
    # np.save(join(args.save_folder, 'gtcls.npy'), gtcls1.numpy())
    # np.save(join(args.save_folder, 'gtseg.npy'), gtseg1.numpy())
    # np.save(join(args.save_folder, 'gtseg3d.npy'), gtseg3d1.numpy())
    # np.save(join(args.save_folder, 'gtmat.npy'), gtmat1.numpy())

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
        logger.info(
            'Val result 3d: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
        logger.info(
            'Val result 2s : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_2d, mAcc_2d, allAcc_2d))
        logger.info(
            'Val result 2dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_mat, mAcc_mat, allAcc_mat))
        logger.info('Class ACC{:.4f}'.format(acc_cls))
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3dmat, mAcc_3dmat, allAcc_3dmat))
        # logger.info('Class ACC{:.4f}'.format(acc_cls))

    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d


if __name__ == '__main__':
    main()
