"""
Training a basic ResNet-18 on the 3DCompat dataset for material tagging.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from datetime import timedelta
from torchvision import models
from torch.nn.parallel import DataParallel
from torchmetrics import F1Score, AveragePrecision


import utils

from material_loader import MaterialTagLoader



def parse_args(argv):
    """
    Parsing input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(description='Training a basic ResNet-18 on the 3DCompat dataset.')

    # miscellaneous args
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')

    # dataset args
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for the dataloader (default=%(default)s)')
    parser.add_argument('--use-tmp', action='store_true',
                        help='Use local temporary cache when loading the dataset (default=%(default)s)')

    # data args
    parser.add_argument('--root-url', type=str, required=True,
                        help='Root URL for WebDataset shards (default=%(default)s)')
    parser.add_argument('--n-comp', type=int, required=True,
                        help='Number of compositions per model to train with')
    parser.add_argument('--view-type', type=str, default='all',
                        choices=['canonical', 'random', 'all'],
                        help='Train on a specific view type (default=%(default)s)')

    # training args
    parser.add_argument('--batch-size', default=32, type=int, required=False,
                        help='Batch size to use (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0005, type=float, required=False,
                        help='Weight decay (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum (default=%(default)s)')

    parser.add_argument('--nepochs', default=1, type=int, required=True,
                        help='Number of epochs to train with (default=%(default)s)')
    parser.add_argument('--patience', type=int, default=3, required=False,
                        help='Use patience while training (default=%(default)s)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training with the last saved model (default=%(default)s)')

    parser.add_argument('--resnet-type', default='resnet18', type=str, required=True,
                        choices=['resnet18', 'resnet50'],
                        help='ResNet variant to be used for training (default=%(default)s)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use a model pre-trained on ImageNet (default=%(default)s)')


    parser.add_argument('--models-dir', type=str, required=True,
                        help='Model directory to use to save models (default=%(default)s)')

    args = parser.parse_args(argv)
    args.view_type = -1 if args.view_type == 'all' else ['canonical', 'random'].index(args.view_type)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def evaluate(net, test_loader, device, f1, ap):
    """
    Evaluating the resulting classifier using a given test set loader.
    """
    # Initializing metrics
    f1_meter = utils.AverageMeter('F1', ':6.4f')
    ap_meter = utils.AverageMeter('AP', ':6.4f')

    # net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device).squeeze()

            outputs = net(images)

            # Updating metrics
            N = images.shape[0]

            f1_score = f1(torch.sigmoid(outputs).float(), labels.int())
            f1_meter.update(f1_score, N)

            ap_score = ap(outputs.float(), labels.int())
            ap_meter.update(ap_score, N)

    return f1_meter, ap_meter


def run_training(args):
    """
    Main training routine.
    """

    # Fixing random seed
    utils.seed_everything(args.seed)

    # Setting up dataset
    print("Loading data from: [%s]" % args.root_url)

    # Checking directory folder
    if not os.path.exists(args.models_dir):
        print("Output model directory [%s] not found. Creating..." % args.models_dir)
        os.makedirs(args.models_dir)
    
    # Initializing materials list
    mat_list = json.load(open("materials.json", "r"))
    num_classes = len(mat_list)

    # Model initialization
    res_model = {'resnet18': models.resnet18, 'resnet50': models.resnet50}
    fv_size   = {'resnet18': 512,             'resnet50': 2048}
    model = res_model[args.resnet_type](pretrained=args.use_pretrained)
    model.fc = nn.Linear(fv_size[args.resnet_type], num_classes)

    device = torch.device("cuda:0")

    # Optionally resume training
    if args.resume:
        all_models = [f for f in os.listdir(args.models_dir) if ".ckpt" in f]
        all_models.sort()

        last_model = os.path.join(args.models_dir, all_models[-1])

        checkpoint = torch.load(last_model)
        print("Loaded last checkpoint from [%s], with %0.2f train top-1 accuracy."
              % (last_model, checkpoint['top1_train_acc']))

        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.to(device)
    model = DataParallel(model)

    # Defining Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model.parameters(),
                             lr=0.1,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ft,
                                                  base_lr=0.01, max_lr=0.1)

    # Instantiating data loaders
    test_transforms = T.Compose([
        T.Normalize([0.8726, 0.8628, 0.8577], [0.2198, 0.2365, 0.2451])
    ])
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize([0.8710, 0.8611, 0.8561], [0.2217, 0.2384, 0.2468])
    ])

    train_loader = (
        MaterialTagLoader(root_url  = args.root_url,
                          split     = "train",
                          n_comp    = args.n_comp,
                          mat_list  = mat_list,
                          cache_dir = '/tmp/' if args.use_tmp else None,
                          view_type = args.view_type,
                          transform = train_transforms)
    ).make_loader(args.batch_size, args.num_workers)

    test_loader = (
        MaterialTagLoader(root_url  = args.root_url,
                          split     = "val",
                          n_comp    = args.n_comp,
                          mat_list  = mat_list,
                          cache_dir = '/tmp/' if args.use_tmp else None,
                          view_type = args.view_type,
                          transform = test_transforms)
    ).make_loader(args.batch_size, args.num_workers)

    # Training
    start_time = time.time()
    print("Starting training...")
    print("Number of training batches: [%d]" % train_loader.length)

    n_batch = 0
    loss = NotImplementedError

    # Defining metric functions
    f1 = F1Score(num_classes=num_classes).cuda()
    ap = AveragePrecision().cuda()

    for n_epoch in range(args.nepochs):
        # Training the model
        optimizer_ft.zero_grad()
        model.train()

        # Initializing metrics
        f1_meter = utils.AverageMeter('F1', ':6.4f')
        metrics  = [f1_meter]

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            targets = targets.squeeze()

            ## Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()
            optimizer_ft.zero_grad()

            ## Udpating metrics
            N = images.shape[0]

            f1_score = f1(torch.sigmoid(outputs).float(), targets.int())
            f1_meter.update(f1_score, N)

            ## Making a checkpoint
            if n_batch % 1000:
                saved_model = os.path.join(args.models_dir,
                "%s_batch_%d.ckpt" % (args.resnet_type, n_batch))

                # Measuring model test-accuracy
                val_metrics = evaluate(model, test_loader, device, f1, ap)
                f1_test_meter, ap_test_meter = val_metrics

                print("Saved model at: [" + saved_model + "]")
                state = {
                    "f1_train": f1_meter.avg,
                    "f1_test": f1_test_meter.avg,
                    "ap_test": ap_test_meter.avg,
                    "state_dict": model.state_dict()
                }
                torch.save(state, saved_model)
                utils.print_progress(n_epoch+1, args.nepochs, 0., loss,
                                     [f1_test_meter, ap_test_meter])

            n_batch += 1
            scheduler.step()

        # Logging results
        elapsed_time = timedelta(seconds=round(time.time() - start_time))
        utils.print_progress(n_epoch+1, args.nepochs, elapsed_time, loss, metrics)

    # Final output
    print('[Elapsed time = {} mn]'.format(elapsed_time))
    print('Done!')

    print('-' * 108)


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()

