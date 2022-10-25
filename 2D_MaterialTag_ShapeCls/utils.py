"""
Utility functions.
"""
import os
import random

import numpy as np
import torch

from datetime import datetime
import os.path as osp
from six.moves import cPickle
from six.moves import range

cudnn_deterministic = True

def seed_everything(seed=0):
    """
    Fixing all random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def compute_topk_acc(pred, targets, topk):
    """
    Computing top-k accuracy given prediction and target vectors.
    Args:
        pred:    Network prediction
        targets: Ground truth labels
        topk:    k value
    """
    topk = min(topk, pred.shape[1])
    _, pred = pred.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    hits_tag = correct[:topk].reshape(-1).float().sum(0)

    return hits_tag


def calculate_metrics(outputs, targets):
    """
    Computing top-1 and top-5 accuracy metrics.
    Args:
        outputs: Network outputs list
        targets: Ground truth labels
    """
    pred = outputs

    # Top-k prediction for TAg
    hits_tag_top5 = compute_topk_acc(pred, targets, 5)
    hits_tag_top1 = compute_topk_acc(pred, targets, 1)

    return hits_tag_top5.item(), hits_tag_top1.item()


def download_file(file_url, file_save_path):
    r = requests.get(file_url)

    if r.status_code != 200:
        print('Download error')

    with open(file_save_path, 'wb') as f:
        f.write(r.content)

def update_log_dir(log_dir):
    """
    Get unused log dir
    :return:
    """
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
    if osp.isdir(osp.join(log_dir, timestamp)):
        extra = 1
        while osp.isdir(osp.join(log_dir, timestamp + '(' + str(extra) + ')')):
            extra += 1
        ret = osp.join(log_dir, timestamp + '(' + str(extra) + ')')
    else:
        ret = osp.join(log_dir, timestamp)

    return ret


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    """
    Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :return: an generator over the un-pickled items.
    """
    in_file = open(file_name, 'rb')
    size = cPickle.load(in_file)

    for _ in range(size):
        yield cPickle.load(in_file)
    in_file.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

