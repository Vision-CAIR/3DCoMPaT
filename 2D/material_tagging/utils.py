"""
Utility functions for material tagging.
"""
import os
import random

import numpy as np
import torch


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


def print_progress(n_epoch, total_epochs, time, loss, metrics):
    print('{:03}/{:03} | {} | Train : loss = {:.4f} | {} '.
            format(n_epoch+1, total_epochs,
                   time, loss,
                   " | ".join([str(m) for m in metrics])))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

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
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
