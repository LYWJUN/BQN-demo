import numpy as np
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import math
from typing import List
import logging
import copy
from torch_sparse import SparseTensor


device = torch.device("cuda:0")

def StandardScaler_crossROI(timeseries: np.array):
    """
    Standardize the parameters passed in
    """
    timeseries = timeseries.transpose(0, 2, 1)
    mean = np.mean(timeseries, axis=-1, keepdims=True)
    std = np.std(timeseries, axis=-1, keepdims=True)
    timeseries = (timeseries - mean) / std

    return timeseries.transpose(0, 2, 1)


def continues_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k ; which is in BNT"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def hyper_para_load(args, dataset):
    """
    load hyper parameters of model
    """
    if args.dataset == 'ABIDE':
        node_sz = dataset[0].shape[1]  # ROI number of each subject
        timeseries_sz = dataset[0].shape[-1]  # dim of timeseries
        node_feature_sz = dataset[1].shape[-1]  # dim of corr
    else:
        node_sz = 200
        timeseries_sz = 100
        node_feature_sz = 200

    layers = args.layers
    dropout = args.dropout

    pooling = args.pooling
    cluster_num = args.cluster_num

    orthogonal = True
    freeze_center = True
    project_assignment = True

    return (node_sz, timeseries_sz, node_feature_sz, layers, dropout,
            pooling, cluster_num)


def count_param(model: nn.Module):
    total_parameters = sum(p.numel() for p in model.parameters())
    return total_parameters


def optimizer_update(optimizer: torch.optim.Optimizer, step: int, total_steps: int, args):
    base_lr = args.base_lr
    target_lr = args.target_lr
    total_steps = total_steps

    current_ratio = step / total_steps
    cosine = math.cos(math.pi * current_ratio)
    lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_formatter() -> logging.Formatter:
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')


def initialize_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = get_formatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
