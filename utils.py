import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, device, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda

    Code taken from: https://github.com/facebookresearch/mixup-cifar10
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing required in https://arxiv.org/abs/1812.01187

    Code taken from: https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    """

    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class AverageMeter(object):
    """Helper class to track the running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0
