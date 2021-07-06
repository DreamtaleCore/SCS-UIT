import torch
import torch.nn as nn
from torch import autograd


class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super().__init__()
        self.eps = 1e-8
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x + self.eps)
        y = y + self.eps
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x + self.eps)
        y = self.log_softmax(y + self.eps)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class WassersteinDistance(nn.Module):
    def __init__(self, drift=0.001):
        super().__init__()
        self.drift = drift
        pass

    def forward(self, real_data, fake_data):
        return (torch.mean(fake_data) - torch.mean(real_data)
                + (self.drift * torch.mean(real_data ** 2)))
