import math
import torch.nn.functional as F
import torch
from torch.nn.utils.weight_norm import WeightNorm
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# class distLinear(nn.Module):
#     def __init__(self, indim, outdim):
#         super(distLinear, self).__init__()
#         self.L = nn.Linear(indim, outdim, bias=False)
#         self.class_wise_learnable_norm = True
#         if self.class_wise_learnable_norm:
#             WeightNorm.apply(self.L, 'weight', dim=0)
#         if outdim <= 200:
#             self.scale_factor = 2
#         else:
#             self.scale_factor = 10
#
#     def forward(self, x):
#         x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
#         x_normalized = x.div(x_norm + 1e-5)
#         if not self.class_wise_learnable_norm:
#             L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
#             self.L.weight.data = self.L.weight.data.div(L_norm + 1e-5)
#         cos_dist = self.L(x_normalized)
#         scores = self.scale_factor * cos_dist
#         return scores


class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).to(device)
        running_var = torch.ones(x.data.size()[1]).to(device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


class ConvBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)
        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)
        if flatten:
            trunk.append(Flatten())
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetNopool(nn.Module):
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            if i == 0:
                indim = 3
            else:
                indim = 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]), padding=0 if i in [0, 1] else 1)
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetSNopool(nn.Module):
    def __init__(self):
        super().__init__()


class ConvNetS(nn.Module):
    def __init__(self):
        super().__init__()


def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def Conv4S():
    return ConvNetS(4)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4SNP():
    return ConvNetSNopool(4)
