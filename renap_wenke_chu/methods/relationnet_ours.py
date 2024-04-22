import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import torch.nn as nn
import numpy as np
import utils
import backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RelationNet_Ours(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, loss_type='mse'):
        super(RelationNet_Ours, self).__init__(model_func, n_way, n_support)
        self.loss_type = loss_type
        self.relation_module = RelationModule(self.feat_dim, 8, self.loss_type)
        self.feat_size = self.feat_dim.copy()[1:]
        self.dim = self.feat_dim[0]
        self.adaptive_module = AdaptiveNet(self.n_support, int(self.n_support / 2))
        self.adaptive_module.apply(weights_init)
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_support = z_support.view(self.n_way, self.n_support, *self.feat_dim)
        assert self.n_support > 1, '其他的情况没有考虑'
        z_proto = self.adaptive_module(z_support.transpose(1, 2).contiguous().view(-1, self.n_support, *self.feat_size))
        z_proto = z_proto.view(self.n_way, self.dim, *self.feat_size)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)
        return relations

    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature is True, '微调只能支持一定修订好的特征'
        full_n_support = self.n_support
        full_n_query = self.n_query
        relation_module_clone = RelationModule(self.feat_dim, 8, self.loss_type)
        relation_module_clone.load_state_dict(self.relation_module.state_dict())
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        set_optimizer = torch.optim.SGD(self.relation_module.parameters(),
                                        lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        self.n_support = 3
        self.n_query = 2
        z_support_cpu = z_support.data.cpu().numpy()
        for epoch in range(100):
            perm_id = np.random.permutation(full_n_support).tolist()
            sub_x = np.array([z_support_cpu[i, perm_id, :, :, :] for i in range(z_support.size(0))])
            sub_x = torch.Tensor(sub_x).to(device)
            if self.change_way:
                self.n_way = sub_x.size(0)
            set_optimizer.zero_grad()
            y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            scores = self.set_forward(sub_x, is_feature=True)
            if self.loss_type == 'mse':
                y_oh = utils.one_hot(y, self.n_way)
                y_oh = Variable(y_oh.to(device))
                loss = self.loss_fn(scores, y_oh)
            else:
                y = Variable(y.to(device))
                loss = self.loss_fn(scores, y)
            loss.backward()
            set_optimizer.step()
        self.n_support = full_n_support
        self.n_query = full_n_query

        # 这里为什么直接mean
        z_proto = z_support.view(self.n_way, self.n_support, *self.feat_dim).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)
        self.relation_module.load_state_dict(relation_module_clone.state_dict())
        return relations

    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        correct_this = float(top1_correct)
        count_this = len(y_query)
        acc = correct_this / count_this
        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = Variable(y_oh.to(device))
            return self.loss_fn(scores, y_oh), acc
        else:
            y = Variable(y.to(device))
            return self.loss_fn(scores, y), acc


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.conv_f = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.conv_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.conv_h = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#
#     def forward(self, x):
#         batch_size, c, h, w = x.size()
#         f = self.conv_f(x).view(batch_size, -1, h * w)
#         g = self.conv_g(x).view(batch_size, -1, h * w)
#         h = self.conv_h(x).view(batch_size, -1, h * w)
#         attention = F.softmax(torch.bmm(f.permute(0, 2, 1), g), dim=1)
#         out = torch.bmm(h, attention.permute(0, 2, 1))
#         out = out.view(batch_size, c, -1, w)
#         out = x + out
#         return out


    

class RelationConvBlock1(nn.Module):
    def __init__(self, indim, outdim, padding=0, num_heads=8):
        super(RelationConvBlock1, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # self.attention = SelfAttention(outdim)
        # self.parametrized_layers = [self.C, self.BN, self.relu, self.pool, self.attention]
        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]
        for layer in self.parametrized_layers:
            backbone.init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out
    
# class RelationConvBlock2(nn.Module):
#     def __init__(self, indim, outdim, padding=0, num_heads=8):
#         super(RelationConvBlock2, self).__init__()
#         self.indim = indim
#         self.outdim = outdim
#         self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
#         self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=outdim, num_heads=num_heads)
#         self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]
#         for layer in self.parametrized_layers:
#             backbone.init_layer(layer)
#         self.trunk = nn.Sequential(*self.parametrized_layers)
#
#     def forward(self, x):
#         out = self.trunk(x)
#         batch_size, channels, height, width = out.size()
#         out = out.view(batch_size, height * width, channels)
#         out = out.transpose(0, 1)
#         out, _ = self.multihead_attn(out, out, out)
#         out = out.transpose(0, 1)
#         out = out.reshape(batch_size, channels, height, width)
#         return out


class RelationModule(nn.Module):
    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(RelationModule, self).__init__()
        self.loss_type = loss_type
        if input_size[1] < 10 and input_size[2] < 10:
            padding = 1
        else:
            padding = 0
        self.layer1 = RelationConvBlock1(input_size[0] * 2, input_size[0], padding=padding)
        self.layer2 = RelationConvBlock1(input_size[0], input_size[0], padding=padding)
        # self.layer3 = RelationConvBlock2(input_size[0] * 2, input_size[0], padding=padding)
        # self.layer4 = RelationConvBlock2(input_size[0], input_size[0], padding=padding)
        # shrink_s函数计算经过两个连续的3x3滤波器、步长为2和填充padding的卷积层后的输出特征图的大小。
        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)
        self.fc1 = nn.Linear(input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.alpha = nn.Parameter(torch.Tensor([0.5]))  # 可学习的融合权重

    def forward(self, x):
        # 使用RelationConvBlock提取出来的特征用out表示
        out = self.layer1(x)
        out = self.layer2(out)
        # 使用RelationConvBlock2提取出来的特征用out2表示
        # out2 = self.layer3(x)
        # out2 = self.layer4(out2)
        # out = self.alpha * out1 + (1 - self.alpha) * out2  # 加权平均融合
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out
    

class AdaptiveNet(nn.Module):
    def __init__(self, indim, outdim):
        super(AdaptiveNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(indim, outdim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(outdim, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(outdim, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(1, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


# 将一个对象m的类作为输入，使用__class__属性获取该对象的类，然后使用__name__属性获取该类的类名。
# 将结果赋值给 classname 变量
def weights_init(m):
    classname = m.__class__.__name__
    # 对于卷积层，使用正态分布随机初始化其权重参数，同时根据该层的输入通道数和输出通道数进行缩放，
    # 以便更好地保持梯度的传播。同时，将偏置参数初始化为0。
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(1 / m.in_channels, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    # 对于批归一化层，将其权重参数初始化为1，将偏置参数初始化为0
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    # 对于全连接层，使用正态分布随机初始化其权重参数，初始化偏置参数为1。
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
