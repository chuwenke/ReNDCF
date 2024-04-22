import os
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    # 用来传入更多的值来解包
    @abstractmethod
    # 通过它可以直接求loss和acc？
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.to(device))
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = np.repeat(range(self.n_way), self.n_query)

        # data是set_forward中的函数，引入@abstractmethod方法，
        # 这个data.topk是什么函数，应该是用来求最大分数的index的函数，并得到标签
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)

        # 为什么？
        topk_ind = topk_labels.cpu().numpy()

        # 预测正确的个数
        top1_correct = np.sum(topk_ind[:, 0] == y_query)

        loss_fn = nn.MSELoss()

        # utils还没定义，其中的one_hot还没定义,它的作用应该是产生一个长n_way的0，1向量
        y_oh = utils.one_hot(y, self.n_way)
        y_oh = Variable(y_oh.to(device))
        loss = loss_fn(scores, y_oh)
        return float(top1_correct), len(y_query), loss

    def train_loop(self, epoch, train_loader, optimizer):
        avg_loss = 0
        avg_acc = 0
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss, acc = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()

            # 这个loss是在set_forward_loss里面求的，他可能带有item方法
            avg_loss = avg_loss + loss.item()
            avg_acc = avg_acc + acc
            if i % 10 == 0:
                pass
        train_writer = SummaryWriter(log_dir=os.path.join('./tensorboard', 'train'))
        train_writer.add_scalar('train_accuracy', avg_acc / iter_num, epoch)
        train_writer.add_scalar('train_loss', avg_loss / iter_num, epoch)
        train_writer.close()
        print('total item:%d  train acc:%.6f  train loss:%.6f' % (iter_num, avg_acc / iter_num, avg_loss / iter_num))

    def test_loop(self, test_loader, epoch, record=None):
        correct = 0
        count = 0
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            # correct_this是预测正确的个数，count_this是query的个数
            correct_this, count_this, loss = self.correct(x)
            acc_all.append(correct_this / count_this)
            avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        # 没有用上啊
        acc_std = np.std(acc_all)
        test_writer = SummaryWriter(log_dir=os.path.join('./tensorboard', 'test'))
        test_writer.add_scalar('test_accuracy', acc_mean, epoch)
        test_writer.add_scalar('test_loss', avg_loss / iter_num, epoch)
        test_writer.close()
        print('total item:%d  test acc:%.6f  test loss:%.6f' % (iter_num, acc_mean, avg_loss / iter_num))
        return acc_mean

    # 进一步适配，默认是修复功能并训练新的softmax分类器
    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature is True, '特征在进一步适应中已修复'
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.to(device))
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.to(device)
        set_optimizer = torch.optim.SGD(linear_clf.parameters(),
                                        lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)
        # batch_size参数可调
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            # 随机排序
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).to(device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores
