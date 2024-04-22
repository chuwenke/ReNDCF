import numpy as np
import torch


def one_hot(y, num_class):
    return torch.zeros(len(y), num_class).scatter_(1, torch.as_tensor(y, dtype=torch.int64).unsqueeze(1), 1)


# 这个函数的作用是计算给定数据集的簇间距离指数(DBI)，用于衡量聚类结果的质量。
# def DBindex(cl_data_file):
#     class_list = cl_data_file.keys()
#     cl_num = len(class_list)
#     cl_means = []
#     stds = []
#     DBs = []
#     for cl in class_list:
#         cl_means.append(np.mean(cl_data_file[cl], axis=0))
#         a = np.square(cl_data_file[cl] - cl_means[-1])
#         stds.append(np.sqrt(np.mean(np.sum(a, axis=1))))
#     mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
#     nu_j = np.transpose(mu_i, (1, 0, 2))
#     mdists = np.sqrt(np.sum(np.square(mu_i - nu_j), axis=2))
#     for i in range(cl_num):
#         DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) is j != i]))
#     return np.mean(DBs)


# 这个函数的作用是计算给定数据集的稀疏度，即每个类别的数据点中非零元素所占的平均比例，
# 稀疏度是指数据点中非零元素所占的比例，用于衡量数据的稀疏性
# def sparsity(cl_data_file):
#     class_list = cl_data_file.keys()
#     cl_sparsity = []
#     for cl in class_list:
#         cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))
#     return np.mean(cl_sparsity)
