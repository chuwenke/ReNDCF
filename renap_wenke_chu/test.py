import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
import configs
from methods.relationnet_ours import RelationNet_Ours
import data.feature_loader as feat_loader
import backbone
from io_utils import parse_args, model_dict, get_assigned_file, get_best_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def feature_evalution(cl_data_file, model, n_way=5, n_support=5, n_query=10, adaptation=False):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
    z_all = torch.from_numpy(np.array(z_all))
    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    params = parse_args('test')
    iter_num = 600
    few_shot_params = dict(
        n_way=params.test_n_way,
        n_support=params.n_shot
    )
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot支持没有train_aug的Conv4'
        params.model = 'Conv4S'

    if params.method in ['relationnet_ours', 'relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse'
        if params.method == 'relationnet':
            print('没考虑relationnet')
        elif params.method == 'relationnet_ours':
            model = RelationNet_Ours(feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method in ['maml', 'maml_approx']:
        print('没考虑maml')
    else:
        raise ValueError('未知方法')
    model.to(device)

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile, map_location='cpu')
            model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    save_file = checkpoint_dir.split('/')[-2] + '_' + checkpoint_dir.split('/')[-1] + '.txt'
    try:
        os.stat('./t-test')
    except:
        os.makedirs('./t-test')
    save_file = './t-test/' + save_file

    try:
        os.remove(save_file)
    except:
        pass

    for k in tqdm(range(30)):
        acc_all = []
        if params.method in ['maml', 'maml_approx']:
            print('maml不考虑')
        else:
            novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split_str + ".hdf5")
            cl_data_file = feat_loader.init_loader(novel_file)
            for i in range(iter_num):
                acc = feature_evalution(cl_data_file, model, n_query=10,
                                        adaptation=params.adaptation, **few_shot_params)
                acc_all.append(acc)
            acc_all = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)
            print('iter_num:%d  test acc=%4.2f%% +- %4.2f%%' %
                  (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        with open(save_file, 'a') as f:
            f.write('%4.2f \n' % acc_mean)
        with open('./record/results.txt', 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++']:
                print('不考虑')
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
                    params.dataset, split_str, params.model, params.method, aug_str,
                    params.n_shot, params.train_n_way, params.test_n_way
                )
            acc_str = 'total test item:%d  test acc=%4.2f%% +- %4.2f%%' % (
                iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)
            )
            f.write('time:%s,  Setting:%s,  Acc:%s \n' % (timestamp, exp_setting, acc_str))



