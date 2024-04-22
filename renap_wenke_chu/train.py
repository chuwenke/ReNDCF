import os.path
import time
import backbone
import numpy as np
from methods.relationnet_ours import RelationNet_Ours
import configs
from data.datamgr import SetDataManager
from io_utils import parse_args, model_dict, get_resume_file
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='./logs/nets')


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('未知的优化器，请自行定义')
    max_acc = 0
    for epoch in range(start_epoch, stop_epoch):
        t1 = time.time()
        model.train()
        model.train_loop(epoch, base_loader, optimizer)
        model.eval()
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        acc = model.test_loop(val_loader, epoch)
        t2 = time.time()
        print('epoch:{}/{}  test acc:{}  time:{}'.format(epoch, stop_epoch - start_epoch, acc, t2 - t1))
        if acc > max_acc:
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            print("best model saved!  best acc:{}".format(acc))
        if epoch % params.save_freq == 0 or epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            print('model saved!')
    return model


if __name__ == '__main__':
    params = parse_args('train')
    np.random.seed(10)
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot只支持没有augmentation的Conv4，要不带--train_aug参数'
        params.model = 'Conv4S'
    optimization = 'Adam'
    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            print("这里没考虑，需要补充")
        else:
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600

    if params.method in ['baseline', 'baseline++']:
        print("这里没考虑，需要补充")
    elif params.method in ['relationnet_ours', 'protonet_ours', 'protonet', 'matchingnet', 'relationnet',
                           'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        if params.method == 'protonet':
            print("不考虑")
        elif params.method == 'matchingnet':
            print("不考虑")
        elif params.method in ['relationnet_ours', 'relationnet', 'relationnet_softmax']:
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
                print("不考虑relationnet")
            elif params.method == 'relationnet_ours':
                model = RelationNet_Ours(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            print("还没考虑maml和maml_approx方法")
    else:
        raise ValueError('未知方法')
    model.to(device)
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        print("没考虑maml情况")
    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
    elif params.warmup:
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('没有warm_up文件')
    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
