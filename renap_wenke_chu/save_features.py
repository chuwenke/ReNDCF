import os
import h5py
import torch
from torch.autograd import Variable
from tqdm import tqdm
import backbone
import configs
from io_utils import parse_args, get_assigned_file, get_best_file, model_dict
from data.datamgr import SimpleDataManager

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    print('total:{}'.format(len(data_loader)))
    for i, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        x = x.to(device)
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


if __name__ == '__main__':
    # 这个参数没写
    params = parse_args('save_features')
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml方法不支持save_feature'
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot只支持没有train_aug'
        params.model = 'Conv4S'
    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == 'cross_char':
        print('没有考虑这种数据集')
    else:
        loadfile = configs.data_dir[params.dataset] + split + '.json'
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               split + "_" + str(params.save_iter) + ".hdf5")
    else:
        modelfile = get_best_file(checkpoint_dir)
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    if params.method in ['relationnet_ours', 'relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            model = backbone.Conv4NP()
        elif params.model == 'Conv6':
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S':
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model](flatten=False)
    elif params.method in ['maml', 'maml_approx']:
        raise ValueError('maml不支持')
    else:
        model = model_dict[params.model]()

    model.to(device)
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    model.load_state_dict(state)
    model.eval()
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    save_features(model, data_loader, outfile)



