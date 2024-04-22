import argparse
import glob
import os
import numpy as np
import backbone

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6
)


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % script)
    parser.add_argument('--model', default='Conv4', help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--dataset', default='CUB', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--method', default='relationnet_ours', help='')
    parser.add_argument('--train_aug', action='store_true', help='表明训练期间是否进行数据扩充')
    parser.add_argument('--n_shot', default=5, type=int, help='每个类中的已标记的数据，与n_support相同')
    parser.add_argument('--train_n_way', default=5, type=int, help='训练时要分类的类数')
    parser.add_argument('--test_n_way', default=5, type=int, help='测试（或者验证）是要分类的类数')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    if script == 'train':
        parser.add_argument('--save_freq', default=50, type=int, help='保存的频率')
        parser.add_argument('--start_epoch', default=0, type=int, help='开始的epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int, help='最大epoch')
        parser.add_argument('--resume', action='store_true', help='用最大的epoch开始从先前的训练模型中继续')
        parser.add_argument('--warmup', action='store_true', help='从baseline继续，如果resume是true那就忽略不计')

    elif script == 'save_features':
        parser.add_argument('--split', default='novel', help='在这里默认测试的是novel的准确度，但也可以选择测试base和val的准确度')
        parser.add_argument('--save_iter', default=-1, type=int, help='在训练第x个epoch时候保存参数，如果参数是-1则使用best model')

    elif script == 'test':
        parser.add_argument('--split', default='novel', help='默认是novel类')
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='在训练第x个epoch时候保存参数，如果参数是-1则使用best model')
        parser.add_argument('--adaptation', action='store_true', help='在test时候更进一步的自适应')

    else:
        raise ValueError('未知的script')

    return parser.parse_args()


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
