import numpy as np
import os
import glob
import argparse

import numpy as np
import os
import glob
import argparse


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='miniImagenet', help='CUB/miniImagenet/cifar')
    parser.add_argument('--method', default='WideResNet28_10_S2M2_R', help='WideResNet28_10_S2M2_R/Conv4_baseline_aug/Conv6_baseline_aug/ResNet10_baseline_aug/ResNet18_baseline_aug/ResNet34_baseline_aug')

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes')  # make it larger than the maximum label value in base class
        parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=400, type=int,
                            help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')
        parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
        parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
        parser.add_argument('--test_batch_size', default=2, type=int, help='batch size ')
        parser.add_argument('--alpha', default=2.0, type=int, help='for S2M2 training ')
    elif script == 'test':
        parser.add_argument('--shot', default=1, type=int, help='n_shot')
        parser.add_argument('--way', default=5, type=int, help='n_ways')
        parser.add_argument('--query', default=15, type=int, help='n_queries')
        
        parser.add_argument('--beta', default=0.5, type=float, help='beta')
        parser.add_argument('--split', default='novel', type=str, help='split') 
        parser.add_argument('--sp', default=150, type=int, help='g_samples')
        parser.add_argument('--st', default=5, type=int, help='st')
        parser.add_argument('--end', default=16, type=int, help='end')
        parser.add_argument('--step', default=5, type=int, help='step')
        parser.add_argument('--run', default=10000, type=int, help='n_runs')
        parser.add_argument('--k', default=5, type=int, help='k')
        parser.add_argument('--gam', default=2, type=int, help='k')
        
    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
