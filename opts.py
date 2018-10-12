from __future__ import print_function
import argparse
import os


def argparser():
    P = argparse.ArgumentParser(description='Train network script')
    P.add_argument('--seed', type=int, default=0, help='maunlly set RNG seed')

    P.add_argument('--nGpu', type=int, default=1, help='number of gpu(s) to use')
    P.add_argument('--snapshot', type=int, default=3, help='save a snapshot every n epoch(s)')
    P.add_argument('--epochs', type=int, default=300, help='Number of total epochs to run')
    P.add_argument('--workers', type=int, default=16, help='number of data loader threads')
    P.add_argument('--baseline', action='store_true', help='train on f7n as baseline')

    # for a single GPU.
    P.add_argument('--train-batch', type=int, default=3, help='minibatch size')
    P.add_argument('--val-batch', type=int, default=2, help='minibatch size')

    P.add_argument('-c', '--checkpoint', type=str, default='work_space/baseline', help='model save path')
    P.add_argument('--resume_big', type=str, default='work_space/baseline_1201/net_deploy_nodepthwise.pth.tar', help='resume from lasteset saved checkpoints')
    P.add_argument('--resume_small', type=str, default='work_space/baseline_145/model_145.pth.tar', help='resume from lasteset saved checkpoints')
    P.add_argument('--resume', type=str, default='work_space/stilling/checkpoint_2.pth.tar', help='resume from lasteset saved checkpoints')
    
    P.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    P.add_argument('--momentum', type=float, default=0.9, help='momentum')
    P.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    P.add_argument('--netType', type=str, default='fan', help='options: fan')
    #P.add_argument(
    #   '--schedule', type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='adjust lr at this epoch')
    P.add_argument('--enlarge_ratio', type=float, default=3, help='bbox enlarging ratio')
    P.add_argument('--fix_mean', type=bool, default=True, help = 'whether use fix mean 0.5')
    
    #P.add_argument(
    #    '--schedule', type=int, nargs="+", default=[30, 60, 110, 130, 145, 170, 220, 250, 270, 300, 330, 360], help='adjust lr at this epoch')
    P.add_argument(
        '--schedule', type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40, 45], help='adjust lr at this epoch')
    P.add_argument('--gamma', type=float, default=0.6, help='lr decay')
    P.add_argument('--sparse_gamma', type=float, default=0.6, help='sparse_gamma decay')
    P.add_argument('--retrain', type=str, default='', help='path to model to retrain with')
    P.add_argument('--scratch', action='store_true', help='train the model from scratch')
    P.add_argument('--optimState', type=str, default='', help='path to optimState to reload from')

    # some data argument factor
    P.add_argument('--shift-std', type=float, default=0.05, help='shift std')
    P.add_argument('--scale-std', type=float, default=0.05, help='scaling std')
    P.add_argument('--rot-std', type=float, default=5, help='rotation std(in degrees)')

    P.add_argument('--shift-std_negative', type=float, default=0.05, help='shift std')
    P.add_argument('--scale-std_negative', type=float, default=0.05, help='scaling std')
    P.add_argument('--rot-std_negative', type=float, default=5, help='rotation std(in degrees)')
    P.add_argument('--is_gray', type=bool, default=False, help='grayscale image')
    P.add_argument('--rot-std-pose', type=float, default=40, help='pose rotation std(in degrees)')
    P.add_argument('-e', '--evaluation', action='store_true', help='show intermediate results')

    P.add_argument('--debug', action='store_true', help='show intermediate results')
    P.add_argument('--flip', action='store_true', help='Flip input image')
    P.add_argument('--score_trainset', type=str, default='datasets/scorelist_train_18_05_06.txt', help='path to score_trainset')
    P.add_argument('--score_testset', type=str, default='datasets/scorelist_valid_18_05_06.txt', help='path to score_testnset')
    P.add_argument('--liveness_trainset', type=str, default='datasets/trainlist.txt', help='path to trainlist')
    P.add_argument('--liveness_testset', type=str, default='datasets/trainlist.txt', help='path to trainlist')
    P.add_argument('--trainset', type=str, default='datasets/trainlist.txt', help='path to trainlist')
    P.add_argument('--testset', type=str, default='datasets/trainlist.txt', help='path to trainlist')
    P.add_argument(
        '--ratio', type=int, nargs="+", default=[1, 1], help='the sample ratio')


    P.add_argument('--model', type=str, default='models.liveness_base_model', help='path to mode file')
    P.add_argument('--model_big', type=str, default='models.model_1201', help='path to model file')
    P.add_argument('--model_small', type=str, default='models.model_145', help='path to model file')

    P.add_argument('--input_size', type=int, default=64, help='data input')
    P.add_argument('--test_score_res', type=str, default='score_res.txt', help='score_res')

    P.add_argument('--ignore_pose', type=int, default = 0, help='ignore pose')
    P.add_argument('--ignore_score', type=int, default = 0, help='ignore score')        

    P.add_argument(
        '--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    args = P.parse_args()

    return args
