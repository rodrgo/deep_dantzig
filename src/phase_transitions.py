import dotenv
import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from ml.train import train_net
from ml.test import test_binary_classification 
from ml.models.s2v import Model
from utils.plot import s2v_loss_grid

from data.randomlp_dataset import RandomLPDataset
from data.plnn_dataset import DatasetPLNN

from benchmark import *

def compute_phaseTransitions(save_path, dataset, benchmark_params, cuda=False, tag=None):

    # Params
    sds     = benchmark_params['seeds']
    eps     = benchmark_params['epochs']
    bss     = benchmark_params['batch_sizes']
    ts      = benchmark_params['rounds_s2v']
    lrs     = benchmark_params['learning_rates']
    mtms    = benchmark_params['momentums']
    wds     = benchmark_params['weight_decays']
    ps      = benchmark_params['ps']

    # Benchmark 
    for seed, ep, bs, t, lr, mtm, wd, p0 in itertools.product(sds, eps, bss, ts, lrs, mtms, wds, ps):
        acc = 1.0
        accs = {}
        losses = {}
        p = p0
        while p > 1 and acc > 0.5:
            params  = wrap_params(ep, bs, t, lr, mtm, wd, p)
            print(','.join(['{0}={1}'.format(k,v) for k,v in params.items()]))
            res, model = run_experiment(params, dataset, seed, cuda, tag)
            accs[p]   = res['out']['acc']
            losses[p] = json.loads(res['out']['losses']['total'])[-1]
            p = p - 1
        pt = {}
        pt['params']    = wrap_params(ep, bs, t, lr, mtm, wd, p0)
        pt['out']       = {'accs': accs, 'losses': losses}
        pt['dataset']   = dataset
        pt['seed']      = seed
        pt['cuda']      = cuda
        pt['tag']       = tag
        save(save_path, 'pt', pt, None)
    return

if __name__ == '__main__':
    import argparse

    # Build relevant paths
    dotenv.load_dotenv(dotenv.find_dotenv())
    root     = os.environ.get('ROOT')
    outpath  = os.path.join(root, 'data/output')

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomlp', 
        action='store_true', help='Test on single random LP')
    parser.add_argument('--plnn', 
        action='store_true', help='Test on single PLNN problem')
    parser.add_argument('--cuda',
       action='store_true', help='Enable cuda')
    parser.add_argument('--device',
       default=0, help='Enable cuda')
    parser.add_argument('--tag', '-t', 
       default=None, help='Experiment tags')

    args = parser.parse_args()

    # tag
    tag = args.tag

    # cuda
    cuda = False
    if args.cuda:
        device = int(args.device)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            cuda = True
        else:
            print('CUDA is not available')

    # randomlp
    if args.randomlp:
        bp = {}
        bp['epochs']            = [250]
        bp['seeds']             = [0]
        bp['batch_sizes']       = [1]
        bp['rounds_s2v']        = [1, 2, 3, 4]
        bp['learning_rates']    = [0.01]
        bp['momentums']         = [0.9]
        bp['weight_decays']     = [0]
        bp['ps']                = [13]
        compute_phaseTransitions(outpath, 'randomlp', bp, cuda, tag)

    # plnn
    if args.plnn:
        phaseTransitions_single_plnn(outpath, cuda, tag)

