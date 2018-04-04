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

def save(data, model, save_path):
    # Build filename
    tag   = data['params']['dataset']
    s     = json.dumps(data['params'], sort_keys=True)
    stamp = hashlib.sha1(s.encode()).hexdigest()
    stamp = stamp[0:11] # Keep only 10 chars in stamp
    # Save data
    fname = '%s_params_%s.json' % (tag, stamp)
    fpath = os.path.join(save_path, fname)
    with open(fpath, 'w') as outfile:
        json.dump(data, outfile)
    # Save model
    fname = '%s_model_%s.json' % (tag, stamp)
    fpath = os.path.join(save_path, fname)
    torch.save(model.state_dict(), fpath)
    return

def dataset_factory(dataset, seed='1111'):
    if dataset == 'plnn':
        trainset = DatasetPLNN(num_lps=1, test=False, seed=seed)
        testset  = DatasetPLNN(num_lps=1, test=True , seed=seed)
    elif dataset == 'randomlp':
        m, n     = (10, 5)
        trainset = RandomLPDataset(m=m, n=n, num_lps=1, test=False, seed=seed)
        testset  = RandomLPDataset(m=m, n=n, num_lps=1, test=True , seed=seed)
    else:
        raise ValueError('Dataset not recognised')
    return trainset, testset

def run_experiment(params):

    # params
    dataset = params['dataset']
    seed    = params['seed']
    bs      = params['batch_size']
    t       = params['rounds_s2v']
    lr      = params['learning_rate']
    mtm     = params['momentum']
    wd      = params['weight_decay']
    epochs  = params['num_epochs']
    cuda    = params['cuda']

    delta   = params['delta'] # p/(n+1)

    # datasets
    trainset, testset = dataset_factory(dataset, seed)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True,  num_workers=1)
    testloader  = DataLoader(testset,  batch_size =1, shuffle=False, num_workers=1)

    # Get params
    lp_params  = trainset.get_lp_params()
    assert(len(lp_params) == 1)
    m       = lp_params[0]['m'] 
    n       = lp_params[0]['n'] 
    p       = int(float(n+1) * delta)

    # model
    model         = Model(m, n, p, t, cuda=cuda)
    if cuda:
        model     = model.cuda()

    # optimization
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.SGD(model.parameters(), lr=lr, momentum=mtm, weight_decay=wd)
    losses      = train_net(model, criterion, optimizer, trainloader, 
                            epochs, bs, testloader, verbose=True, cuda=cuda)

    # test
    acc         = test_binary_classification(testloader, model, verbose=True, cuda=cuda)

    # record
    out             = {}
    out['losses']   = losses
    out['lps']      = trainset.get_lp_params()
    out['acc']      = acc

    d               = {}
    d['params']     = {k:v for k,v in params.items()}
    d['out']        = out

    return d, model

def wrap_params(ds, seed, bs, t, lr, mtm, wd, epochs, delta, cuda):
    params = {}
    params['dataset']       = ds
    params['seed']          = seed
    params['batch_size']    = bs
    params['rounds_s2v']    = t
    params['learning_rate'] = lr
    params['momentum']      = mtm
    params['weight_decay']  = wd
    params['num_epochs']    = epochs
    params['delta']         = delta
    params['cuda']          = cuda
    return params

def benchmark_single_randomlp(save_path, cuda=False):

    # Params
    dataset = 'randomlp'
    epochs  = 250
    seeds   = [0, 3]
    bss     = [1, 5]    	# Batch size
    ts      = [0, 2, 4]     # Rounds of s2v
    lrs     = [0.01, 0.001] # Learning rates
    mtms    = [0.9]         # Momentums
    wds     = [0]           # Weight decays

    delta = 1.0

    # Benchmark 
    for seed, bs, t, lr, mtm, wd in itertools.product(seeds, bss, ts, lrs, mtms, wds):
        params = wrap_params(dataset, seed, bs, t, lr, mtm, wd, epochs, delta, cuda)
        print(','.join(['{0}={1}'.format(k,v) for k,v in params.items()]))
        exp_params, model = run_experiment(params)
        save(exp_params, model, save_path)
    return

def benchmark_single_plnn(save_path, cuda=False):

    # Params
    dataset = 'plnn'
    epochs  = 250
    seeds   = [3]
    bss     = [1, 5]        # Batch size
    ts      = [2]           # Rounds of s2v
    lrs     = [0.01]        # Learning rates
    mtms    = [0.9]         # Momentums
    wds     = [0]           # Weight decays

    delta = 1.0

    # Benchmark 
    for seed, bs, t, lr, mtm, wd in itertools.product(seeds, bss, ts, lrs, mtms, wds):
        params = wrap_params(dataset, seed, bs, t, lr, mtm, wd, epochs, delta, cuda)
        print(','.join(['{0}={1}'.format(k,v) for k,v in params.items()]))
        exp_params, model = run_experiment(params)
        save(exp_params, model, save_path)
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
        action='store_true', default=False, help='Enable cuda')

    args = parser.parse_args()

    # cuda
    cuda = False
    if args.cuda:
        if torch.cuda.is_available():
            cuda = True
        else:
            print('CUDA is not available')

    # randomlp
    if args.randomlp:
        benchmark_single_randomlp(outpath, cuda)

    # plnn
    if args.plnn:
        benchmark_single_plnn(outpath, cuda)

