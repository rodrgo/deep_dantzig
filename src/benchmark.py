import dotenv
import hashlib
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import sys
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
from ml.models.s2v import Model
from utils.plot import s2v_loss_grid

from data.randomlp_dataset import RandomLPDataset
from data.plnn_dataset import DatasetPLNN

def save(save_path, stype, res, model=None):
    # Build filename
    dset  = res['dataset']
    s     = json.dumps(res, sort_keys=True)
    stamp = hashlib.sha1(s.encode()).hexdigest()
    stamp = stamp[0:11] # Keep only 10 chars in stamp
    # Save data
    if not res is None:
        fname = '%s_%s_res_%s.json' % (stype, dset, stamp)
        fpath = os.path.join(save_path, fname)
        with open(fpath, 'w') as outfile:
            json.dump(res, outfile)
    # Save model
    if not model is None:
        fname = '%s_%s_model_%s.json' % (stype, dset, stamp)
        fpath = os.path.join(save_path, fname)
        torch.save(model.state_dict(), fpath)
    return

def dataset_factory(dataset, seed):
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

def run_experiment(params, dataset, seed=1111, cuda=False, tag=None):

    # params
    epochs  = params['num_epochs']
    bs      = params['batch_size']
    t       = params['rounds_s2v']
    lr      = params['learning_rate']
    mtm     = params['momentum']
    wd      = params['weight_decay']
    p       = params['p']

    # datasets
    trainset, testset = dataset_factory(dataset, seed)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=1,  shuffle=False)

    print('%d seed LPs in trainset' % (len(trainset.lp_dirs)))
    print('%d problems in trainset' % (len(trainset)))

    # model
    model       = Model(p, t, cuda=cuda)
    if cuda:
        model   = model.cuda()

    # optimization
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.SGD(model.parameters(), lr=lr, momentum=mtm, weight_decay=wd)
    results     = train_net(model, criterion, optimizer, trainloader, 
                            epochs, bs, testloader, verbose=True, cuda=cuda)

    # record
    out             = {}
    out['lps']      = trainset.get_lp_params()
    out['results']  = results

    d               = {}
    d['params']     = {k:v for k,v in params.items()}
    d['out']        = out
    d['dataset']    = dataset
    d['seed']       = seed
    d['cuda']       = cuda
    d['tag']        = tag

    return d, model

def wrap_params(ep, bs, t, lr, mtm, wd, p):
    params = {}
    params['num_epochs']    = ep
    params['batch_size']    = bs
    params['rounds_s2v']    = t
    params['learning_rate'] = lr
    params['momentum']      = mtm
    params['weight_decay']  = wd
    params['p']             = p
    return params

def run_benchmark(save_path, dataset, benchmark_params, cuda=False, tag=None):

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
    for seed, ep, bs, t, lr, mtm, wd, p in itertools.product(sds, eps, bss, ts, lrs, mtms, wds, ps):
        params = wrap_params(ep, bs, t, lr, mtm, wd, p)
        print(','.join(['{0}={1}'.format(k,v) for k,v in params.items()]))
        res, model = run_experiment(params, dataset, seed, cuda, tag)
        save(save_path, 'benchmark', res, model)
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
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            #torch.cuda.set_device(device)
            cuda = True
        else:
            print('CUDA is not available')

    # randomlp
    if args.randomlp:
        bp = {}
        bp['epochs']            = [150]
        bp['seeds']             = [0, 3]
        bp['batch_sizes']       = [1]
        bp['rounds_s2v']        = [1, 2, 4]
        bp['learning_rates']    = [0.01, 0.001]
        bp['momentums']         = [0.9]
        bp['weight_decays']     = [0]
        bp['ps']                = [12]
        run_benchmark(outpath, 'randomlp', bp, cuda, tag)

    # plnn
    if args.plnn:
        bp = {}
        bp['epochs']            = [1000]
        bp['seeds']             = [3, 4]
        bp['batch_sizes']       = [1]
        bp['rounds_s2v']        = [1, 2, 3]
        bp['learning_rates']    = [0.01]
        bp['momentums']         = [0.9]
        bp['weight_decays']     = [0]
        bp['ps']                = [35, 40, 45]
        run_benchmark(outpath, 'plnn', bp, cuda, tag)

