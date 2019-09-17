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

def run_experiment_batch(dataset, graph, elem_type, 
    num_elems, p, rounds_s2v, epochs, batch_size, learning_rate, 
    momentum, weight_decay, seed, cuda=True, tag=None):

    # datasets
    trainset = DatasetPLNN(dataset, graph, num_elems, elem_type, seed, test=False)
    testset  = DatasetPLNN(dataset, graph, num_elems, elem_type, seed, test=True)

    trainloader = DataLoader(trainset, batch_size=batch_size,  shuffle=True)
    testloader  = DataLoader(testset,  batch_size=batch_size,  shuffle=False)

    # model
    model       = Model(graph, p, rounds_s2v, cuda)
    if cuda:
        model   = model.cuda()

    # optimization
    '''
    Class index `(0 to C-1, where C = number of classes)`
    reduce=True and size_average=False (Losses summed for each minibatch):
        \ell(x, y) = \sum_{n=1}^N l_n
    weight=[num_pos/num_total, num_neg/num_total]
    '''

    if cuda:
        weight = Variable(torch.cuda.FloatTensor(trainset.weight))
    else:
        weight = Variable(torch.FloatTensor(trainset.weight))

    criterion = nn.NLLLoss(weight=weight, size_average=False, reduce=True)
    optimizer = optim.SGD(model.parameters(), 
        lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    #optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    results   = train_net(model, criterion, optimizer, trainloader, 
        testloader, epochs, batch_size, cuda=cuda)

    # record
    out             = {}
    out['lps']      = trainset.get_source_dir()
    out['results']  = results

    d               = {}
    d['params']     = {k:v for k,v in params.items()}
    d['out']        = out
    d['dataset']    = dataset
    d['seed']       = seed
    d['cuda']       = cuda
    d['tag']        = tag

    return d, model

def run_benchmark(params, save_path, cuda=False, tag=None):
    # params is a dictionary of key-list pairs
    assert(all([type(v) is list for k,v in params.items()]))
    params_keys = params.keys()
    for params_values in itertools.product(*[params[k] for k in params_keys]):
        params_batch = {k:v for k,v in zip(params_keys, params_values)}
        print(','.join(['{0}={1}'.format(k,v) for k,v in params_batch.items()]))
        res, model = run_experiment_batch(**params_batch, cuda=cuda, tag=tag)
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

    parser.add_argument('--dataset', 
        default='mnist', help='Chooses dataset (mnist, plnn, randomlp)')
    parser.add_argument('--graph', 
        default='bipartite', help='Chooses graph structure (complete or bipartite)')
    parser.add_argument('--tag', '-t', 
       default=None, help='Experiment tags')

    # CUDA PARAMS

    parser.add_argument('--cuda',
        action='store_true', default=True, help='Enable cuda')
    parser.add_argument('--device',
       default=1, help='Enable cuda')


    args = parser.parse_args()

    dataset = args.dataset
    graph   = args.graph
    tag     = args.tag

    assert(dataset in ['mnist', 'plnn'])
    assert(graph   in ['complete', 'bipartite'])

    print('Running:\n\tdataset=%s\n\tgraph=%s\n\ttag=%s' % (dataset, graph, tag))

    # cuda
    cuda = False
    if args.cuda:
        device = int(args.device)
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            cuda = True
        else:
            print('CUDA is not available')

    params = {}
    params['seed']               = [3]

    # dataset
    params['dataset']            = [dataset]
    params['graph']              = [graph]
    params['elem_type']          = ['lp']
    params['num_elems']          = [None]
    params['num_elems']          = [10]

    # model
    params['p']                  = [40]
    params['rounds_s2v']         = [3]

    # optim
    params['epochs']             = [5000]
    params['batch_size']         = [1]
    params['learning_rate']      = [0.001]
    params['momentum']           = [0.9]
    params['weight_decay']       = [0]

    run_benchmark(params, outpath, cuda, tag)

