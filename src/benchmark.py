import dotenv
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
from data.generate import RandomLPDataset

def main():

    # Build relevant paths
    dotenv.load_dotenv(dotenv.find_dotenv())
    root     = os.environ.get('ROOT')
    figspath = os.path.join(root, 'figs')

    # ----------------
    # params
    # ----------------

    # Problem Params
    n  = 5 
    m  = 10
    p  = 12

    # Dataset params
    N  = 2000

    # Opt Params
    num_epochs      = 5

    # Varying Params
    seeds   = [i for i in range(2)]
    bss     = [1, 5, 10]            # Batch size
    ts      = [0, 2, 4]             # Rounds of s2v
    lrs     = [0.01, 0.001]         # Learning rates
    mtms    = [0.9]                 # Momentums
    wds     = [0]                   # Weight decays

    # ----------------
    # Benchmark 
    # ----------------

    results = []
    for seed in seeds:
        for bs, t, lr, mtm, wd in itertools.product(bss, ts, lrs, mtms, wds):
            print('Starting seed=%d, bs=%d, T=%d, lr=%g, mtm=%g, wd=%g' % (seed, bs, t, lr, mtm, wd))
            # Datasets
            trainset    = RandomLPDataset(m=m, n=n, N=N, num_lps=1, test=False, seed=seed)
            testset     = RandomLPDataset(m=m, n=n, N=m, num_lps=1, test=True , seed=seed)

            trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)
            testloader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

            # init
            net         = Model(m, n, p, t)
            criterion   = nn.CrossEntropyLoss()
            optimizer   = optim.SGD(net.parameters(), lr=lr, momentum=mtm, weight_decay=wd)
            # train
            losses      = train_net(net, criterion, optimizer, trainloader, num_epochs, bs)
            # test
            net.eval()
            acc         = test_binary_classification(testloader, net, verbose=True)

            # record
            d           = {}
            d['lp']     = {'m': m, 'n': n, 'p': p}
            d['seed']   = seed 
            d['bs']     = bs
            d['T']      = t
            d['lr']     = lr
            d['mtm']    = mtm
            d['wd']     = wd
            d['loss']   = pd.Series(losses).to_json(orient='values')
            d['acc']    = acc

            # Append results
            results.append(d)

    with open('results/benchmark.json', 'w') as outfile:
        json.dump(results, outfile)

if __name__ == '__main__':
    main()

