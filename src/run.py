import dotenv
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ml.test import test_binary_classification 
from data.generate import RandomLPDataset
from ml.models.s2v import Model

def train_net(net, criterion, optimizer, trainloader, num_epochs):
    losses  = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # x: input, y: output
            inputs = data['x']
            label  = Variable(data['y'].type(torch.LongTensor))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # Loss and running loss
            loss_val = float(loss.data[0])
            losses = np.append(losses, loss_val)
            running_loss += loss_val

            # Stats
            if i % 100 == 0:
                print('[%d, %5d] loss=%.7f' % (epoch+1, i+1, loss_val))
    print('Finished training')
    return losses

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
    Ts = [0, 1, 2, 3, 4]

    # Dataset params
    N  = 2000

    # Opt Params
    lr              = 0.001
    momentum        = 0.9
    weight_decay    = 0.01
    num_epochs      = 4

    # ----------------
    # Datasets
    # ----------------

    # Training and testing dataset
    trainset = RandomLPDataset(m=m, n=n, N=N, num_lps=1, test=False)
    testset  = RandomLPDataset(m=m, n=n, N=m, num_lps=1, test=True)

    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=1, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,
            batch_size=1, shuffle=False, num_workers=1)

    # ----------------
    # Train
    # ----------------

    results = {}

    for T in Ts:
        # init
        net = Model(m, n, p, T)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), 
                lr=lr, momentum=momentum, weight_decay=weight_decay)
        # train
        losses      = train_net(net, criterion, optimizer, trainloader, num_epochs)
        # test
        net.eval()
        acc         = test_binary_classification(testloader, net, verbose=True)
        # record
        results[T]  = [losses, acc]

    # -------------------
    # Plot
    # -------------------

    if figspath:

        lp_params   = '(m, n, p) = (%d, %d, %d)' % (m, n, p) 
        opt_params  = '(lr, momentum, weight_decay) = (%g, %g, %g)' % (lr, momentum, weight_decay) 
        comments    = 'T: number of s2v rounds, acc: test-accuracy'
        title = lp_params + "\n" + opt_params + "\n" + comments

        fig = plt.figure(1)
        fpath = os.path.join(figspath, 'loss.png')
        # plot
        for T in results.keys():
            x, acc = results[T]
            plt.plot(range(len(x)), x, alpha=0.15*(T+1))
        plt.legend(['T=%d, acc=%g' % (T, results[T][1]) for T in results.keys()], loc='upper right')
        # cosmetics
        plt.title(title)
        plt.xlabel('number of SGD steps')
        plt.ylabel('loss(yhat, y)')
        # save
        plt.tight_layout()
        plt.savefig(fpath)

        fig = plt.figure(2)
        fpath = os.path.join(figspath, 'log_loss.png')
        # plot
        for T in results.keys():
            x, acc = results[T]
            plt.plot(range(len(x)), x, alpha=0.15*(T+1))
            plt.yscale('log')
        plt.legend(['T=%d, acc=%g' % (T, results[T][1]) for T in results.keys()], loc='upper right')
        # cosmetics
        plt.title(title)
        plt.xlabel('number of SGD steps')
        plt.ylabel('log(loss(yhat, y))')
        # save
        plt.tight_layout()
        plt.savefig(fpath)


if __name__ == '__main__':
    main()

