from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np
from ml.types import LongTensor
from ml.test import get_accuracy

def batched(data, batch_size):
    batch   = 0
    for batch in range(batch_size):
        dp = {}
        dp['x']      = {}
        dp['x']['A'] = data['lp']['A'][batch, :, :].unsqueeze(0)
        dp['x']['b'] = data['lp']['b'][batch, :].unsqueeze(0)
        dp['x']['c'] = data['lp']['c'][batch, :].unsqueeze(0)
        dp['x']['node_features']  = data['node_features']
        dp['y']  = data['node_labels'].squeeze(0)
        yield dp

def total_loss(testloader, model, criterion, cuda=False):
    # Compute total loss
    if testloader is None:
        total_loss = None 
    else:
        total_loss = 0.0
        model.require_grads(False)
        for i, data in enumerate(testloader, 0):
            for dp in batched(data, 1):
                # x: input, y: output
                x           = dp['x']
                y           = Variable(LongTensor(dp['y'], cuda=cuda))
                # forward + backward + optimize
                fx          = model(x)
                loss        = criterion(fx, y)
                total_loss  += loss.data[0] 
        model.require_grads(True)
    return total_loss

def train_net(model, criterion, optimizer, trainloader, num_epochs, batch_size, testloader, verbose=False, cuda=False, acc_break=None):

    model.train()
    losses      = {'test': [], 'train': []}
    accuracy    = {'test': [], 'train': []}
    precision   = {'test': [], 'train': []}
    recall      = {'test': [], 'train': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            for dp in batched(data, batch_size):
                # x: input, y: output
                x       = dp['x']
                y       = Variable(LongTensor(dp['y'], cuda=cuda))
                # forward + backward + optimize
                fx      = model(x)
                loss    = criterion(fx, y)
                loss.backward()
            optimizer.step()

            # Running Loss
            loss_val        = float(loss.data[0]) # averaged automatically
            running_loss    += loss_val

        # Losses
        # Computed once per epoch

        train = total_loss(trainloader, model, criterion, cuda=cuda)
        test  = total_loss(testloader, model, criterion, cuda=cuda) 

        losses['train'].append(train)
        losses['test'].append(test)

        # Accuracy
        test  = get_accuracy(testloader, model, verbose=False, cuda=cuda)
        train = get_accuracy(trainloader, model, verbose=False, cuda=cuda)

        accuracy['train'].append(train['accuracy'])
        accuracy['test'].append(test['accuracy'])

        precision['train'].append(train['precision'])
        precision['test'].append(test['precision'])

        recall['train'].append(train['recall'])
        recall['test'].append(test['recall'])
    
        if not acc_break is None and accuracy['test'] >= acc_break:
            print('Reached %g accuracy in %d epochs' % (accuracy['test'], epoch))
            break

        # Stats
        if verbose:
            r2str = lambda x : ', '.join(['%s=%.2f' % (k,v[-1]) for k,v in x.items()])
            loss = 'loss(%s)' % (r2str(losses))
            acc  = 'acc(%s)'  % (r2str(accuracy))
            prec = 'prec(%s)' % (r2str(precision))
            rec  = 'rec(%s)'  % (r2str(recall))
            print('(epoch=%d) %s, %s, %s, %s' % (epoch+1, loss, acc, prec,rec))

    if verbose:
        print('Finished training')

    #to_series  = lambda x : pd.Series(x).to_json(orient='values')
    out = {}
    out['loss']      = losses
    out['accuracy']  = accuracy
    out['precision'] = precision
    out['recall']    = recall

    return out

