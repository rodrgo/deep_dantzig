from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np
from ml.types import LongTensor

def batched(data, batch_size):
    batch   = 0
    for batch in range(batch_size):
        dp = {}
        dp['x']      = {}
        dp['x']['A'] = data['x']['A'][batch, :, :].unsqueeze(0)
        dp['x']['b'] = data['x']['b'][batch, :].unsqueeze(0)
        dp['x']['c'] = data['x']['c'][batch, :].unsqueeze(0)
        dp['x']['i'] = torch.LongTensor([data['x']['i'][batch]])
        dp['y']      = torch.LongTensor([data['y'][batch]])
        yield dp

def get_total_loss(testloader, net, criterion, cuda=False):
    # Compute total loss
    if testloader is None:
        total_loss = None 
    else:
        total_loss = 0.0
        net.require_grads(False)
        for i, data in enumerate(testloader, 0):
            for dp in batched(data, 1):
                inputs      = dp['x']
                label       = Variable(LongTensor(dp['y'], cuda=cuda))
                outputs     = net(inputs)
                loss        = criterion(outputs, label)
                total_loss  += loss.data[0] 
        net.require_grads(True)
    return total_loss

def train_net(net, criterion, optimizer, trainloader, num_epochs, batch_size, testloader=None, verbose=False, cuda=False, acc_break=None):
    net.train()
    losses          = [] 
    running_losses  = []
    total_losses    = []
    accuracies      = [] 
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            for dp in batched(data, batch_size):
                # x: input, y: output
                inputs  = dp['x']
                label   = Variable(LongTensor(dp['y'], cuda=cuda))
                # forward + backward + optimize
                outputs = net(inputs)
                loss    = criterion(outputs, label)
                loss.backward()
            optimizer.step()

            # Loss
            # averaged automatically
            loss_val        = float(loss.data[0]) 
            losses          = np.append(losses, loss_val)

            # Running Loss
            running_loss    += loss_val
            running_losses  = np.append(running_losses, running_loss)

        # Total Loss
        # Computed once per epoch
        total_loss      = get_total_loss(testloader, net, criterion, cuda=cuda) 
        total_losses    = np.append(total_losses, total_loss)

        # Accuracy
        acc = test_binary_classification(testloader, net, 
                verbose=False, cuda=cuda)
        accuracies.append(acc)

        if not acc_break is None and acc >= acc_break:
            print('Reached %g accuracy in %d epochs' % (acc, epoch))
            break

        # Stats
        if verbose:
            print('(epoch=%d) Loss: val=%.7f, running=%.7f, total=%.7f, accuracy=%1.3f' % (epoch+1, loss_val, running_loss, total_loss, acc))

    if verbose:
        print('Finished training')

    to_series = lambda x : pd.Series(x).to_json(orient='values')
    losses_out = {}
    losses_out['batch']   = to_series(losses)
    losses_out['running'] = to_series(running_losses)
    losses_out['total']   = to_series(total_losses)
    # acc is not a loss, but put it here too
    losses_out['accs']    = to_series(accuracies)
    return losses_out

