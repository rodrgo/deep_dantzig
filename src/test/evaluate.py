
import torch
from torch.autograd import Variable
import numpy as np

def test_binary_classification(testloader, net, verbose=False):
    correct = 0
    total   = 0
    for data in testloader:
        # x: input, y: output
        data_x  = data['x']
        data_y  = data['y']

        # True label
        labels  = Variable(data_y.type(torch.LongTensor))
        y       = labels.data.numpy()[0]

        # Predicted label
        outputs      = net(data_x)
        _, predicted = torch.max(outputs.data, 1)
        total        = total + labels.size(0)
        yhat         = predicted.numpy()[0]
        correct      = correct + (yhat == y).sum()

        if verbose:
            if y == yhat:
                print('(yhat, y) = (%d, %d)'  % (yhat, y))
            else:
                print('(yhat, y) = (%d, %d)*' % (yhat, y))
    accuracy = float(correct / total)
    if verbose:
        print('Accuracy on %d datapoints: %d %%' % (len(testloader), 100*accuracy))
    return accuracy

