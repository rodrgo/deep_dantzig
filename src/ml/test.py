
import torch
from torch.autograd import Variable
from ml.types import LongTensor
import numpy as np

def test_binary_classification(testloader, net, verbose=False, cuda=False):
    net.eval()

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    for data in testloader:
        y = np.asarray(data['node_labels'])
        x = data['lp']
        x['node_features'] = data['node_features']

        # True labels
        y  = Variable(LongTensor(torch.from_numpy(y), cuda=cuda))
        if cuda:
            y = y.cpu()
        y = y.squeeze(0)

        # Predicted label
        yhat  = net(x)
        _, predicted = torch.max(yhat.data, 1)
        if cuda: 
            predicted = predicted.cpu()
        predicted    = predicted.numpy()
        y            = y.data.numpy()

        tp = sum(np.logical_and(predicted == y, y == 1))
        fp = sum(np.logical_and(predicted != y, y == 1))
        tn = sum(np.logical_and(predicted == y, y == 0))
        fn = sum(np.logical_and(predicted != y, y == 0))

        tps += tp 
        fps += fp
        tns += tn
        fns += fn

        precision   = tp/(tp+fp)
        recall      = tp/(tp+fn) 
        acc         = float((predicted == y).sum())/len(y)

        if verbose:
            print('Accuracy: %1.2f, Precision %1.2f, Recall: %1.2f' % (acc, precision, recall))

    acc         = (tps + tns)/(tps + fps + tns + fns)
    precision   = tps/(tps + fps)
    recall      = tps/(tps + fns)

    net.train()
    return acc, precision, recall

