
import torch
from torch.autograd import Variable
from ml.types import LongTensor
import numpy as np

def get_accuracy(testloader, net, verbose=False, cuda=False):
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

    acc         = (tps + tns)/(tps + fps + tns + fns)
    precision   = tps/(tps + fps)
    recall      = tps/(tps + fns)

    res = {}
    res['accuracy']     = acc
    res['precision']    = precision
    res['recall']       = recall

    net.train()
    return res

