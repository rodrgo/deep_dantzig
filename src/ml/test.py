
import torch
from torch.autograd import Variable
from ml.types import LongTensor
import numpy as np
from sklearn.metrics import roc_curve
import visdom
from ml.utils import batched

def get_accuracy(testloader, model, prob_thresh=0.5):
    model.eval()

    tps, fps, tns, fns = [0]*4
    imp = 0 # Improvement over training strategy
    num_ineq = 0
    pos_ineq = 0 
    for data in testloader:
        for x, y in batched(data, 1, model.graph):
            fx = model(x)
            probs = model.probs.data
            predicted = probs[:,[1]] >= prob_thresh
            if model.on_cuda: 
                predicted = predicted.cpu()
            if False:
                # Fix predicted
                # imp
                ineq = [int(l) == 1 for l in list(x['c_feats'][:,0])]
                ineq = torch.FloatTensor(ineq)
                imp += sum(np.logical_and( ineq == 0, predicted.squeeze(1) == 1))
                idx = [j for j,q in enumerate(ineq.numpy().tolist()) if q == 0]
                #predicted[idx] = 0

            predicted = predicted.numpy().squeeze(axis=1)
            tps += sum(np.logical_and( y == 1, predicted == 1))
            fps += sum(np.logical_and( y == 0, predicted == 1))
            tns += sum(np.logical_and( y == 0, predicted == 0))
            fns += sum(np.logical_and( y == 1, predicted == 0))
            #pos_ineq += sum(np.logical_and(predicted==1, ineq == 1))
            #num_ineq += sum(np.logical_and(ineq == 1, ineq ==1))


    res = {}
    res['accuracy']  = (tps + tns)/(tps + fps + tns + fns)
    res['precision'] = tps/max((tps + fps), 1)
    res['recall']    = tps/max((tps + fns), 1)
    res['y_pos']     = (tps + fns)/(tps + fps + tns + fns)
    res['y_neg']     = (fps + tns)/(tps + fps + tns + fns)
    res['pred_pos']  = (tps + fps)/(tps + fps + tns + fns)
    res['pred_neg']  = (tns + fns)/(tps + fps + tns + fns)
    #res['imp']       = imp
    #res['ineq_red']  = pos_ineq/max(num_ineq, 1)

    model.train()
    return res

def plot_loss(losses, epoch_num):
    vis = visdom.Visdom()

    def live_loss(y, loss_type):
        trace = dict(x=list(range(len(y))), y=y, 
                    mode="lines", type='custom',
                    marker={'color': 'black', 'symbol': 'dot', 'size': "1"},
                    text=None, name='roc_curve')
        title   = '%s loss (Epoch=%d)' % (loss_type, epoch_num)
        layout  = dict(title=title, xaxis={'title': 'epoch'}, yaxis={'type': 'log', 'autorange': True, 'title': 'loss'})
        vis._send({'data': [trace], 'layout': layout, 'win': '%s_live' % (loss_type)})

    live_loss(losses['test'], 'test')
    live_loss(losses['train'], 'train')
    return

