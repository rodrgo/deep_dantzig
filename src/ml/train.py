from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np
from ml.types import LongTensor
from ml.test import plot_loss
from ml.utils import batched
from sklearn.metrics import roc_curve
import visdom
import time

def train_net(model, criterion, optimizer, trainloader, testloader, epochs, batch_size, cuda=False):
    #######
    # Methodology:
    #   Constraints: Equality + Inequality (matrix and bound constraints)
    #   Model is a bipartite graph
    #
    #   Normalise [ai, -bi] to get [aip, bip]. Features are
    #
    #   (bip, <aip, c>, is_inequality, is_bound) c-vertices
    #   (value in objective) for v-vertices
    #
    #   positive labels = is_inequality, is_active, ~is_bound
    #
    #   Loss function includes: is_inequality, ~is_bound (ONLY)
    #
    #   REMAINS TO-DO:
    #       + Consider only normalising ai rather than [ai, -bi], and then compute <aip, c>
    #       + Didn't include any information about whether objective is max/min
    #       + Should we pass inner products between (ai,aj)?
    #       + If so, we go back to past case.
    #       + Check relative weights in the training set (seems like positives have a better learning rate)
    #       + Decay the learning rate
    #       + Add memory?
    #
    #   Didn't add LB and UB as features since the scale is too large.  
    #   Features:    For vertices
    #   +:           Active matrix inequality
    #   -:           Inactive matrix inequality + Equality + all Bounds
    #   Loss:        matrix inequalities
    #################
    
    # TODO: Try AdamNet, AdaGRAD, ADAdelta

    ACC_BREAK = True
    VERBOSE   = True
    VISDOM    = True

    model.train()

    metrics  = {'test': [], 'train': []}

    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        #print('\n\nStart Training loop')
        model.verbose = False
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            for x, y in batched(data, batch_size, model.graph):
                y       = Variable(LongTensor(y, cuda=cuda))
                fx      = model(x)
                loss    = criterion(fx, y)
                loss.backward()
            optimizer.step()

            # Running Loss
            loss_val        = float(loss.data[0]) # averaged automatically (turned it off)
            running_loss    += loss_val
        print('%d: running_loss %g (epoch %g secs, elapsed %g secs)' % (epoch, running_loss, time.time()-epoch_start, time.time()-train_start))
        model.verbose = False
        #print('End Training loop')

        p_train, p_test = plot_roc(model, epoch+1, trainloader, testloader)
        #prob_thresh = get_prob_recall_one(testloader, model)
        train_metrics = performance(trainloader, model, criterion, p_train)
        test_metrics  = performance(testloader, model, criterion, p_train) 

        metrics['train'].append(train_metrics)
        metrics['test'].append(test_metrics)

        # ROC curve
        if VISDOM:
            metric_titles = metrics['test'][0].keys()
            for metric_title in metric_titles:
                y_train = [d[metric_title] for d in metrics['train']]
                y_test  = [d[metric_title] for d in metrics['test']]
                live_visdom_metric(epoch+1, metric_title, y_train, y_test) 

        # Stats
        if False:
            r2str = lambda x : ', '.join(['%s=%.2f' % (k,v) for k,v in sorted(x.items())])
            print('(epoch=%d)' % (epoch+1))
            print('\tprob_thresh=%g' % (prob_thresh))
            print('\tLoss train=%g, Loss test=%g' % (train_loss, test_loss))
            print('\tTrain: %s' % (r2str(train_stats)))
            print('\tTest : %s' % (r2str(test_stats)))

    return metrics

def get_prob_recall_one(trainloader, model):
    # For recall==1, need zero false_negatives
    model.eval()
    pmins = []
    for i, data in enumerate(trainloader, 0):
        for x, y in batched(data, 1, model.graph):
            fx  = model(x)
            probs = model.probs.data[:,[1]] # probs[0]+probs[1] == 1
            if model.on_cuda: 
                probs = probs.cpu()
            pmin = np.amin(probs.numpy()[y == 1])
            pmins.append(pmin)
    p_threshold = float(np.amin(np.array(pmins)))
    model.train()
    return p_threshold

def plot_roc(model, epoch, trainloader=None, testloader=None):
    model.eval()

    def get_traces(dataloader, model, metric_type):
        y_true = np.empty([0,0])
        y_prob = np.empty([0,0])
        for data in dataloader:
            for x, y in batched(data, 1, model.graph):
                y_true = np.append(y_true, y)
                fx = model(x)
                probs = model.probs.data[:,[1]]
                if model.on_cuda: 
                    probs = probs.cpu()
                probs = probs.numpy().squeeze(1)
                y_prob = np.append(y_prob, probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)

        color = 'black' if metric_type == 'train' else 'red'

        # tpr == recall
        itemindex = np.where(tpr==1.0)
        fpr_thresh = fpr[itemindex].tolist()[0]
        p_thresh = thresholds[itemindex].tolist()[0]

        trace = dict(x=fpr.tolist(), y=tpr.tolist(), 
                    mode="lines", type='custom',
                    marker={'color': color, 'symbol': 'dot', 'size': "1"},
                    text=None, name='roc_curve_%s' % (metric_type))
        trace_fpr = dict(x=[fpr_thresh, fpr_thresh], y=[0, 1], 
                    mode="lines", type='custom',
                    marker={'color': color, 'symbol': 'dot', 'size': "1"},
                    text=None, name='fpr_thresh_%s' % (metric_type))
        return trace, trace_fpr, p_thresh

    # PLOT
    vis = visdom.Visdom()

    p_train = None
    p_test  = None

    traces = []
    if trainloader:
        trace_roc, trace_fpr, p_train = get_traces(trainloader, model, 'train')
        traces.append(trace_roc)
        traces.append(trace_fpr)
    if testloader:
        trace_roc, trace_fpr, p_test = get_traces(testloader, model, 'test')
        traces.append(trace_roc)
        traces.append(trace_fpr)

    title   = 'ROC curve (Epoch=%d)' % (epoch)
    layout  = dict(title=title, xaxis={'title': 'FPR'}, yaxis={'title': 'TPR'}, showlegend=False)
    vis._send({'data': traces, 'layout': layout, 'win': 'roc_curve'})

    return p_train, p_test 

def performance(testloader, model, criterion, prob_thresh=0.5):
    # Compute total loss
    model.eval()
    total_loss = 0.0
    tps, fps, tns, fns = [0]*4
    mineqs = 0 # matrix inequality constraint
    bineqs = 0 # bound inequality constraint
    tp_mineqs = 0
    tp_bineqs = 0
    fp_mineqs = 0
    fp_bineqs = 0
    for i, data in enumerate(testloader, 0):
        for x, y in batched(data, 1, model.graph):
            y           = Variable(LongTensor(y, cuda=model.on_cuda))
            fx          = model(x)
            # Loss
            loss        = criterion(fx, y)
            total_loss  += loss.data[0] 
            probs       = model.probs.data[:,[1]]
            # Accuracy
            predicted = probs >= prob_thresh
            if model.on_cuda: 
                predicted = predicted.cpu()
                y = y.data.cpu()
            predicted = predicted.numpy().squeeze(axis=1)
            y         = y.numpy()
            tps += sum(np.logical_and( y == 1, predicted == 1))
            fps += sum(np.logical_and( y == 0, predicted == 1))
            tns += sum(np.logical_and( y == 0, predicted == 0))
            fns += sum(np.logical_and( y == 1, predicted == 0))
            # Slice by is_inequality and is_bound
            is_inequality = x['c_feats'][:,0]
            is_bound      = x['c_feats'][:,2]
            if model.on_cuda:
                is_inequality = is_inequality.cpu()
                is_bound      = is_bound.cpu()
            is_inequality = is_inequality.numpy()
            is_bound = is_bound.numpy()

            if False:
                is_tp = np.logical_and( y == 1, predicted == 1)
                is_fp = np.logical_and( y == 0, predicted == 1)
                is_mineq = np.logical_and( is_inequality == 1, is_bound == 0)
                is_bineq = np.logical_and( is_inequality == 1, is_bound == 1)

                tp_mineqs += sum(np.logical_and( is_tp == 1, is_mineq == 1))
                tp_bineqs += sum(np.logical_and( is_tp == 1, is_bineq == 1))

                fp_mineqs += sum(np.logical_and( is_fp == 1, is_mineq == 1))
                fp_bineqs += sum(np.logical_and( is_fp == 1, is_bineq == 1))

                mineqs += sum(is_mineq)
                bineqs += sum(is_bineq)

    res = {}
    res['total_loss'] = total_loss
    res['accuracy']  = (tps + tns)/(tps + fps + tns + fns)
    res['precision'] = tps/max((tps + fps), 1)
    res['recall']    = tps/max((tps + fns), 1)
    res['y_pos']     = (tps + fns)/(tps + fps + tns + fns)
    res['y_neg']     = (fps + tns)/(tps + fps + tns + fns)
    res['pred_pos']  = (tps + fps)/(tps + fps + tns + fns)
    res['pred_neg']  = (tns + fns)/(tps + fps + tns + fns)

    if False:
        res['tp_mineq']  = (tp_mineqs)/(mineqs)
        res['tp_bineq']  = (tp_bineqs)/(bineqs)

        res['fp_mineq']  = (fp_mineqs)/(mineqs)
        res['fp_bineq']  = (fp_bineqs)/(bineqs)

    model.train()
    return res

def live_visdom_metric(epoch_num, metric_title, y_train=None, y_test=None):
    vis = visdom.Visdom()
    def get_trace(y, metric_type):
        color = 'black' if metric_type == 'train' else 'red'
        trace = dict(x=list(range(len(y))), y=y, 
                    mode="lines", type='custom',
                    marker={'color': color, 'symbol': 'dot', 'size': "1"},
                    text=None, name='%s' % (metric_type))
        return trace
    title  = '%s (Epoch=%d)' % (metric_title, epoch_num)
    yaxis  = {'title': metric_title}
    if metric_title == 'total_loss':
        yaxis['type']       = 'log'
        yaxis['autorange']  = True
    layout = dict(title=title, xaxis={'title': 'epoch'}, yaxis=yaxis)
    traces = []
    if y_train:
        train_trace = get_trace(y_train, 'train')
        traces.append(train_trace)
    if y_test:
        test_trace = get_trace(y_test, 'test')
        traces.append(test_trace)
    vis._send({'data': traces, 'layout': layout, 'win': '%s_live' % (metric_title)})
    return
