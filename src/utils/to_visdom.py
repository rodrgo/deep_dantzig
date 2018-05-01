
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.fs import FileSystem
import pandas as pd
import itertools
import json
import visdom
import random
import os
import sys
import math

def color_hue(idx, alpha=0):
    # idx (colour index) <= num_colours
    def rotate(l, n):
        return l[n:] + l[:n]
    P = 3
    idx = idx % P      
    t = math.floor(255/P)
    red   = rotate([t*i for i in range(P)], 0)
    green = rotate([t*i for i in range(P)], 1)
    blue  = rotate([t*i for i in range(P)], 2)
    r, g, b = next(itertools.islice(zip(red,green,blue), idx, idx+1))
    return 'rgba(%d,%d,%d,%g)' % (r, g, b, alpha)

def plot_in_visdom(vis, results):
    '''
        Multiplot
    '''

    def build_traces(rs, results_type, results_mode, seed):
        # Get unique/non-unique keys in r['params']
        df = pd.DataFrame([r['params'] for r in rs])
        assert(not df.isnull().values.any())
        d = json.loads(df.apply(lambda x: len(x.unique()) == 1, axis=0).to_json())
        non_uniques = [k for k in d.keys() if not d[k]] # Identifies a trace
        uniques     = [k for k in d.keys() if d[k]]     # Identifies group of traces (plot)
        kv          = lambda k, v, t : '%s=%s' % (k, str(v)) if t is False else '<br>%s=%s' % (k, str(v))
        params2name = lambda x : ','.join([kv(k, x[k], False) for k in non_uniques])
        params2plot = lambda x : ','.join([kv(k, x[k], True) if i == int(len(uniques)/2) else kv(k, x[k], False) for i, k in enumerate(uniques)])

        # Choose parameters for colour and alpha

        cvar = 'rounds_s2v'
        alphavar = 'p'

		# Continue

        cvars      = list(set([r['params'][cvar] for r in rs]))
        alphavars  = list(set([r['params'][alphavar] for r in rs]))
        colour_index = {p:j for j,p in enumerate(cvars)}
        alpha_index = {q:float((j+1)/len(alphavars)) for j,q in enumerate(alphavars)}

        # Plot data
        traces = []
        for j, r in enumerate(rs):
            params = r['params']
            y = r['out']['results'][results_type][results_mode]
            x = [i for i in range(len(y))]
            colour = colour_index[params[cvar]]
            alpha  = alpha_index[params[alphavar]]
            trace = dict(x=x, y=y, mode="lines", type='custom',
                        marker={'color': color_hue(colour, alpha), 'symbol': 'dot', 'size': "1"},
                        text=None, name=params2name(params))
            traces.append(trace)
        # title
        params_rename = {'batch_size': 'bs', 'learning_rate': 'lr', 'momentum': 'mtm', 'num_epochs': 'epochs', 'weight_decay': 'wd'}
        dataset = rs[0]['dataset']
        params  = {params_rename[k]:v for k,v in params.items() if k in params_rename}
        uniques = [params_rename[k] for k in uniques if k in params_rename]
        tag     = params2plot(params)
        title   = '<b>%s %s (%s) </b><br>%s<br>seed=%d' % (results_type, results_mode, dataset, tag, seed)
        # layout
        layout = dict(title=title, xaxis={'title': 'epochs'}, yaxis={'type': 'log', 'title': 'loss'}, legend=dict(x=1.1, y=1.3))
        return traces, layout

    def build_trace_and_send(vis, rs, seed, res_type, res_mode):
        traces, layout = build_traces(rs, res_type, res_mode, seed)
        vis._send({'data': traces, 'layout': layout, 'win': 'win_%s_%s_%d' % (res_type, res_mode, seed)})
        return

    select = lambda x, s, lr, bs: x['seed'] == s and x['params']['learning_rate'] == lr and x['params']['batch_size'] == bs
    seeds  = list(set([r['seed'] for r in results]))
    for seed in seeds:
        # Ideally we should do one grid per seed
        hvals = list(set([r['params']['learning_rate'] for r in results]))
        vvals = list(set([r['params']['batch_size'] for r in results]))
        for w, (hval, vval) in enumerate(zip(hvals, vvals)):
            rs = [r for r in results if select(r, seed, hval, vval)]

            # Loss
            build_trace_and_send(vis, rs, seed, 'loss', 'test')
            build_trace_and_send(vis, rs, seed, 'loss', 'train')

            # Accuracy
            build_trace_and_send(vis, rs, seed, 'accuracy', 'test')
            build_trace_and_send(vis, rs, seed, 'accuracy', 'train')

            # Precision
            build_trace_and_send(vis, rs, seed, 'precision', 'test')
            build_trace_and_send(vis, rs, seed, 'precision', 'train')

            # Recall
            build_trace_and_send(vis, rs, seed, 'recall', 'test')
            build_trace_and_send(vis, rs, seed, 'recall', 'train')

    return

def plot_benchmark():
    loc     = 'output'
    tag     = 'benchmark_plnn_res'
    tag     = 'benchmark_mnist_res'

    fs      = FileSystem()
    fpath   = fs._infer_location(loc)
    fnames  = [x for x in os.listdir(fpath) if x.startswith(tag)]

    results = []
    for fname in fnames:
        res = fs.read_json(loc, fname)
        results.append(res)

    vis = visdom.Visdom()
    plot_in_visdom(vis, results)

def main():

    if True:
        plot_benchmark()


if __name__ == '__main__':
    main()

