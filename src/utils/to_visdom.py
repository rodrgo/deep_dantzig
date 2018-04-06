
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
    def rotate(l, n):
        return l[n:] + l[:n]
    P = 3 
    t = math.floor(255/P)
    red   = rotate([t*i for i in range(P)], 0*int(P/3))
    green = rotate([t*i for i in range(P)], 1*int(P/3))
    blue  = rotate([t*i for i in range(P)], 2*int(P/3))
    r, g, b = next(itertools.islice(zip(red,green,blue), idx, idx+1))
    return 'rgba(%d,%d,%d,%g)' % (r, g, b, alpha)

def plot_in_visdom(vis, results):
    '''
        Multiplot
    '''
    def build_traces(rs, loss_type):
        # Get unique/non-unique keys in r['params']
        df = pd.DataFrame([r['params'] for r in rs])
        assert(not df.isnull().values.any())
        d = json.loads(df.apply(lambda x: len(x.unique()) == 1, axis=0).to_json())
        non_uniques = [k for k in d.keys() if not d[k]] # Identifies a trace
        uniques     = [k for k in d.keys() if d[k]]     # Identifies group of traces (plot)
        kv          = lambda k, v, t : '%s=%s' % (k, str(v)) if t is False else '<br>%s=%s' % (k, str(v))
        params2name = lambda x : ','.join([kv(k, x[k], False) for k in non_uniques])
        params2plot = lambda x : ','.join([kv(k, x[k], True) if i == int(len(uniques)/2) else kv(k, x[k], False) for i, k in enumerate(uniques)])
        # Plot data
        traces = []
        for j, r in enumerate(rs):
            params = r['params']
            y = json.loads(r['out']['losses'][loss_type]) 
            x = [i for i in range(len(y))]
            trace = dict(x=x, y=y, mode="lines", type='custom',
                        marker={'color': color_hue(j, 0.4), 'symbol': 'dot', 'size': "1"},
                        text=None, name=params2name(params))
            traces.append(trace)
        # title
        dataset = rs[0]['dataset']
        params_rename = {'batch_size': 'bs', 'learning_rate': 'lr', 'momentum': 'mtm', 'num_epochs': 'epochs', 'weight_decay': 'wd'}
        params = {params_rename[k]:v for k,v in params.items() if k in params_rename}
        uniques = [params_rename[k] for k in uniques if k in params_rename]
        tag = params2plot(params)
        title  = '<b>%s loss (%s) </b><br>%s' % (loss_type, dataset, tag)
        # layout
        if loss_type == 'running':
            xaxis = {'title': 'iterations'}
        elif loss_type == 'total':
            xaxis = {'title': 'epochs'}
        layout = dict(title=title, xaxis=xaxis, yaxis={'type': 'log', 'title': 'loss'})
        return traces, layout

    select = lambda x, s, lr, bs: x['seed'] == s and x['params']['learning_rate'] == lr and x['params']['batch_size'] == bs
    seeds  = list(set([r['seed'] for r in results]))
    for seed in seeds:
        # Ideally we should do one grid per seed
        hvals = list(set([r['params']['learning_rate'] for r in results]))
        vvals = list(set([r['params']['batch_size'] for r in results]))
        for w, (hval, vval) in enumerate(zip(hvals, vvals)):
            rs = [r for r in results if select(r, seed, hval, vval)]
            # Running loss
            traces, layout = build_traces(rs, 'running')
            vis._send({'data': traces, 'layout': layout, 'win': 'win_run_%d_%d' % (w, seed)})
            # Total loss
            traces, layout = build_traces(rs, 'total')
            vis._send({'data': traces, 'layout': layout, 'win': 'win_tot_%d_%d' % (w, seed)})
    return

def plot_benchmark():
    loc     = 'output'
    tag     = 'benchmark_randomlp_res'

    fs      = FileSystem()
    fpath 	= fs._infer_location(loc)
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

