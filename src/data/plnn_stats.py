import dotenv
import json
import os
import math
import numpy as np
import sys
import itertools
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import visdom
from tqdm import tqdm

from data.plnn_dataset import DatasetPLNN
from gurobipy import read
from utils.to_visdom import color_hue

from timeit import default_timer as timer

def read_json(fpath):
    with open(fpath, 'r') as f:
        json_load = json.load(f)
    return json_load

def plot_problem_stats(vis, df):

    # -----------
    # Active vs m
    # -----------

    def active_vs_m(df, index, active_type, label):
        # active_type in ['active', 'bound_active', 'matrix_active']
        df['pct_active_type'] = df[active_type]/df['m']
        fs      = [x.split('.mps')[0] for x in df['source'].tolist()]
        pt_text = ['%s\n(m,n)=(%d,%d)' % (f, m, n) for f, m, n in zip(fs, df['m'].tolist(), df['n'].tolist())]
        trace = dict(x=df['m'].tolist(), y=df['pct_active_type'].tolist(), 
                    mode="markers", type='custom',
                    marker={'color': color_hue(index, 0.4), 'symbol': 'dot', 'size': "3"},
                    text=pt_text, name=label)
        return trace

    # Plot
    traces = [None] * 3
    traces[0] = active_vs_m(df, 0, 'active', 'total')
    traces[1] = active_vs_m(df, 1, 'bounds_active', 'bounds')
    traces[2] = active_vs_m(df, 2, 'matrix_active', 'matrix')

    title = 'Percentage active vs m'
    layout = dict(title=title, xaxis={'title': 'm'}, yaxis={'title': 'pct_active'})
    vis._send({'data': traces, 'layout': layout, 'win': 'win_plnn_pct_active_vs_m'})

    # -----------
    # Time vs m
    # -----------

    def time_vs_m(df, index, label):
        fs      = [x.split('.mps')[0] for x in df['source'].tolist()]
        pt_text = ['%s\n(m,n)=(%d,%d)' % (f, m, n) for f, m, n in zip(fs, df['m'].tolist(), df['n'].tolist())]
        trace = dict(x=df['m'].tolist(), y=df['time'].tolist(), 
                    mode="markers", type='custom',
                    marker={'color': color_hue(index, 0.4), 'symbol': 'dot', 'size': "3"},
                    text=pt_text, name=label)
        return trace

    # Plot
    traces    = [None]
    traces[0] = time_vs_m(df, 0, 'time')

    title   = 'Gurobi time vs m'
    layout  = dict(title=title, xaxis={'title': 'm'}, yaxis={'type': 'log', 'title': 'seconds'})
    vis._send({'data': traces, 'layout': layout, 'win': 'win_plnn_m_vs_seconds'})

def timing_forward_pass(vis):
    from ml.models.s2v import Model
    import torch
    from torch.utils.data import DataLoader

    # Get dpath
    dotenv.load_dotenv(dotenv.find_dotenv())
    root    = os.environ.get('ROOT')
    dpath = os.path.join(root, 'data/output')

    # Get benchmark_plnn_* files by key
    fid = lambda x : x.split('.json')[0].split('_')[-1]
    fty = lambda x : x.split('.json')[0].split('_')[2]

    fhead = 'benchmark_plnn_'
    fs = [{'id': fid(f), 'type': fty(f), 'path': f} for f in os.listdir(dpath) if f.startswith(fhead)]
    d  = {f['id']:{'model': None, 'res': None} for f in fs}
    for f in fs:
        d[f['id']][f['type']] = os.path.join(dpath, f['path'])
    fs = d

    fdata = []
    for k,f in fs.items():
        # Get parameters
        res = read_json(f['res'])
        params = res['params']
        p   = params['p']
        t   = params['rounds_s2v']
        lps = res['out']['lps']
        # Build model from file
        model = Model(p, t, cuda=True)
        model.load_state_dict(torch.load(f['model']))
        model.eval()
        # Create dataset
        pset = DatasetPLNN(num_lps=1)
        pset.override_fpaths(lps)
        ploader = DataLoader(pset, batch_size=1, shuffle=True)
        for data in ploader:
            # get input data
            x = data['lp']
            x['node_features'] = data['node_features']
            # get timing for gurobi
            start = timer()
            yhat  = model(x)
            end   = timer()
            forward_time = end - start
            # get more info
            infopath = os.path.splitext(data['mps_path'][0])[0] + '.info'
            pparams = infopath_2_params(infopath)
            gurobi_time = pparams['time']
            m = pparams['m']
            ratio = forward_time/gurobi_time
            fdata.append({'m': m, 'time_ratio': ratio, 'fpath': infopath})

    index = 1
    pt_text = ['%s' % (f['fpath']) for f in fdata]
    trace = dict(x=[f['m'] for f in fdata], y=[f['time_ratio'] for f in fdata], 
                mode="markers", type='custom',
                marker={'color': color_hue(index, 0.4), 'symbol': 'dot', 'size': "3"},
                text=pt_text, name='forward_vs_gurobi')

    title   = 'Forward_pass/Gurobi (time) vs m'
    layout  = dict(title=title, xaxis={'title': 'm'}, yaxis={'type': 'log', 'title': 'time ratio'})
    vis._send({'data': [trace], 'layout': layout, 'win': 'win_plnn_forward_vs_gurobi_time'})

    return

def get_active_bounds(info, model):
    x = info['x_opt']
    active = 0
    for v in model.getVars():
        try:
            lb = v.LB
        except AttributeError:
            lb = None
            pass
        try:
            ub = v.UB
        except AttributeError:
            ub = None
            pass 
        active += 1 if (lb and x[v.varName] == lb) or (ub and x[v.varName] == ub) else 0
    return active

def infopath_2_params(fpath):
    ks = ['sc', 'num_constrs', 'num_vars', 'time', 'source']
    # info
    info    = read_json(fpath)
    # mps
    mpspath = os.path.splitext(fpath)[0] + '.mps'
    model   = read(mpspath)
    # Assemble
    d                  = {k:v for k,v in info.items() if k in ks}
    d['num_bounds']    = info['num_bounds']['lb'] + info['num_bounds']['ub']
    d['matrix_eq']     = len([c for c in model.getConstrs() if c.Sense == '='])
    d['matrix_ineq']   = len([c for c in model.getConstrs() if c.Sense != '='])
    d['matrix_active'] = len(info['active'])
    d['bounds_active'] = get_active_bounds(info, model)
    d['n']             = d['num_vars']
    d['m']             = d['num_constrs'] + d['num_bounds']
    d['active']        = d['matrix_active'] + d['bounds_active']
    return d

def stats():

    # Get paths
    fpaths, fdirs = DatasetPLNN.get_mps_paths(ext='.info', num_lps=20)

    stats = []
    for fpath in tqdm(fpaths):
        d = infopath_2_params(fpath, ks)
        stats.append(d)

    assert(all([x['sc'] == 2 for x in stats]))

    df = pd.DataFrame(stats)
    df = df.drop(columns=['sc'])

    # Plot
    vis = visdom.Visdom()
    plot_problem_stats(vis, df)

def timing():

    # Plot
    vis = visdom.Visdom()
    timing_forward_pass(vis)

if __name__ == '__main__':
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', 
        action='store_true', help='Stats about problems')
    parser.add_argument('--timing', 
        action='store_true', help='Timing of forward pass')

    args = parser.parse_args()
    if args.stats:
        stats()
    if args.timing:
        timing()

