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

def plot_problem_stats(vis, df):
    x = df['delta'].tolist()
    y = df['rho'].tolist()

    ms = df['m'].tolist()
    ns = df['n'].tolist()
    fs = [x.split('.mps')[0] for x in df['fpath'].tolist()]
    pt_text = ['%s\n(m,n)=(%d,%d)' % (f, m, n) for f, m, n in zip(fs, ms, ns)]

    trace = dict(x=x, y=y, mode="markers", type='custom',
                marker={'color': 'black', 'symbol': 'dot', 'size': "5"},
                text=pt_text, name='1st Trace')
    layout = dict(title="Problem stats", xaxis={'title': 'ineq'}, yaxis={'title': 'active/ineq'})
    vis._send({'data': [trace], 'layout': layout, 'win': 'win_plnn_stats'})

def main():

    dotenv.load_dotenv(dotenv.find_dotenv())
    root    = os.environ.get('ROOT')
    fpath   = os.path.join(root, 'data/plnn/stats.json')
    with open(fpath, 'r') as f:
        d = json.load(f)

    assert(all([x['success'] for x in d]))
    assert(all([x['sc'] in [1, 2] for x in d]))

    df = pd.DataFrame(d)
    df = df.drop(columns=['objval', 'sc', 'success'])

    # Plot
    #   delta=ineq/m vs rho=active/ineq
    #   bubble size by size of n

    df['delta'] = df['ineq']
    df['rho']   = df['active']/df['ineq']  

    vis = visdom.Visdom()
    plot_problem_stats(vis, df)

if __name__ == '__main__':
	main()
