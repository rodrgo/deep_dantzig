
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.fs import FileSystem
import pandas as pd
import itertools
import json
import random

def s2v_loss_grid(data, x_var, ax_row_var, ax_col_var, fig_var, line_var, legend_tags, fixed_params):
    '''
    fig_data is a list of dictionaries containing information
    about the figure.
    fig_data is a list of dictionaries

    Function creates a sequence of subplots with several plots in them

            fig_vars[0]          ...          fig_vars[k]
    
     =========================         ========================= 
    | ax_data[0] | ax_data[1] |       | ax_data[0] | ax_data[1] |
    |            |            |       |            |            |
    |            |            |       |            |            |
     -------------------------   ...   ------------------------- 
    |   ...      | ax_data[n] |       |   ...      | ax_data[n] |
    |            |            |       |            |            |
    |            |            |       |            |            |
     =========================         ========================= 

    '''
    
    fs = FileSystem()

    m = fixed_params['m']
    n = fixed_params['n']
    p = fixed_params['p']

    # title
    lp_title   = lambda m, n, p : '(m, n, p) = (%d, %d, %d)' % (m, n, p)
    opt_title  = lambda a, b, c : '(lr, mtm, wd) = (%g, %g, %g)' % (a, b, c)
    comments   = 'T: number of s2v rounds, acc: test-accuracy'

    # formatter
    def formatter(b):
        if isinstance(b, int):
            return '%d' % (b) 
        elif isinstance(b, float):
            return '%g' % (b) 
        elif isinstance(b, str):
            return '%s' % (b) 
        else:
            raise ValueError

    # ---------------
    # Parse fig_data by levels
    # 1. Figure
    # 2. Axis (horizontal, vertical)
    # 3. Line 
    # ---------------

    ds = {}

    # split figures
    for d in data:
        key = d[fig_var]
        if key in ds:
            ds[key].append(d)
        else:
            ds[key] = [d]

    # Line variables
    line_vars = list(set([x[line_var] for x in data]))

    # subplots
    for fig_val in ds.keys():
        figdata  = ds[fig_val]
        ax_rows  = sorted(list(set([x[ax_row_var] for x in figdata]))) 
        ax_cols  = sorted(list(set([x[ax_col_var] for x in figdata])))
        f, axs   = plt.subplots(len(ax_rows), len(ax_cols), sharex='col', sharey='row')
        for i, j in itertools.product(range(len(ax_rows)), range(len(ax_cols))):
            ax_r = ax_rows[i]
            ax_c = ax_cols[j]
            plot_data = [d for d in figdata if d[ax_row_var] == ax_r and d[ax_col_var] == ax_c]
            legends = []
            for t, d in enumerate(plot_data):
                x = eval(d[x_var])
                lines = axs[i, j].plot(range(len(x)), x, alpha=0.15*(t+1), label='t=%d' % (t))
                # yscale
                axs[i, j].set_yscale('log')
                # Rotate ticks
                plt.setp(axs[i, j].get_xticklabels(), rotation=30, horizontalalignment='right')
                # legends
                legends.append(', '.join(['%s:%s' % (var, formatter(d[var])) for var in legend_tags]))
                # inline label
                idx = math.ceil(len(x)/(2*len(plot_data) + 1) * (2*t + 1))
                color = lines[len(lines)-1].get_color()
                axs[i, j].text(idx, x[idx], 'acc=%s' % (formatter(d['acc'])), color = color, alpha=1)
            # get legends
            if i == len(ax_rows) - 1 and j == len(ax_cols) - 1:
                axs[i, j].legend(legends, loc='lower left')
            # cosmetics
            axs[i, j].set_xlabel('Iterations')
            axs[i, j].set_ylabel('log(loss(yhat, y))')

        # Label axes
        col_tags = ['%s=%s' % (ax_col_var, formatter(x)) for x in ax_cols]
        row_tags = ['%s=%s' % (ax_row_var, formatter(x)) for x in ax_rows]
        for ax, col in zip(axs[0], col_tags):
            ax.set_title(col)
        for ax, row in zip(axs[:,0], row_tags):
            ax.set_ylabel(row, rotation=90, size='large')

        plt.suptitle(lp_title(m, n, p) + " / " + comments, fontsize=15)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fpath = fs.get_path('outputs', 'overfitting_s2v_benchmark_%s_%s.png' % (fig_var, str(fig_val)))
        plt.savefig(fpath)
    return

def view_factors(data, params):
    data_view = [{k:x[k] for k in x.keys() if k in params} for x in data]
    df = pd.DataFrame(data_view)
    print(df)
    return

def main():
    fs       = FileSystem()
    data = fs.read_json('outputs', 'benchmark.json')
    if False:
        view_factors(data, ['seed', 'bs', 'T', 'lr', 'mtm', 'wd'])
    # Varying factors are [seed, bs, T, lr]
    params   = {'m': 10, 'n': 5, 'p': 12}
    ax_h_var = 'lr'
    ax_v_var = 'bs'
    line_var = 'T'
    fig_var  = 'seed'
    x_var    = 'loss'
    legend_tags = ['T']
    s2v_loss_grid(data, x_var, ax_h_var, ax_v_var, fig_var, line_var, legend_tags, params)

if __name__ == '__main__':
    main()

