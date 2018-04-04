
from torch.utils.data.dataset import Dataset
import numpy as np
import math
import random
import sys
import os
import subprocess
import dotenv
import json

from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import load_snapshot_and_simplify
from data.mps2numpy import mps2numpy  
from data.gurobi_lp import LinProg

class DatasetPLNN(Dataset):

    def __init__(self, num_lps=1, test=False, seed=3231):
        # set main parameters
        # (m,n) : dimension of A
        # N     : number of datapoints
        if not seed is None:
            np.random.seed(seed)
        # set seet for numpy
        np.random.seed(seed)
        self.seed       = seed
        self.test_mode  = test
        # Generate 'num_lps' problems
        fpaths          = self.get_mps_paths()
        self._problems  = self._generate_problems(fpaths, num_lps)
        self._index     = [] # Lists num_lps*m restrictions
        self._populate_index()

    def _populate_index(self):
        def is_active(z, prob):
            if z in prob['active']:
                return 1
            else:
                return 0
        points = []
        for j in range(len(self._problems)):
            prob = self._problems[j]
            for i in range(prob['A'].shape[0]):
                points.append({'prob': j, 'row': i, 'active': is_active(i, prob)})
        actives   = [x for x in points if x['active']]
        inactives = [x for x in points if not x['active']]
        # Replicate to have class balance 
        self._index = points
        #print('%d actives, %d inactives' % (len(actives), len(inactives)))
        return

    def get_lp_params(self):
        params = []
        for p in self._problems:
            params.append(p['stats'])
        return params

    @staticmethod
    def get_lp_dir():
        dotenv.load_dotenv(dotenv.find_dotenv())
        root    = os.environ.get('ROOT')
        lp_dir  = os.path.join(root, 'data/plnn')
        return lp_dir

    @staticmethod
    def get_mps_paths():
        lp_dir  = DatasetPLNN.get_lp_dir()
        fpaths  = [os.path.join(lp_dir, f) for f in os.listdir(lp_dir) if f.endswith('.mps')]
        return fpaths

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        # Extract datapoint from index
        point = self._index[idx % len(self._index)]
        num_problem = point['prob']
        row         = point['row']
        active      = point['active']
        # Problems 
        p = self._problems[num_problem]
        sample = {'x': {'A': p['A'], 'b': p['b'], 'c': p['c'], 'i': row}, 
                  'y': active }
        return sample

    def _generate_problems(self, fpaths, num_lps):
        if num_lps < len(fpaths):
            idxs    = np.random.randint(0, len(fpaths), num_lps) 
            fpaths  = [fpaths[idx] for idx in idxs]
        problems = []
        for fpath in fpaths:
            p = self.extract_lp_problem(fpath, with_stats=True)
            problems.append(p)
        return problems

    @staticmethod
    def extract_lp_problem(fpath, with_stats=False, standardize=True):
        '''
        We are interested in programs of the form
            min_x   c*x
            st      A*x <= b
        SciPy solves LPs of the form
            min_x   c*x
            st      A_ub*x <= b_ub
                    A_eq*x == b_eq
        '''
        # Generate initial data
        A, b, c, ops, obj = mps2numpy(fpath, standardize)
        # Solve for x
        lp = LinProg(A, b, c, obj, ops)
        lp.optimize()
        sc = lp.get_statuscode()
        success = True 
        if sc in [1, 2]:
            # Get active constraints
            active = lp.get_active_constraints()
            assert(len(active) > 0)
        else:
            success = False
            active = []
            print('WARNING: Linear program did not succeed!')
        prob = {'A': A, 'b': b, 'c': c, 'active': active}
        # Stats
        if with_stats:
            stats = {}
            stats['id']      = os.path.basename(fpath)
            stats['m']       = A.shape[0]
            stats['n']       = A.shape[1]
            stats['eq']      = len([x for x in ops if x == '='])
            stats['ineq']    = len([x for x in ops if x != '='])
            stats['active']  = len(active)
            stats['sc']      = sc
            stats['objval']  = lp.model.objVal
            stats['success'] = success
        else:
            stats = None
        prob['stats'] = stats
        return prob

def generate_plnn_stats():
    # Get paths
    fpaths = DatasetPLNN.get_mps_paths()
    # Generate and save
    probs = []
    for fpath in fpaths:
        prob = DatasetPLNN.extract_lp_problem(fpath, with_stats=True)
        probs.append(prob['stats'])
    lp_dir   = DatasetPLNN.get_lp_dir()
    outpath  = os.path.join(lp_dir, 'stats.json')
    with open(outpath, 'w') as outfile:
        json.dump(probs, outfile)
    return

def test_plnn_dataset():
    from gurobipy import read
    # Get paths
    fpaths = DatasetPLNN.get_mps_paths()
    # Solve them
    results = []
    print('%d problems to test' % (len(fpaths)))
    for i, fpath in enumerate(fpaths):
        # Plain
        model = read(fpath)
        model.setParam('OutputFlag', False) 
        try:
            model.optimize()
        except GurobiError:
            pass
        sc0     = model.status
        objval0 = model.objVal
        # Extract lp problem (not standardize)
        prob1   = DatasetPLNN.extract_lp_problem(fpath,
            with_stats=True, standardize=False)
        sc1     = prob1['stats']['sc']
        objval1 = prob1['stats']['objval']
        # Extract lp problem (standardize)
        prob2   = DatasetPLNN.extract_lp_problem(fpath,
            with_stats=True, standardize=True)
        sc2     = prob2['stats']['sc']
        objval2 = prob2['stats']['objval']
        # Assemble line
        bn = os.path.basename(fpath)
        line = '%s,%d,%d,%d,%g,%g,%g' % (bn, 
            sc0, sc1, sc2, objval0, objval1, objval2) 
        print('%d.- %s' % (i, line))
        results.append(line)
    # Print results
    lp_dir   = DatasetPLNN.get_lp_dir()
    outpath  = os.path.join(lp_dir, 'plnn_dataset_tests.csv')
    with open(outpath, 'w') as f:
        for line in results:
            f.write(line)
            f.write('\n')
    return

def generate_plnn_dataset():
    '''
    $prop   = file to prove
    $target = output file
    We only use $prop, are interested in using this 
    file and saving the output of LinearizedNetwork

    Creating a dataset:
        1. Get only the paths of $prop
        2. call load_and_simplify with LinearizedNetwork 
        3. When the LinearizedNetwork is solving the problem, 
            save the linearised networks before they go into gurobi 
    '''

    # Build relevant paths
    dotenv.load_dotenv(dotenv.find_dotenv())
    root      = os.environ.get('ROOT')
    plnnDir   = os.path.join(root, '../PLNN-verification')
    data_plnn = os.path.join(root, 'data/plnn')
    sh_path   = os.path.join(root, 'src/data', 'plnn_dataset_paths.sh')

    sys.path.append(plnnDir)
    cmd     = ['sh', sh_path, plnnDir]
    result  = subprocess.run(cmd, stdout=subprocess.PIPE)
    result  = result.stdout.decode('utf-8').split('\n')

    # First path in each line is the property that
    # the PLNN wants to prove
    rlvs = [y for y in [x.split(' ')[0] for x in result] if y]
    for i, rlv in enumerate(rlvs):
        save_params = {'fpath': data_plnn, 'tag': 'problem_%d' % (i)}
        with open(rlv, 'r') as rlv_infile:
            network, domain = load_snapshot_and_simplify(rlv_infile, LinearizedNetwork, save_params)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', 
        action='store_true', help='Generate PLNN dataset')
    parser.add_argument('--test', 
        action='store_true', help='Generate tests for  PLNN dataset')
    parser.add_argument('--stats', 
        action='store_true', help='Compute stats on PLNN dataset')

    args = parser.parse_args()
    if args.generate:
        generate_plnn_dataset()
    if args.test:
        test_plnn_dataset()
    if args.stats:
        generate_plnn_stats()

