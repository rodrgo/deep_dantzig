
from torch.utils.data.dataset import Dataset
import numpy as np
import math
import random
import sys
import os
import subprocess
import dotenv
import json

from data.mps2numpy import mps2numpy  
from data.gurobi_lp import LinProg
from utils.toolbox import np_to_sparse_serial, sparse_serial_to_np

from timeit import default_timer as timer

def read_json(fpath):
    with open(fpath) as json_data:
        d = json.load(json_data)
    return d

class DatasetPLNN(Dataset):

    def __init__(self, num_lps=None, test=False, seed=3231):
        # set main parameters
        # (m,n) : dimension of A
        # N     : number of datapoints
        self.test = test
        # Train,test splits
        TRAIN_PCT = 0.90
        assert(0.0 < TRAIN_PCT and TRAIN_PCT < 1.0)
        TEST_PCT  = 1 - TRAIN_PCT
        # set set for numpy
        np.random.seed(seed)
        fpaths = self.get_mps_paths(ext='.mps', num_lps=num_lps)
        # Define number of training and test sets
        N = len(fpaths)
        train_index = np.random.randint(0, N, int(N * TRAIN_PCT))
        if self.test:
            self._fpaths = [fpaths[i] for i in range(N) if not i in train_index]
        else:
            self._fpaths = [fpaths[i] for i in range(N) if i in train_index]

    def __len__(self):
        return len(self._fpaths)

    def __getitem__(self, idx):
        fpath = self._fpaths[idx]
        item = LinProg.getitem(fpath)
        return item

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
    def get_mps_paths(ext='.mps', num_lps=None):
        h  = DatasetPLNN.get_lp_dir()
        ds = [os.path.join(h,f) for f in os.listdir(h) if f.startswith('problem_')]
        ds = [d for d in ds if os.path.isdir(d)]
        if num_lps and num_lps < len(ds):
            ds = [ds[j] for j in np.random.randint(0, len(ds), num_lps)]
        fs = [os.path.join(d,f) for d in ds for f in os.listdir(d) if f.endswith(ext)]
        return fs

    @staticmethod
    def extract_lp_problem(fpath, standardize=True):
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
        start = timer()
        lp.optimize()
        end = timer()
        total_time = end - start
        # LP Info
        source = fpath
        sc = lp.model.status
        cs = lp.model.getConstrs()
        vs = lp.model.getVars()
        constr_names = [c.ConstrName for c in cs]
        var_names    = [v.varName for v in vs]
        d = {'sc':           sc,
            'constrs':       constr_names,
            'vars':          var_names,
            'num_bounds':    None,
            'constr_sense':  {c.ConstrName:c.Sense for c in cs},
            'time':          total_time,
            'source':        source,
            'num_constrs':   len(constr_names),
            'num_vars':      len(var_names),
            'upper_bounds':  {},
            'lower_bounds':  {},
            'slacks':        {},
            'x':             {},
            'obj_val':       None}
        if sc == 2:
            d['slacks']     = {c.ConstrName:c.Slack for c in cs}
            d['x']          = {v.varName:v.x for v in vs}
            d['obj_val']    = lp.model.objVal
            # Lower and upper bounds
            for v in vs: 
                try:
                    d['lower_bounds'][v.VarName] = v.LB
                except AttributeError:
                    pass
                try:
                    d['upper_bounds'][v.VarName] = v.UB
                except AttributeError:
                    pass
            d['num_bounds'] = len(d['lower_bounds']) + len(d['upper_bounds'])
        # Rename d as prob_info
        prob_info = d
        # Train information
        # We serialize matrix to coo
        nindex  = {c:i for i,c in enumerate(constr_names)}
        is_zero = lambda z : 1 if z == 0 else 0
        is_eq   = lambda z : 1 if z == '=' else 0
        prob = {'A': np_to_sparse_serial(A),
                'b': b.tolist(),
                'c': c.tolist(),
                'node_labels': {nindex[k]:is_zero(v) for k,v in d['slacks'].items()},
                'node_features': {nindex[k]:is_eq(v) for k,v in d['constr_sense'].items()}}
        return prob, prob_info

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
    # Sample some of the paths
    N = 2000
    np.random.seed(1010)
    fpaths = [fpaths[i] for i in random.sample(range(len(fpaths)), N)]
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

def _test_consistency(p0, p1):
    # test consistency between two problems
    # p0 is info for original problem
    # p1 is info transformed problem 
    # Gets nontrivial lower bounds (lbs) and upper bounds (ubs)
    lbs  = lambda d : {k:v for k,v in d.items() if v > -1e100}
    ubs  = lambda d : {k:v for k,v in d.items() if v < 1e100}
    lbs0 = lbs(p0['lower_bounds'])
    ubs0 = ubs(p0['upper_bounds'])
    lbs1 = lbs(p1['lower_bounds'])
    ubs1 = ubs(p1['upper_bounds'])
    tests = {}
    tests['sc']       = (p0['sc'] == p1['sc'])
    tests['obj_val']  = (abs(p0['obj_val']) - abs(p1['obj_val']) <= 1e-7)
    tests['num_vars'] = (p0['num_vars'] == p1['num_vars'])
    tests['bounds_1'] = (len(lbs1) == 0 and len(ubs1) == 0)
    tests['m']        = (p0['num_constrs'] + len(lbs0) + len(ubs0) == p1['num_constrs'])
    tests['sense']    = all([v in ['<', '='] for k,v in p1['constr_sense'].items()])
    return tests

def mps_to_training_data(root):
    from gurobipy import read
    # Assert consistency
    ASSERT_CONSISTENCY = True
    # Get mps paths
    fpaths = DatasetPLNN.get_mps_paths()
    print('%d problems to test' % (len(fpaths)))
    for i, fpath in enumerate(fpaths):
        # file tag and path
        dirname  = os.path.dirname(fpath)
        file_tag = os.path.basename(fpath).split('.mps')[0]
        # Extract lp problem (standardize)
        prob, p1 = DatasetPLNN.extract_lp_problem(fpath)
        # Check consistency against original problem
        if ASSERT_CONSISTENCY:
            # Original info
            p0 = read_json(os.path.join(dirname, '%s.info' % (file_tag)))
            tests = _test_consistency(p0, p1)
            if not all([v for k,v in tests.items()]):
                print('failed tests at %s' % fpath)
                p1['consistency_tests'] = tests
                file_tag = file_tag + '_inconsistent'
            else:
                p1['consistency_tests'] = None
        # Save problem for training
        prob_path = os.path.join(dirname, '%s.train' % (file_tag)) 
        with open(prob_path, 'w') as outfile:
            json.dump(prob, outfile)
        # Save info of standardized path
        info_path = os.path.join(dirname, '%s.traininfo' % (file_tag))
        with open(info_path, 'w') as outfile:
            json.dump(p1, outfile)
    return

def generate_plnn_dataset(root):
    from plnn.network_linear_approximation import LinearizedNetwork
    from plnn.model import load_snapshot_and_simplify

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
    plnnDir   = os.path.join(root, '../PLNN-verification')
    data_plnn = os.path.join(root, 'data/plnn')
    sh_path   = os.path.join(root, 'src/data', 'plnn_dataset_paths.sh')

    #sys.path.append(plnnDir)
    cmd     = ['sh', sh_path, plnnDir]
    result  = subprocess.run(cmd, stdout=subprocess.PIPE)
    result  = result.stdout.decode('utf-8').split('\n')
    result  = [x for x in result if len(x) > 0]

    # First path in each line is the property that
    # the PLNN wants to prove
    rlvs = [y for y in [x.split(' ')[0] for x in result] if y]
    for i, rlv in enumerate(rlvs):
        save_params = {'fpath': data_plnn, 
            'tag': 'problem_%d' % (i),
            'source': rlv.split('..')[-1]}
        print(save_params['source'])
        with open(rlv, 'r') as rlv_infile:
            network, domain = load_snapshot_and_simplify(rlv_infile, LinearizedNetwork, save_params)

if __name__ == '__main__':
    import argparse
    dotenv.load_dotenv(dotenv.find_dotenv())
    root    = os.environ.get('ROOT')
    plnnDir = os.path.abspath(os.path.join(root, '../PLNN-verification'))
    print(plnnDir)
    sys.path.insert(0, plnnDir)
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', 
        action='store_true', help='Generate PLNN dataset')
    parser.add_argument('--transform', 
        action='store_true', help='Transform MPS to training data')
    parser.add_argument('--test', 
        action='store_true', help='Generate tests for  PLNN dataset')
    parser.add_argument('--stats', 
        action='store_true', help='Compute stats on PLNN dataset')
    parser.add_argument('--test_consistency', 
        action='store_true', help='Test consistency of PLNN dataset')

    args = parser.parse_args()
    if args.generate:
        generate_plnn_dataset(root)
    if args.transform:
        mps_to_training_data(root)
    if args.test:
        test_plnn_dataset()
    if args.stats:
        generate_plnn_stats()

