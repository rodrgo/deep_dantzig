
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
from data.gurobi_lp import LinProg as LP
from utils.toolbox import np_to_sparse_serial, sparse_serial_to_np

from timeit import default_timer as timer

def read_json(fpath):
    with open(fpath) as json_data:
        d = json.load(json_data)
    return d

class DatasetPLNN(Dataset):

    def __init__(self, dataset='mnist', graph='bipartite', 
        num_elems=None, elem_type='lp', seed=1111, test=False): 

        '''
        Training can be done at several levels:
            + property:     Train and predict at property levels
            + LP:           Fix property. Train and predict at LP level
            + constraint:   Fix property and LP (take largest LP). Train and predict at constraint level

        For 'property' and 'LP', iterator runs over fpaths (return self._fpaths for iterator)
        For 'constraint', iterator runs over constraints   (return ?)
        '''

        # Other params
        self.test = test
        self.seed = seed
        self.graph_structure = graph

        TRAIN_PCT = 0.90 # Has to be between 0 and 1

        assert(elem_type in ['property', 'lp', 'constraint'])

        np.random.seed(self.seed)
        property_dirs = list(np.random.permutation(self.get_prop_dirs(dataset)))

        if elem_type in ['property', 'lp']:

            if elem_type == 'property':
                train_props, test_props = self._train_test_split(property_dirs, num_elems, TRAIN_PCT)
                train_lps = [os.path.join(d,f) for d in train_props for f in os.listdir(d) if f.endswith('.mps')]
                test_lps  = [os.path.join(d,f) for d in test_props  for f in os.listdir(d) if f.endswith('.mps')]
            elif elem_type == 'lp':
                # choose property directory with the most problems
                pdir = max([{'dir': pdir, 'len': len(os.listdir(pdir))} for pdir in property_dirs], key=lambda x: x['len'])['dir']
                element_dirs = [os.path.join(pdir,f) for f in os.listdir(pdir) if f.endswith('.mps')]
                # Get only elements with at least one inequality constraint
                ok      = lambda d : d['num_ineq'] > 0
                fs      = element_dirs
                fs_ineq = [LP.ineq_num(f) for f in fs]
                fs_ok   = [d['path'] for d in fs_ineq if ok(d)]
                print('\t %d/%d LPs with inequality constraints' % (len(fs_ok), len(fs)))
                train_lps, test_lps = self._train_test_split(fs_ok, num_elems, TRAIN_PCT)

            if self.test:
                print('Test set')
                fs = test_lps 
            else:
                print('Train set')
                fs = train_lps

            fs_ineq = [LP.ineq_num(f) for f in fs]
            fs_ok   = [d['path'] for d in fs_ineq if ok(d)]
            print('\t %d/%d LPs with inequality constraints' % (len(fs_ok), len(fs)))

            total_lps = len(train_lps) + len(test_lps)
            print('(%s) %d/%d problems' % (elem_type, len(fs), total_lps))

            # category split
            n_pos         = sum([d['num_pos'] for d in fs_ineq if ok(d)])
            n_neg         = sum([d['num_neg'] for d in fs_ineq if ok(d)])
            n_eq          = sum([d['num_eq'] for d in fs_ineq if ok(d)])
            n_ineq        = sum([d['num_ineq'] for d in fs_ineq if ok(d)])
            n_inact_ineq  = sum([d['num_inactive_ineq'] for d in fs_ineq if ok(d)])
            n_total       = n_pos + n_neg 

        elif elem_type in ['constraint']:
            # Choose largest problem in property_dirs[0]
            pdir  = property_dirs[0]
            fs    = [os.path.join(pdir,f) for f in os.listdir(pdir) if f.endswith('.mps')]
            d  = max([LP.ineq_num(f) for f in fs], key=lambda x:x['num_constrs'])
            fs_ok = [d['path']]

            # category split
            n_pos         = d['num_pos']
            n_neg         = d['num_neg']
            n_eq          = d['num_eq']
            n_ineq        = d['num_ineq']
            n_inact_ineq  = d['num_inactive_ineq']
            n_total       = n_pos + n_neg

        else:
            raise ValueError('elem_type not recognised')

        self._fpaths        = fs_ok
        self.n_pos          = n_pos
        self.n_neg          = n_neg
        self.n_eq           = n_eq
        self.n_ineq         = n_ineq
        self.n_inact_ineq   = n_inact_ineq
        self.n_total        = n_total

        self.print_baselines()

        self.weight = [n_pos/n_total, n_neg/n_total]
        print(self.weight)

        # Load all data
        self.__items = []
        for fpath in self._fpaths:
            self.__items.append(self._fpath2item(fpath))

    def _fpath2item(self, fpath):
        if self.graph_structure == 'complete':
            item = LP.getitem_complete(fpath)
        elif self.graph_structure == 'bipartite':
            item = LP.getitem_bipartite(fpath)
        else:
            raise(ValueError)
        return item

    def print_baselines(self):
        if self.test:
            btype = 'Test'
        else:
            btype = 'Train'
        print('%s set' % (btype))
        print('%d total LPs' % (len(self._fpaths)))
        print('\t %d total constraints\n\
               \t %d positive\n\
               \t %d negative\n\
               \t %d equality\n\
               \t %d inequality\n\
               \t %d active_inequality\n'
               % (self.n_total, self.n_pos, self.n_neg, self.n_eq, self.n_ineq,
                   self.n_inact_ineq))
        return

    def override_fpaths(self, lps):
        fpaths = [os.path.join(h,f) for h in lps for f in os.listdir(h) if f.endswith('.mps')]
        self._fpaths = fpaths
        return

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, idx):
        item = self.__items[idx]
        return item

    def get_source_dir(self):
        return list(set([os.path.dirname(f) for f in self._fpaths]))

    def _train_test_split(self, items, num_items=None, TRAIN_PCT=0.90):

        assert(len(items) > 1)
        items = list(np.random.permutation(items))
        if not num_items is None:
            assert(num_items > 1)
            if num_items <= len(items):
                items = items[:num_items]
            else:
                raise(ValueError('num_items > len(items)'))

        N_train = min(int(len(items) * TRAIN_PCT), len(items) - 1)
        N_test  = len(items) - N_train

        assert(0 < N_train and N_train < len(items))
        assert(0 < N_test  and N_test  < len(items))

        test_items  = items[N_train:] if N_test  > 1 else [items[-1]] 
        train_items = items[:N_train] if N_train > 1 else [items[0]] 

        return train_items, test_items

    @staticmethod
    def get_lp_dir(dataset=None):
        dotenv.load_dotenv(dotenv.find_dotenv())
        root    = os.environ.get('ROOT')
        if dataset == 'mnist':
            lp_dir  = os.path.join(root, 'data/mnist/problems')
        else:
            lp_dir  = os.path.join(root, 'data/plnn')
        return lp_dir

    @staticmethod
    def get_prop_dirs(dataset):
        h  = DatasetPLNN.get_lp_dir(dataset)
        ds = [os.path.join(h,f) for f in os.listdir(h) if f.startswith('problem_')]
        return ds

    @staticmethod
    def get_mps_paths(ext='.mps', num_lps=None, seed=1111, dataset=None):
        h  = DatasetPLNN.get_lp_dir(dataset)
        ds = [os.path.join(h,f) for f in os.listdir(h) if f.startswith('problem_')]
        ds = [d for d in ds if os.path.isdir(d)]
        if num_lps and num_lps < len(ds):
            np.random.seed(seed)
            ds = np.random.choice(ds, size=num_lps, replace=False).tolist()
        fs = [os.path.join(d,f) for d in ds for f in os.listdir(d) if f.endswith(ext)]
        return fs, ds

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

def generate_plnn_mnist_dataset(root):
    from plnn.network_linear_approximation import LinearizedNetwork
    from plnn.model import load_snapshot_and_simplify

    '''
    Similar to generate_plnn_dataset
    '''

    # Build relevant paths
    plnnDir   = os.path.join(root, '../PLNN-verification')
    data_mnist = os.path.join(root, 'data/mnist')

    # Get paths
    rlvs = []
    for root, dirs, files in os.walk(data_mnist):
        for name in files:
            if name.endswith((".rlv")):
                rlvs.append(os.path.join(root, name))

    # First path in each line is the property that
    # the PLNN wants to prove
    for i, rlv in enumerate(rlvs):
        save_params = {'fpath': os.path.join(data_mnist, 'problems'),
            'tag': 'problem_%d' % (i),
            'source': rlv.split('..')[-1]}
        print(save_params['tag'])
        with open(rlv, 'r') as rlv_infile:
            network, domain = load_snapshot_and_simplify(rlv_infile, LinearizedNetwork, save_params)

if __name__ == '__main__':
    import argparse
    dotenv.load_dotenv(dotenv.find_dotenv())
    root    = os.environ.get('ROOT')
    plnnDir = os.path.abspath(os.path.join(root, '../PLNN-verification'))
    sys.path.insert(0, plnnDir)
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', 
        action='store_true', help='Generate PLNN dataset')
    parser.add_argument('--generatemnist', 
        action='store_true', help='Generate PLNN MNIST dataset')
    parser.add_argument('--transform', 
        action='store_true', help='Transform MPS to training data')
    parser.add_argument('--test', 
        action='store_true', help='Generate tests for  PLNN dataset')

    args = parser.parse_args()
    if args.generatemnist:
        generate_plnn_mnist_dataset(root)
    if args.generate:
        generate_plnn_dataset(root)
    if args.transform:
        mps_to_training_data(root)
    if args.test:
        test_plnn_dataset()

