
from torch.utils.data.dataset import Dataset
import numpy as np
from scipy.optimize import linprog
from scipy import stats
import math
import random
import sys

from data.gurobi_lp import LinProg

class RandomLPDataset(Dataset):
    '''
    Let (A, b, c) be an LP problem of the form
        min_z   c*z
        st      A*z <= b
    where,
        size(A) = (m,n) 
        size(b) = (m,1)
        size(c) = (1,n)
    Let
        g = b-A*z
    Hence,
        x = (A, b, c, i)
        y = (g[i] == 0)   i.e. constraint "i" is active.
    However, data is generated 'per-problem' as:
        {'lp'       : {'A': A, 'b': b, 'c', c}, 
         'labels'   : [ (i, label(i)) for i in range(m)] }
    '''

    def __init__(self, m, n, num_lps=1, test=False, seed=3231):
        # set main parameters
        # (m,n) : dimension of A
        self.m          = m
        self.n          = n
        # set seet for numpy
        np.random.seed(seed)
        self.seed       = seed
        self.test_mode  = test
        # Generate 'num_lps' problems
        step            = np.random.randint(1, 1000) 
        self._seeds     = [seed + i*step for i in range(num_lps)]
        self._problems  = self._generate_problems()

    def __len__(self):
        return len(self._problems)

    def __getitem__(self, idx):
        p = self._problems[idx % len(self._problems)]
        return {'lp': {'A': p['A'], 'b': p['b'], 'c': p['c']}, 'labels': p['labels']}

    def get_lp_params(self):
        params = []
        for p in self._problems:
            params.append(p['stats'])
        return params

    def _generate_problems(self):
        problems = []
        for seed in self._seeds:
            prob = self.create_lp_problem(self.m, self.n, seed=seed, with_stats=True)
            problems.append(prob)
        return problems

    @staticmethod
    def create_lp_problem(m, n, seed=None, with_stats=False):
        '''
        We are interested in programs of the form
            min_x   c*x
            st      A*x <= b
        SciPy solves LPs of the form
            min_x   c*x
            st      A_ub*x <= b_ub
                    A_eq*x == b_eq
        '''
        def reseed(seed):
            if not seed is None:
                np.random.seed(seed)
            return
        # Generate initial data
        reseed(seed)
        A = np.random.randn(m, n)
        b = A.dot(np.random.randn(n)) + np.absolute(np.random.randn(m))
        c = np.absolute(np.random.randn(n))
        ops = ['<'] * m
        obj = 'min'
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
        # Labels
        is_active   = lambda z : 1 if z in active else 0
        labels      = [(i, is_active(i)) for i in range(A.shape[0])]
        nactive     = len([x for x in labels if x[1] == 1])
        ninactive   = len([x for x in labels if x[1] == 0])
        if nactive != ninactive:
            print('WARNING: class inbalance')
        # Stats
        if with_stats:
            stats = {}
            stats['id']      = seed
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
        # Assemble
        prob = {'A': A, 
                'b': b, 
                'c': c, 
                'active': active, 
                'labels': labels,
                'stats' : stats}
        return prob

def main():
    # Problem Params
    n  = 5 
    m  = 10
    seed = 0

    # create_lp_problem
    prob = RandomLPDataset.create_lp_problem(m, n, seed)
    print(prob)
    print('%d actives out of %d' % (len(prob['active']), m))

    # Init
    dataset = RandomLPDataset(m, n, num_lps=1, test=False, seed=seed)

if __name__ == '__main__':
    main()

