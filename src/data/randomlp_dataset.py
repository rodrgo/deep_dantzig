
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
        self._index     = [] # Lists num_lps*m restrictions
        self._populate_index()

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        '''
        Each problem has 'm' restrictions, 
        so we have a total of N*m datapoints
        '''
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
        # In test_mode we return all points
        # since this dataset is meant to be used in
        # overfitting tests
        if len(actives) == len(inactives) or self.test_mode:
            self._index = points 
        else:
            print('WARNING: class inbalance')
            self._index = len(actives)*inactives + len(inactives)*actives 

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
        prob = {'A': A, 'b': b, 'c': c, 'active': active}
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
        prob['stats'] = stats
        return prob

def main():
    # Problem Params
    n  = 5 
    m  = 10
    seed = 0

    prob = RandomLPDataset.create_lp_problem(m, n, seed)
    print(prob)
    print('%d actives out of %d' % (len(prob['active']), m))

if __name__ == '__main__':
    main()

