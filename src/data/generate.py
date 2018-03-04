
from torch.utils.data.dataset import Dataset
import numpy as np
from scipy.optimize import linprog
from scipy import stats
import math
import random
import sys

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

    def __init__(self, m, n, N, num_lps=1, test=False, seed=3231):
        # set main parameters
        # (m,n) : dimension of A
        # N     : number of datapoints
        self.m          = m
        self.n          = n
        self.N          = N
        # set seet for numpy
        np.random.seed(seed)
        self.seed       = seed
        self.test_mode  = test
        # Generate 'num_lps' problems
        num_lps         = 1
        step            = np.random.randint(1, 1000) 
        self._seeds     = [num_lps + i*step for i in range(num_lps)]
        self._problems  = self._generate_problems()
        self._index     = [] # Lists num_lps*m restrictions
        self._populate_index()

    def __len__(self):
        return self.N

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

    def _generate_problems(self):
        problems = []
        for seed in self._seeds:
            prob = self.create_lp_problem(self.m, self.n, seed=seed)
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
        if self.test_mode:
            self._index = points 
        else:
            self._index = len(actives)*inactives + len(inactives)*actives 

    @staticmethod
    def create_lp_problem(m, n, seed=None):
        '''
        We are interested in programs of the form
            min_x   c*x
            st      A*x <= b
        SciPy solves LPs of the form
            min_x   c*x
            st      A_ub*x <= b_ub
                    A_eq*x == b_eq
        '''
        def threshold(x):
            x[np.abs(x)<=1e-7] = 0
            return x
        if seed:
            np.random.seed(seed)
        # Generate initial data
        A = np.random.rand(m, n)
        b = np.square(np.random.rand(m) + 1)
        c = np.random.uniform(low=-1.0,high=1.0,size=n)
        # Solve for x
        sol = linprog(c=c, A_ub=A, b_ub=b, A_eq=None, b_eq=None, bounds=None)
        if not sol.success:
            print('WARNING: Linear program did not succeed!')
        # Get active constraints
        x = sol.x
        active = (threshold(b-A.dot(x)) == 0).nonzero()[0]
        assert(len(active) > 0) # There must be at least one active
        prob = {'A': A, 'b': b, 'c': c, 'active': active}
        return prob

