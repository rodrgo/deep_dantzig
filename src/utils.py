
from torch.utils.data.dataset import Dataset
import numpy as np
from scipy.optimize import linprog
from scipy import stats
import math
import random

class LPDataset(Dataset):
    '''
    Let (A, b, c) be an LP problem with 'm' restrictions
    and 'n' variables, and let 'z' be the optimal
    solution to
        min_z   c*z
        st      A*z <= b
    Define g = b-A*z
    data of the form
        x = (A, b, c, i)
        y = (g[i] == 0)
    Each LP problem contains 'm' constraints.
    The constraints of each problem are added sequentially.
    To get i-th datapoint:
        Look at (i%(m+1))-th row in ceil(i/m)-th problem.
    To generate N datapoints:
        Generate ceil(N/m) problems
    '''

    def __init__(self, m, n, seed, N, num_lps='infer'):
        # set main parameters
        # (m,n) : dimension of A
        # N     : number of datapoints
        self.m          = m
        self.n          = n
        self.N          = N
        self.num_lps    = num_lps
        # set seet for numpy
        np.random.seed(seed)
        self.seed       = seed
        # Generate problems
        if num_lps == 'infer':
            self.num_lps = math.ceil(float(N)/float(m))
        else:
            try:
                num_lps = int(num_lps)
            except ValueError:
                print('Could not transform {} to int'.format(str(num_lps)))
        # Generate 'num_lps' problems
        step            = np.random.randint(1, 1000) 
        self._seeds     = [num_lps + i*step for i in range(num_lps)]
        self._problems  = self._generate_problems()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        '''
        Each problem has 'm' restrictions, 
        so we have a total of N*m datapoints
        '''
        num_problem = math.floor(idx / self.m) % self.num_lps
        num_row     = (idx % self.m)
        prob = self._problems[num_problem]
        A  = prob['A']
        b  = prob['b']
        c  = prob['c']
        i  = idx % (self.m)
        def is_active(z):
            if z in prob['active']:
                return 1
            else:
                return 0
        actives = [x for x in prob['active']]
        inactives = [x for x in range(self.m) if x not in actives]
        if idx % 2 == 0:
            # draw inactive
            i = inactives[0]
            #i = random.choice(inactives)
        else:
            # draw active
            i = actives[0]
            #i = random.choice(actives)
        #is_active = lambda z : z in prob['active'] 
        sample = {'x': {'A': A, 'b': b, 'c': c, 'i': i}, 
                  'y': is_active(i) }
        return sample

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
            print('Linear program did not succeed!')
        # Get active constraints
        x = sol.x
        active = (threshold(b-A.dot(x)) == 0).nonzero()[0]
        prob = {'A': A, 'b': b, 'c': c, 'active': active}
        return prob

    def _generate_problems(self):
        problems = []
        for seed in self._seeds:
            prob = self.create_lp_problem(self.m, self.n, seed=seed)
            problems.append(prob)
        return problems

