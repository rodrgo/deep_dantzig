from gurobipy import *
from scipy import stats
import numpy as np
import math
from data.mps2numpy import model2numpy

class LinProg(object): 

    def init(self, A, b, c, obj='min', ops=None):
        # Old constructor
        self.A      = A
        self.m      = A.shape[0]
        self.n      = A.shape[1]

        self.b      = b
        self.c      = c
        self.x      = []
        self.obj    = obj
        self.ops    = ops
        self.error_msg = 'Gurobi Error, it has been reported'
        self.model  = Model('lp')
        self.model.setParam('OutputFlag', False) 

        # Construct
        self._add_variables()
        self._add_constraints()
        self._add_objective()


    @staticmethod
    def getitem(mps_path):
        # Read mps
        model = read(mps_path)

        # Standardize
        item = model2numpy(model, standardize=True)

        # node_features
        name2index  = item['cnames']
        name2sense  = item['csenses']
        assert(all([True if v in ['<', '='] else False for k,v in name2sense.items()]))

        assert(item['A'].shape[0] == len(name2sense))

        # 1 if sense is '<' and 0 otherwise
        node_features = [None] * item['A'].shape[0]
        for k,sense in name2sense.items(): 
            node_features[name2index[k]] = 1 if sense == '<' else 0

        # node_labels
        model.setParam('OutputFlag', 0)
        model.optimize()
        slacks = {c.ConstrName:c.Slack for c in model.getConstrs()}

        node_labels = [None] * item['A'].shape[0]
        for k,slack in slacks.items():
            node_labels[name2index[k]] = 1 if slack == 0 else 0

        x = {v.varName:v.x for v in model.getVars()}
        xbounds = item['bounds']
        for k,v in x.items():
            # we flip the lower bounds so need to test v == -lb['val']
            lb = xbounds[k]['lb']
            if lb:
                node_labels[name2index[lb['name']]] = 1 if v == -lb['val'] else 0
            ub = xbounds[k]['ub']
            if ub:
                node_labels[name2index[ub['name']]] = 1 if v == ub['val'] else 0
        
        assert(all([True if not v is None else False for v in node_features]))
        assert(all([True if not v is None else False for v in node_labels]))

        # Append an extra zero to node_features to account for constraints
        node_features.append(0)

        lp_item = {'lp': {'A': item['A'], 'b': item['b'], 'c': item['c']},
                   'node_features': np.asarray(node_features),
                   'node_labels':   np.asarray(node_labels)}

        return lp_item

    def dot(self, C, X):
        '''
        C = List of integer or float costs
        X = List of gurobi variables
        '''
        prod = None
        for i in range(len(C)):
            if i == 0:
                prod = C[0] * X[0]
            else:
                prod += C[i] * X[i]
        return prod

    def _add_variables(self):
        for i in range(self.n):
            lb = -GRB.INFINITY
            ub = GRB.INFINITY
            vt = GRB.CONTINUOUS
            name = 'x_%d' % (i)
            self.x.append(self.model.addVar(lb=lb, ub=ub, vtype=vt, name=name))
        return

    def _add_constraints(self):
        for i in range(self.m):
            name = 'a_%d' % (i) 
            ai   = self.A[i]
            bi   = self.b[i]
            if self.ops:
                op = self.ops[i]
            else:
                op = '<'
            try:
                if op == '=':
                    self.model.addConstr(self.dot(ai, self.x) == bi, name)
                elif op == '<':
                    self.model.addConstr(self.dot(ai, self.x) <= bi, name)
                elif op == '>':
                    self.model.addConstr(self.dot(ai, self.x) >= bi, name)
                else:
                    raise ValueError
            except GurobiError:
                print(self.error_msg)
        return

    def _add_objective(self):
        assert(self.x)
        if self.obj == 'max': 
            obj_var = GRB.MAXIMIZE
        elif self.obj == 'min':
            obj_var = GRB.MINIMIZE
        else:
            raise ValueError
        try:
            self.model.setObjective(self.dot(self.c, self.x), obj_var)
        except GurobiError:
            print(self.error_msg)
        return

    def optimize(self):
        try:
            self.model.optimize()
        except GurobiError:
            print(self.error_msg)
        return

    def get_active_constraints(self):
        def threshold(x):
            x[np.abs(x)<=1e-7] = 0
            return x
        x = np.zeros(self.n)
        for i, v in enumerate(self.model.getVars()):
            x[i] = v.x
        active = (threshold(self.b-self.A.dot(x)) == 0).nonzero()[0]
        return active

    def get_statuscode(self):
        # http://www.gurobi.com/documentation/7.5/refman/optimization_status_codes.html#sec:StatusCodes
        statuscodes = {1: 'loaded', 
            2: 'optimal', 
            3: 'infeasible',
            4: 'inf_or_unbd', 
            5: 'unbounded',
            6: 'cutoff',
            7: 'iteration_limit',
            8: 'node_limit',
            9: 'time_limit',
            10: 'solution_limit', 
            11: 'interrupted', 
            12: 'numeric',
            13: 'suboptimal',
            14: 'inprogress', 
            15: 'user_obj_limit'}
        s = self.model.status
        if not s in [1, 2]:
            print(statuscodes[s])
        return s

    def print_all(self):
        print(self.model)
        for v in self.model.getConstrs():
            if v.getAttr('ConstrName').endswith('neutral'): 
                print(v.getAttr('Slack'))
        for v in self.model.getVars():
            print('%s\t: %5.2f' % (v.varName, v.x))
        print('Obj:', self.model.objVal)
        return

