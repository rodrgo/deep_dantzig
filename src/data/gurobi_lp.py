from gurobipy import *
import torch
from scipy import stats
import numpy as np
import math
import json
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
    def ineq_num(mps_path):
        # Need to get number of active and inactive 
        # in matrix inequalities.
        model = read(mps_path)
        mineq = set([c.ConstrName for c in model.getConstrs() if c.Sense != '='])
        meq   = set([c.ConstrName for c in model.getConstrs() if c.Sense == '='])
        # Get list of active (from info)
        # Read info  
        with open(os.path.splitext(mps_path)[0] + '.info', 'r') as f:
            info = json.load(f)
        active = set(info['active']) # includes inequality and equality

        num_ineq            = len(mineq)
        num_active_ineq     = len(list(active & mineq))
        num_inactive_ineq   = len(list(mineq - active))
        num_eq              = len(meq)
        d = { 'path': mps_path,
              'num_active_ineq': num_active_ineq, 
              'num_inactive_ineq': num_inactive_ineq, 
              'num_ineq': num_ineq,
              'num_eq': num_eq,
              'num_constrs': num_eq+num_ineq,
              # Next two labels dictate positive
              'num_pos': num_active_ineq,
              'num_neg': num_inactive_ineq + num_eq}
        return d

    @staticmethod
    def has_matrix_inequalities(mps_path):
        model = read(mps_path)
        return any([True for c in model.getConstrs() if c.Sense != '='])

    @staticmethod
    def getitem_bipartite(mps_path):
        '''
        Given an LP of the form
            min c*x
            st  A_eq * x =  b_eq
                A_in * x <= b_eq

                lb <= x <= ub

        Let A = [A_eq; A_in] be a matrix of size (m,n)
        Generate a weighted bipartite graph G = (C, V, E, W)
        with    
                C = [m], 
                V = [m], 
                E = {(i,j)  : A(i,j) != 0}, 
                W = {A(i,j) : (i,j) in E}
        '''

        graph    = LinProg.mps_to_bipartite_graph(mps_path)
        c_feats  = graph['c_feats']
        v_feats  = graph['v_feats']
        e_feats  = graph['e_feats']
        v_bounds = graph['v_bounds']
        active   = graph['active']
        mps_path = graph['mps_path']

        # ----------
        # Transform LBs and UBs to constraints and add to c_feats and e_feats
        # ----------

        BOUNDS_TO_CONSTRAINT = True
        if BOUNDS_TO_CONSTRAINT:
            # label all existing constraints as non bound
            for cname in c_feats.keys():
                c_feats[cname]['is_bound'] = 0

            lb2cfeat = lambda b : {'sense': '>', 'rhs': b['lb'], 'is_bound': 1}
            lbs = {'%s_lb' % (vname):lb2cfeat(vf) for vname, vf in v_feats.items() if vf['lb']}

            ub2cfeat = lambda b : {'sense': '<', 'rhs': b['ub'], 'is_bound': 1}
            ubs = {'%s_ub' % (vname):ub2cfeat(vf) for vname, vf in v_feats.items() if vf['ub']}

            b_feats  = {}
            b_edges  = []
            b_active = []
            for vname, vf in v_feats.items():
                if vf['lb']:
                    cname = '%s_lb' % (vname)
                    b_feats[cname] = {'sense': '>', 'rhs': vf['lb'], 'is_bound': 1}
                    b_edges.append({'vname': vname, 'cname': cname, 'coeff': 1.0})
                if vf['ub']:
                    cname = '%s_ub' % (vname)
                    b_feats[cname] = {'sense': '<', 'rhs': vf['ub'], 'is_bound': 1}
                    b_edges.append({'vname': vname, 'cname': cname, 'coeff': 1.0})
                if v_bounds[vname]:
                    b_active.append(cname)

            for k,v in b_feats.items():
                c_feats[k] = v
            e_feats.extend(b_edges)
            active.extend(b_active)

        cnames  = c_feats.keys()
        vnames  = v_feats.keys()
        m, n    = len(cnames), len(vnames)

        # ----------
        # Nomrmalisation
        # ----------

        # Build is_inequality label
        for cname in c_feats.keys():
            if c_feats[cname]['sense'] != '=':
                c_feats[cname]['is_inequality'] = 1
            else:
                c_feats[cname]['is_inequality'] = 0

        # Flip sense, rhs, coeff
        for e in e_feats:
            if e['cname'] in c_feats[e['cname']]['sense'] == '>': 
                e['coeff'] = -1.0 * e['coeff']

        for cname in c_feats.keys():
            if c_feats[cname]['sense'] == '>': 
                c_feats[cname]['rhs']   = -1.0 * c_feats[cname]['rhs']
                c_feats[cname]['sense'] = '<'

        # c_labels
        is_pos  = lambda c : c_feats[c]['sense'] != '=' and c in active and c_feats[c]['is_bound'] == 0
        c_labels  = [is_pos(cname) for cname in cnames]

        # c_feats: Features for constraints (nodes in C)
        ks      = ['is_inequality', 'rhs', 'is_bound']
        c_feats = [[c_feats[c][k] for k in ks] for c in cnames]
    
        # v_feats: Features for variables   (nodes in V)
        #ks      = ['lb', 'ub', 'obj']
        ks      = ['obj']
        v_feats = [[v_feats[v][k] for k in ks] for v in vnames]

        # e_feats: Features for edges       (nodes in E)
        cx      = {c:j for j,c in enumerate(cnames)}
        vx      = {v:j for j,v in enumerate(vnames)}
        i       = [[cx[e['cname']], vx[e['vname']]] for e in e_feats]
        coeffs  = [e['coeff'] for e in e_feats]
        e_feats = {'i': i, 'coeffs': coeffs}

        # to torch
        c_feats  = torch.FloatTensor(c_feats)
        v_feats  = torch.FloatTensor(v_feats)
        c_labels = torch.FloatTensor(c_labels)

        # in_loss = matrix_inequalities
        in_loss = np.nonzero(np.logical_and( c_feats[:,0] == 1, c_feats[:,2] == 0 )).numpy().tolist()
        in_loss = [x[0] for x in in_loss]

        lp_item = {'c_feats'   : c_feats,
                   'v_feats'   : v_feats,
                   'e_feats'   : e_feats,
                   'c_labels'  : c_labels,
                   'in_loss'   : in_loss,
                   'dims'      : {'m': m, 'n': n},
                   'mps_path'  : mps_path}

        return lp_item

    @staticmethod
    def mps_to_bipartite_graph(mps_path): 

        def has_lb(v):
            try:
                lb = v.LB
            except AttributeError:
                lb = -1e100
            return True if lb > -1e100 else False

        def has_ub(v):
            try:
                ub = v.UB
            except AttributeError:
                ub = 1e100
            return True if ub < 1e100 else False

        model = read(mps_path)

        # Constraint and Variable names
        cnames = [c.ConstrName for c in model.getConstrs()]
        vnames = [v.VarName    for v in model.getVars()]

        cs = {c.ConstrName:c for c in model.getConstrs()}
        vs = {v.VarName:v    for v in model.getVars()}

        cindex = {c:j for j,c in enumerate(model.getConstrs())}
        vindex = {v:j for j,v in enumerate(model.getVars())}

        # Constraint features
        get_c_feat = lambda c : {'sense': c.Sense, 'rhs' : c.RHS}
        c_feats = {cname: get_c_feat(c) for cname,c in cs.items()} 

        # Variable features
        get_v_feat = lambda v : {'lb'   : v.LB if has_lb(v) else None, 
                                 'ub'   : v.UB if has_ub(v) else None,
                                 'obj'  : model.Obj[vindex[v]]}
        v_feats = {vname:get_v_feat(v) for vname,v in vs.items()} 

        # E features
        e_feats = []
        for cname, c in cs.items():
            for j in range(model.getRow(c).size()):
                vname = model.getRow(c).getVar(j).varName
                coeff = model.getRow(c).getCoeff(j)
                if coeff != 0.0:
                    e_feats.append({'vname': vname, 
                                    'cname': cname, 
                                    'coeff': coeff})

        # active
        finfo = os.path.splitext(mps_path)[0] + '.info'
        with open(finfo, 'r') as f:
            lp_info = json.load(f)
        active = lp_info['active'] # All active constraints
        x_opt  = lp_info['x_opt']

        v_bounds = {vname:None for vname in v_feats.keys()}
        for vname, vfeat in v_feats.items():
            xj = x_opt[vname]
            if vfeat['lb'] and xj == vfeat['lb']:
                v_bounds[vname] = 'lb'
            if vfeat['ub'] and xj == vfeat['ub']:
                v_bounds[vname] = 'ub'

        lp_item = {'c_feats'   : c_feats,
                   'v_feats'   : v_feats,
                   'e_feats'   : e_feats,
                   'active'    : active,
                   'v_bounds'  : v_bounds,
                   'mps_path'  : mps_path}
        
        return lp_item

    @staticmethod
    def getitem_complete(mps_path):
        '''
        Given an LP of the form
            min c*x
            st  A_eq * x =  b_eq
                A_in * x <= b_eq

                lb <= x <= ub

        Let A = [A_eq; A_in; A_bounds] be a matrix of size (m+num_bounds,n)
        Generate a weighted complete graph G = (V, E, W)
        with    
                V = [m+1], 
                E = [m+1] x [m+1]
                W = {inner_product(i,j) : (i,j) in E}
        '''


        # Append an extra zero to node_features to account for constraints
        node_features.append(0)

        lp_item = {'c_feats':       {'A': item['A'], 'b': item['b'], 'c': item['c']},
                   'n_feats': np.asarray(node_features),
                   'node_labels':   np.asarray(node_labels),
                   'in_loss':       item['in_loss'], 
                   'mps_path':      mps_path}

        return lp_item

    @staticmethod
    def getitem_complete_copy(mps_path):
        '''
        Given an LP of the form
            min c*x
            st  A_eq * x =  b_eq
                A_in * x <= b_eq

                lb <= x <= ub

        Let A = [A_eq; A_in; A_bounds] be a matrix of size (m+num_bounds,n)
        Generate a weighted complete graph G = (V, E, W)
        with    
                V = [m+1], 
                E = [m+1] x [m+1]
                W = {inner_product(i,j) : (i,j) in E}
        '''

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
        node_labels = [None] * item['A'].shape[0]

        # Read info  
        info_path = os.path.splitext(mps_path)[0] + '.info'
        with open(info_path, 'r') as f:
            lp_info = json.load(f)
        x       = lp_info['x_opt']
        active  = lp_info['active']

        for c in model.getConstrs():
            node_labels[name2index[c.ConstrName]] = 0
        for c in active:
            node_labels[name2index[c]] = 1

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

        lp_item = {'lp':            {'A': item['A'], 'b': item['b'], 'c': item['c']},
                   'node_features': np.asarray(node_features),
                   'node_labels':   np.asarray(node_labels),
                   'in_loss':       item['in_loss'], 
                   'mps_path':      mps_path}

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

