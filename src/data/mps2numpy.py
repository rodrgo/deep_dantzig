from gurobipy import *
import numpy as np
import os
import dotenv

def get_expr_coeffs(constr, var_indices):
    for i in range(constr.size()):
        dvar = constr.getVar(i)
        yield var_indices[dvar], constr.getCoeff(i)

def constr2numpy(model, standardize=True):
    vs = model.getVars()
    cs = model.getConstrs()
    var_indices = {v:j for j, v in enumerate(vs)}
    for i, constr in enumerate(cs):
        ai = np.zeros(len(vs)) 
        bi = constr.RHS
        for j, coeff in get_expr_coeffs(model.getRow(constr), var_indices):
            ai[j] = coeff
        sense = constr.Sense
        if standardize and sense == '>':
            ai = -ai
            bi = -bi
            sense = '<'
        yield i, ai, bi, sense, constr.ConstrName

def bounds2numpy(model, m):
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

    vs      = model.getVars()
    bs      = []
    constrs = []
    cnames  = {}
    ops     = {}
    bounds  = {}
    for j, v in enumerate(vs):
        bounds[v.varName] = {'lb': None, 'ub': None}
        # lower bound
        if has_lb(v):
            bs.append(-v.LB)
            constr = np.zeros(len(vs)) 
            constr[j] = -1.0
            constrs.append(constr)

            cname = '%s_lb' % (v.varName)
            cnames[cname] = m + len(constrs) - 1
            ops[cname]    = '<'
            bounds[v.varName]['lb'] = {'val': -v.LB, 'sense': '<', 'name': cname}
        # upper bound
        if has_ub(v):
            bs.append(v.UB)
            constr = np.zeros(len(vs)) 
            constr[j] = 1.0
            constrs.append(constr)

            cname = '%s_ub' % (v.varName)
            cnames[cname] = m + len(constrs) - 1
            ops[cname]    = '<'
            bounds[v.varName]['ub'] = {'val': v.UB, 'sense': '<', 'name': cname}
    return constrs, bs, ops, cnames, bounds

def model2numpy(model, standardize=True):
    '''
    Converts an mps model into (A,b,c) such that LP is
        min c*x
        st  A*x <= b
    '''
    n   = len(model.getVars())
    m   = len(model.getConstrs())

    A   = np.zeros((m,n))
    b   = np.zeros(m)
    c   = np.zeros(n)
    csenses = {}
    cnames  = {}

    # Constraints
    for i, ai, bi, csense, cname in constr2numpy(model, standardize):
        A[i,:]          = ai
        b[i]            = bi
        csenses[cname]  = csense
        cnames[cname]   = i

    # Lower and Upper bounds
    # They all come in standardized form
    # lbs and ubs are dictionaries varName:boundValue of
    # non-trivial lower an upper bounds
    A_b, b_b, bsenses, bname2index, bounds = bounds2numpy(model, m)

    # Stack
    A = np.vstack((A, A_b))
    b = np.concatenate((b, b_b))
    for k in bsenses.keys():
        csenses[k] = bsenses[k]
    for k in bname2index.keys(): 
        cnames[k] = bname2index[k]

    c[:] = model.Obj
    sense = model.ModelSense
    if standardize and sense == -1:
        # 1 = minimize, -1 = maximize
        c = -c
        sense = 1

    if sense == -1:
        obj = 'max'
    else:
        obj = 'min'

    lp_item = {'A': A, 'b': b, 'c': c, 
        'obj': obj, 'csenses': csenses, 
        'cnames': cnames, 'bounds': bounds}

    return lp_item

def mps2numpy(fpath, standardize=True):
    model               = read(fpath)
    A, b, c, ops, obj   = model2numpy(model, standardize)
    return A, b, c, ops, obj

def main():
    model   = Model('lp')
    dotenv.load_dotenv(dotenv.find_dotenv())
    root    = os.environ.get('ROOT')
    lp_dir  = os.path.join(root, 'data/plnn')
    fpaths  = [os.path.join(lp_dir, f) for f in os.listdir(lp_dir) if f.endswith('.mps')] 

    fpath = os.path.join(lp_dir, 'problem_749_0.mps')
    A, b, c, ops, obj = mps2numpy(fpath)

if __name__ == '__main__':
    main()

