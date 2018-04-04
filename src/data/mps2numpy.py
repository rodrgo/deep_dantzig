from gurobipy import *
import numpy as np
import os
import dotenv

def get_expr_coeffs(expr, var_indices):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield var_indices[dvar], expr.getCoeff(i)

def constr2numpy(model, standardize=True):
    dvars   = model.getVars()
    constrs = model.getConstrs()
    var_indices = {v:j for j, v in enumerate(dvars)}
    for i, constr in enumerate(constrs):
        ai = np.zeros(len(dvars)) 
        bi = constr.RHS
        for j, coeff in get_expr_coeffs(model.getRow(constr), var_indices):
            ai[j] = coeff
        op = constr.Sense
        if standardize and op == '>':
            ai = -ai
            bi = -bi
            op = '<'
        yield i, ai, bi, op

def bounds2numpy(model):
    dvars   = model.getVars()
    bs      = []
    ais     = []
    ops     = []
    for j, dvar in enumerate(dvars):
        # lower bound
        try:
            b_lb    = -dvar.LB
            a_lb    = np.zeros(len(dvars)) 
            a_lb[j] = -1
            ais.append(a_lb)
            bs.append(b_lb)
            ops.append('<')
        except AttributeError:
            pass
        # upper bound
        try:
            b_ub    = dvar.UB
            a_ub    = np.zeros(len(dvars)) 
            a_ub[j] = 1
            ais.append(a_ub)
            bs.append(b_ub)
            ops.append('<')
        except AttributeError:
            pass
    return ais, bs, ops

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
    ops = [] 

    # Constraints
    for i, ai, bi, op in constr2numpy(model, standardize):
        A[i,:] = ai
        b[i]   = bi
        ops.append(op)

    # Lower and Upper bounds
    A_b, b_b, ops_b = bounds2numpy(model)

    # Stack
    A = np.vstack((A, A_b))
    b = np.concatenate((b, b_b))
    ops.extend(ops_b)

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

    return A, b, c, ops, obj

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

