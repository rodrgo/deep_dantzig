import numpy as np
from scipy.sparse import csr_matrix

def np_to_sparse_serial(A):
    m, n  = A.shape
    A_csr = csr_matrix(A)
    a = A_csr.tocoo()
    d = {'row': a.row.tolist(), 
         'col': a.col.tolist(), 
         'data': a.data.tolist(), 
         'm': m,
         'n': n} 
    return d

def sparse_serial_to_np(d):
    row = np.array(d['row'])
    col = np.array(d['col'])
    data = np.array(d['data'])
    m = d['m']
    n = d['n']
    return csr_matrix((data, (row, col)), shape=(m,n)).toarray()
    
