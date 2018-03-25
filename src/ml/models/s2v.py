import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

class Model(nn.Module):

    dtype = torch.FloatTensor
    #dtype = torch.cuda.FloatTensor

    def __init__(self, m, n, p, T):
        super(Model, self).__init__()
        '''
        This implements a variation of the structure2vec 
        message passing procedure, presented in
            Dai et al. - "Learning Combinatorial Optimization
            Algorithms over Graphs"
            https://arxiv.org/pdf/1704.01665.pdf
        
        mu_     = mu
        mu[v]   = relu(  theta1r  * f(v)                                            if v <  m 
                       + theta1c  * f(v)                                            if v == m
                       + theta2rr * mu_[Nrr(v)].sum()                               if v <  m
                       + theta2rc * mu_[Nrc(v)].sum()                               if v <  m
                       + theta2cr * mu_[Ncr(v)].sum()                               if v == m
                       + theta3rr * sum([relu(theta4rr * w(v,u)) for u in Nrr(v)])  if v <  m
                       + theta3rc * sum([relu(theta4rc * w(v,u)) for u in Nrc(v)])  if v <  m
                       + theta3cr * sum([relu(theta4cr * w(v,u)) for u in Ncr(v)])  if v == m
                       )
        where,
            w(v,u) = f(v)*f(u)
        and
            f(v) = normalised((A[v,:], b[v])) if v <  m
            f(v) = normalised((c     , 0   )) if v == m

        NOTES:
            if v <  m,
                mu_[Nrr(v)].sum() == C - mu_[v] for some constant vector C
                mu_[Nrc(v)].sum() == mu_[m]
            if v == m,
                mu_[Ncr(v)].sum() == C2         for some constant vector C2
        '''

        self.torch_version = torch.__version__ 

        # Types
        # @params
        # Problem is of dimension (m,n)
        # p: Dimension of feature space
        # T: Number of rounds in message-passing
        self.m = m
        self.n = n
        self.p = p
        self.T = T 

        # Scaling to reduce variance
        scale = lambda x : math.sqrt(1/x)

        # -------------------
        # theta1's
        # -------------------
        self.theta1r = nn.Parameter(scale(n) * torch.randn(p,n+1).type(self.dtype), requires_grad=True)
        self.theta1c = nn.Parameter(scale(n) * torch.randn(p,n+1).type(self.dtype), requires_grad=True)

        # -------------------
        # theta2's
        # -------------------
        # Message passing: rows -> rows
        self.theta2rr = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        # Message passing: rows -> c
        self.theta2rc = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        # Message passing: c    -> rows
        self.theta2cr = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # -------------------
        # theta3's
        # -------------------
        # Message passing: rows -> rows 
        self.theta3rr = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        # Message passing: rows -> c
        self.theta3rc = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        # Message passing: c -> rows
        self.theta3cr = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # -------------------
        # theta4's
        # -------------------
        # Message passing: rows -> rows 
        self.theta4rr = nn.Parameter(scale(p) * torch.randn(p,1).type(self.dtype), requires_grad=True)
        # Message passing: rows -> c
        self.theta4rc = nn.Parameter(scale(p) * torch.randn(p,1).type(self.dtype), requires_grad=True)
        # Message passing: c -> rows
        self.theta4cr = nn.Parameter(scale(p) * torch.randn(p,1).type(self.dtype), requires_grad=True)

        # -------------------
        # theta6's
        # -------------------
        # Aggregators
        self.theta6r = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.theta6c = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # -------------------
        # theta7
        # -------------------
        self.theta7 = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # -------------------
        # Linear layer: Probabilities
        # -------------------
        self.theta8 = nn.Linear(2*p, 2)

    def _s2v(self, A, b, c, z, mu):
        '''
        structure2vec
        '''
        def f(u):
            # Build vectors and normalise
            if u < self.m:
                if self.torch_version == '0.3.0.post4': 
                    vec = torch.cat((A[:,u,0:self.n], b[:,u]), 1)
                else:
                    vec = torch.cat((A[:,u,0:self.n], b[:,u].unsqueeze(0)), 1)
                vec.div_(torch.norm(vec, 2)) 
                return vec 
            else:
                vec = torch.cat((c, Variable(torch.zeros(1,1).type(self.dtype))), 1)
                vec.div_(torch.norm(vec, 2)) 
                return vec

        Nrr  = lambda v    : [w for w in range(self.m) if w != v]
        Nrc  = lambda v    : [self.m]
        Ncr  = lambda v    : list(range(self.m))
        w    = lambda u, v : float(f(u).dot(f(v)).data)

        for t in range(self.T):
            mu_ = mu
            for v in range(self.m+1):
                # Neighbours
                rr = Nrr(v) # rows to rows
                rc = Nrc(v) # rows to cost
                cr = Ncr(v) # cost to rows
                # term1: Isolated term
                if v < self.m:
                    term1 = torch.mm(self.theta1r, torch.t(f(z)))
                else:
                    term1 = torch.mm(self.theta1c, torch.t(f(z)))
                # term2: Linear transformations
                if v < self.m:
                    term2rr = torch.mm(self.theta2rr, mu_[:,[v]].sum(dim=1).unsqueeze(1))
                    term2rc = torch.mm(self.theta2rc, mu_[:, rc].sum(dim=1).unsqueeze(1))
                    term2   = term2rr + term2rc
                else:
                    term2   = torch.mm(self.theta2cr, mu_[:, cr].sum(dim=1).unsqueeze(1))
                # term3: Non-linear transformations
                if v < self.m:
                    # term3rr
                    term3rr = Variable(torch.zeros(self.p,1).type(self.dtype))
                    for u in rr:
                        term3rr = term3rr + F.relu(self.theta4rr * w(v,u))
                    term3rr = torch.mm(self.theta3rr, term3rr)
                    # term3rc
                    u = rc[0] # only one element, self.m
                    term3rc = torch.mm(self.theta3rc, F.relu(self.theta4rc * w(v,u)))
                    # term3
                    term3 = term3rr + term3rc
                else:
                    # term3cr
                    term3cr = Variable(torch.zeros(self.p,1).type(self.dtype))
                    for u in cr:
                        term3cr = term3cr + F.relu(self.theta4cr * w(v,u))
                    term3cr = torch.mm(self.theta3cr, term3cr)
                    # term3
                    term3 = term3cr
                mu[:,v] = F.relu(term1 + term2 + term3)

        # Pool embeddings        
        # term6
        cols   = range(self.m)
        term6r = torch.mm(self.theta6r, torch.t(torch.sum(mu[:,  cols  ],dim=1).unsqueeze(0)))
        term6c = torch.mm(self.theta6c, torch.t(torch.sum(mu[:,[self.m]],dim=1).unsqueeze(0)))
        term6  = term6r + term6c
        # term7 
        term7      = torch.mm(self.theta7, torch.transpose(mu[:,z].unsqueeze(0),0,1))
        feature_z  = torch.t(F.sigmoid(torch.cat((term6,term7),0)))
        return feature_z

    def forward(self, data_x):
        '''
        Forward pass
        '''
        # get inputs
        A   = Variable(data_x['A'].type(self.dtype))
        b   = Variable(data_x['b'].type(self.dtype))
        c   = Variable(data_x['c'].type(self.dtype))
        row = Variable(data_x['i'].type(self.dtype))

        # Extract row value 
        r = int(row.data)

        # [0, ..., m-1] : constraints 
        # [m]           : cost vector
        row_initialise = True
        if row_initialise:
            '''
            mu has dimensions (p, m+1)
                 | A'  c' |
            mu = | b'  0  |
                 | 0   0  |
            '''
            Ab = torch.cat((A.squeeze(0), torch.t(b)), 1)
            c0 = torch.cat((c           , Variable(torch.zeros(1,1).type(self.dtype))), 1)
            mu = torch.cat((torch.t(Ab) , torch.t(c0)), 1)
            pp = self.p - mu.size(0)
            if pp > 0:
                mu = torch.cat((mu, Variable(torch.zeros(pp, self.m+1).type(self.dtype))),0)
            elif pp < 0:
                raise ValueError
        else:
            mu = Variable(torch.zeros(self.p,self.m+1).type(self.dtype))

        # structure2vec
        feature_z = self._s2v(A, b, c, r, mu)

        # Classification
        res = self.theta8(feature_z)

        return res

