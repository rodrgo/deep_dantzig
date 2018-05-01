import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

class Model(nn.Module):

    def __init__(self, p, T, cuda=False):
        super(Model, self).__init__()
        self.with_cuda = cuda 
        if self.with_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
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
        self.p = p
        self.T = T 

        # Scaling to reduce variance
        scale = lambda x : math.sqrt(1/x)

        # -------------------
        # theta1's
        # -------------------
        self.theta1 = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)

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
        #self.theta8 = nn.Linear(2*p, 2)
        self.theta8 = nn.Parameter(scale(p) * torch.randn(2,2*p).type(self.dtype), requires_grad=True)

        # Other placeholders
        self.theta6 = None

    def require_grads(self, req=True):
        for param in self.parameters():
            param.requires_grad = req

    def _s2v(self, mu, z, W, m):
        '''
            structure2vec
            (self, mu, node_feat, edge_feat, adjacency)
            z = node_features
            W = dense symmetric matrix of edge features
            adjacency = fully connected
        '''
        # TODO: Expand vs accumulate, recycle variable pointers.

        scale  = 1/(float(m))
        
        '''
        term1 = theta1 * node_features
        '''
        term1  =  self.theta1.matmul(z)

        '''
        term2 = [theta2rr * mu[:,r] + theta2rc * mu[:,c],
                 theta2c  * mu[:,r].sum(dim=1)] 
        '''
        term2rr = self.theta2rr.matmul(mu[:,:m])
        term2rc = self.theta2rc.matmul(mu[:,[m]]).repeat(1, m)
        term2c  = self.theta2cr.matmul(scale * mu[:,:m].sum(dim=1)).unsqueeze(1)
        term2   = torch.cat((term2rr + term2rc, term2c), 1)

        '''
        term3 = [theta3rr * relu_rr + theta3rc * relu_rc,
                 theta3cr * relu_cr]
        '''
        # ((m,p,1),(m,1,m))->(m,p,m)->(m,p)->(m,p,1)
        relu_rr = F.relu(torch.bmm(self.theta4rr.unsqueeze(0).expand(m,-1,-1), W[:m,:m].unsqueeze(1))).sum(dim=2).unsqueeze(2)
        # ((p,p)->(m,p,p),(m,p,1))->(m,p,1)->(m,p)->(p,m)
        term3rr = torch.t(torch.bmm(self.theta3rr.expand(m,-1,-1), relu_rr).squeeze(2))

        # ((p,1),(1,m))->(p,m)->(p,1)
        relu_rc = F.relu(torch.ger(self.theta4rc.squeeze(1), W[m,:m].squeeze(0))).sum(dim=1).unsqueeze(1)
        term3rc = self.theta3rc.matmul(relu_rc)
        term3r  = term3rr + term3rc.expand(-1,m) 

        # ((p),(m))->(p,m)->(p,1)
        relu_cr = F.relu(torch.ger(self.theta4cr.squeeze(1), W[:m,m])).sum(dim=1).unsqueeze(1)
        term3cr = self.theta3cr.matmul(relu_cr)
        term3   = torch.cat((term3r, term3cr), 1)

        return F.relu(term1 + term2 + term3)

    def forward(self, lp_data):
        '''
        Forward pass
        '''

        # get inputs
        A = lp_data['A']
        b = lp_data['b']
        c = lp_data['c']
        node_features = lp_data['node_features'].type(self.dtype)
        in_loss = lp_data['in_loss']

        m = max(list(b.size()))
       
        if True:
            if self.with_cuda:
                A = A.cuda()
                b = b.cuda()
                c = c.cuda()
            Ab = torch.cat((A.squeeze(0), b.squeeze(0).unsqueeze(1)), 1)
            Ab = torch.div(Ab, torch.max(torch.max(torch.abs(Ab),0)[0]))
            c0 = torch.from_numpy(np.concatenate((c, np.zeros((1,1))), axis=1)).type(self.dtype)
            c0 = torch.div(c0, torch.max(torch.abs(c0)))
            G  = torch.cat((Ab.type(self.dtype), c0.type(self.dtype)), 0)
        else:
            Ab = np.concatenate((A.squeeze(0), np.transpose(b)), axis=1)
            Ab = Ab / np.max(np.absolute(Ab))

            c0 = np.concatenate((c, np.zeros((1,1))), axis=1)
            c0 = c0 / np.max(np.absolute(c0))

            # Create single element from this
            # G.shape = (m+1, n+1)
            # W = G*G' will give the weights w(u,v)
            G = np.concatenate((Ab, c0), axis=0)

            # Torchify
            G = torch.from_numpy(G).type(self.dtype)
            if self.with_cuda:
                node_features = node_features.cuda()
                G = G.cuda()

        node_features = Variable(node_features)
        G = Variable(G)
        W = torch.matmul(G, torch.transpose(G, 0, 1))

        # W = G*G' with zero diagonals
        mask = Variable(torch.eye(W.shape[0]))
        if self.with_cuda:
            mask = mask.cuda()
        mask = torch.eq(mask, 1)
        W.masked_fill_(mask, 0)

        # Embeddings
        # Initialise embeddings to None
        # [0, ..., m-1] : constraints 
        # [m]           : cost vector

        mu = Variable(torch.zeros(self.p,m+1).type(self.dtype))
        if self.with_cuda:
            mu = mu.cuda()

        # structure2vec
        #   Create embeddings
        for t in range(self.T):
            mu = self._s2v(mu, node_features, W, m)

        # Features
        # Sum of all columns
        scale  = 1/(float(m))
        term6  = self.theta6r.matmul(scale * mu[:,:m].sum(dim=1)) + self.theta6c.matmul(mu[:,m])

        ALL_IN_LOSS = False
        if ALL_IN_LOSS:
            term7  = self.theta7.matmul(mu[:,:m])
            feats  = F.relu(torch.cat((term6.unsqueeze(1).repeat(1,m), term7),0))
            # ((m, 2, 2p), (m, 2p, 1))
            scores = torch.bmm(self.theta8.unsqueeze(0).repeat(m,1,1), torch.t(feats).unsqueeze(2)).squeeze(2)
        else:
            term7  = self.theta7.matmul(mu[:,in_loss])
            feats  = F.relu(torch.cat((term6.unsqueeze(1).repeat(1,len(in_loss)), term7),0))
            # ((m, 2, 2p), (m, 2p, 1))
            scores = torch.bmm(self.theta8.unsqueeze(0).repeat(len(in_loss),1,1), torch.t(feats).unsqueeze(2)).squeeze(2)

        return scores

if __name__ == '__main__':
    from data.plnn_dataset import DatasetPLNN
    from torch.utils.data import DataLoader
    from ml.types import LongTensor
    from ml.test import test_binary_classification as tester
    import torch.optim as optim

    seed        = 3
    bs          = 1
    num_epochs  = 10000
    t           = 2 
    lr          = 0.1
    mtm         = 0.9
    wd          = 0
    p           = 15
    cuda        = True

    # Dataset
    trainset = DatasetPLNN(num_lps=1, test=False, seed=seed)
    testset  = DatasetPLNN(num_lps=1, test=True , seed=seed)

    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=1,  shuffle=False)

    # Get params
    lp_params  = trainset.get_lp_params()
    assert(len(lp_params) == 1)
    m       = lp_params[0]['m'] 
    n       = lp_params[0]['n'] 

    # model
    model       = Model(m, n, p, t, cuda=cuda)
    if cuda:
        model   = model.cuda()

    # optimization
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.SGD(model.parameters(), lr=lr, momentum=mtm, weight_decay=wd)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            labels = np.asarray(data['labels'])
            labels = Variable(LongTensor(torch.from_numpy(labels), cuda=cuda))

            optimizer.zero_grad()
            outputs = model(data['lp'])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.data[0])
        # Accuracy
        acc, correct, total = tester(testloader, model, verbose=False, cuda=cuda)
        print('epoch=%d, loss=%5.10f, correct=%d, total=%d, acc=%5.10f' % (epoch, loss_val, correct, total, acc))

