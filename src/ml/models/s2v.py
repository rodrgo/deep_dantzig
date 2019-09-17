import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

class Model(nn.Module):
    '''
    This implements a variation of the structure2vec 
    message passing procedure, presented in
        Dai et al. - "Learning Combinatorial Optimization
        Algorithms over Graphs"
        https://arxiv.org/pdf/1704.01665.pdf
    '''

    logsoftmax = nn.LogSoftmax(dim=1)
    softmax    = nn.Softmax(dim=1)

    def __init__(self, graph, p, rounds_s2v, on_cuda=False):
        # p: Dimension of embedding space

        super(Model, self).__init__()
        self.torch_version = torch.__version__ 

        self.verbose = False

        self.on_cuda = on_cuda 
        if self.on_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.graph = graph
        self.p     = p
        self.T     = rounds_s2v
        print('rounds_s2v: %d ' % (self.T))

        if self.graph == 'bipartite':
            self._init_bipartite()
        elif self.graph == 'complete':
            self._init_complete()

    def forward(self, item):

        if self.graph == 'complete':
            scores_out = self._forward_complete(item)
        elif self.graph == 'bipartite':
            scores_out = self._forward_bipartite(item)
        else:
            raise(ValueError('Graph not recognised'))

        return scores_out

    def require_grads(self, req=True):
        for param in self.parameters():
            param.requires_grad = req

    def _init_complete(self):

        p = self.p
        C = math.sqrt(1/p)
    
        # Bias and node features
        self.t0 = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)
        self.t1 = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)

        # Message passing: (embeddings)
        self.t2rr = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t2rc = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t2cr = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # Message passing: (outer products)
        self.t3rr = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t3rc = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t3cr = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)

        self.t4rr = nn.Parameter(C * torch.randn(p,1).type(self.dtype), requires_grad=True)
        self.t4rc = nn.Parameter(C * torch.randn(p).type(self.dtype), requires_grad=True)
        self.t4cr = nn.Parameter(C * torch.randn(p).type(self.dtype), requires_grad=True)

        # Features
        self.t6r  = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t6c  = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t7   = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # Probabilities/Scores
        self.t8 = nn.Parameter(C * torch.randn(2,2*p).type(self.dtype), requires_grad=True)

    def _s2v_complete(self, mu, node_feats, edge_feats):
        '''
            m+1 nodes, (1 for each constraint + 1 for cost vector)
            mu: embeddings
            node_feats: (binary vector)
            edge_feats: (symmetric matrix)
        '''

        m = node_feats.shape[1]-1
        C = 1/float(m)
        W = edge_feats
        
        #u1 = t0 + t1 * node_feats
        u1  = self.t0 + self.t1.matmul(node_feats)

        #u2 = [t2rr * mu[:,r] + t2rc * mu[:,c], t2c * mu[:,r].sum(dim=1)] 
        u2r = self.t2rr.matmul(mu[:,:m]) + self.t2rc.matmul(mu[:,[m]]).repeat(1, m)
        u2c = self.t2cr.matmul(C * mu[:,:m].sum(dim=1)).unsqueeze(1)
        u2  = torch.cat((u2r, u2c), 1)

        #u3 = [t3rr * relu_rr + t3rc * relu_rc, t3cr * relu_cr]
        relu_rr = F.relu(torch.bmm(self.t4rr.unsqueeze(0).expand(m,-1,-1), W[:m,:m].unsqueeze(1))).sum(dim=2).unsqueeze(2) 
        u3rr    = torch.t(torch.bmm(self.t3rr.expand(m,-1,-1), relu_rr).squeeze(2))

        relu_rc = F.relu(torch.ger(self.t4rc, W[m,:m].squeeze(0))).sum(dim=1).unsqueeze(1)
        u3r     = u3rr + self.t4rc.matmul(relu_rc).expand(m) 

        relu_cr = F.relu(torch.ger(self.t4cr, W[:m,m])).sum(dim=1).unsqueeze(1)
        u3cr    = self.t3cr.matmul(relu_cr)
        u3      = torch.cat((u3r, u3cr), 1)

        return F.relu(u1 + u2 + u3)

    def _forward_complete(self, item):

        # get inputs
        A = item['A']
        b = item['b']
        c = item['c']
        node_feats = item['node_features'].type(self.dtype)

        m = max(list(b.size()))

        REDUCE_LOSS = False
        if REDUCE_LOSS:
            in_loss = list(range(m))
        else:
            in_loss = item['in_loss']
       
        if self.on_cuda:
            A  = A.cuda()
            b  = b.cuda()
            c  = c.cuda()

        Ab = F.normalize(torch.cat((A.squeeze(0), b.squeeze(0).unsqueeze(1)), 1), p=2, dim=1)
        c0 = torch.from_numpy(np.concatenate((c, np.zeros((1,1))), axis=1)).type(self.dtype)
        G  = torch.cat((Ab.type(self.dtype), c0.type(self.dtype)), 0)

        if self.on_cuda:
            G  = G.cuda()
            node_feats = node_feats.cuda()

        node_feats = Variable(node_feats)
        G = Variable(G)
        W = torch.matmul(G, torch.transpose(G, 0, 1))

        # W = G*G' with zero diagonals
        mask = Variable(torch.eye(W.shape[0]))
        if self.on_cuda:
            mask = mask.cuda()
        mask = torch.eq(mask, 1)
        W.masked_fill_(mask, 0)

        # Embeddings
        # Initialise embeddings to None
        # [0, ..., m-1] : constraints 
        # [m]           : cost vector
        mu = Variable(torch.zeros(self.p,m+1).type(self.dtype))
        if self.on_cuda:
            mu = mu.cuda()

        # structure2vec
        #   Create embeddings
        for t in range(self.T):
            mu = self._s2v_complete(mu, node_feats, W)

        # Features
        C  = 1/(float(m))
        u6 = self.t6r.matmul(C * mu[:,:m].sum(dim=1)) + self.t6c.matmul(mu[:,m])

        u7     = self.t7.matmul(mu[:,in_loss])
        feats  = F.relu(torch.cat((u6.unsqueeze(1).repeat(1,len(in_loss)), u7),0))
        scores = torch.bmm(self.t8.unsqueeze(0).repeat(len(in_loss),1,1), torch.t(feats).unsqueeze(2)).squeeze(2)

        self.probs = self.softmax(scores)

        return self.logsoftmax(scores)

    def _init_bipartite(self):
        p = self.p
        C = math.sqrt(1/p)
        K = lambda x : math.sqrt(1/x)

        # Bias and node features
        self.t0  = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)
        self.t1c = nn.Parameter(K(4) * torch.randn(p,4).type(self.dtype), requires_grad=True)
        self.t1v = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)

        # Message passing: (embeddings)
        self.t2c = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t2v = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # Message passing: (outer products)
        self.t3c = nn.Parameter(C * torch.randn(1,p,p).type(self.dtype), requires_grad=True)
        self.t3v = nn.Parameter(C * torch.randn(1,p,p).type(self.dtype), requires_grad=True)

        self.t4c = nn.Parameter(C * torch.randn(1,p,1).type(self.dtype), requires_grad=True)
        self.t4v = nn.Parameter(C * torch.randn(1,p,1).type(self.dtype), requires_grad=True)

        # Features
        self.t6c  = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t6v  = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.t7   = nn.Parameter(C * torch.randn(p,p).type(self.dtype), requires_grad=True)

        # Probabilities/Scores
        self.t8 = nn.Parameter(K(2*p+4) * torch.randn(2,2*p+4).type(self.dtype), requires_grad=True)

    def _s2v_bipartite(self, mu, c_feat, v_feat, A, adj):
        # A is edge_feat

        m,  n   = c_feat.shape[0], v_feat.shape[0]
        Cm, Cn  = 1/float(m), 1/float(n)
        Cm, Cn  = 1/float(math.sqrt(m)), 1/float(math.sqrt(n))
        ic, iv  = list(range(0,m)), list(range(m,m+n))

        C   = 1/float(m + n)

        term1 = self.t0 + torch.cat((self.t1c.matmul(c_feat.t()), self.t1v.matmul(v_feat.t())), 1)

        cadj  = F.normalize(adj, p=1, dim=0)
        radj  = F.normalize(adj.t(), p=1, dim=0)
        term2 = torch.cat((self.t2c.matmul(mu[:,ic].matmul(cadj)), self.t2v.matmul(mu[:,iv].matmul(radj))), 1)

        # (m,p,1),(m,1,n) -> (m,p,n) -> (m,p)
        # (m,p,p),(m,p,1) -> (m,p,1) -> (m,p)
        relu  = F.relu(torch.bmm(self.t4c.expand(m,-1,-1), A.unsqueeze(1))).sum(dim=2)
        u3c   = torch.bmm(self.t3c.expand(m,-1,-1), relu.unsqueeze(2)).squeeze(2).t()

        relu  = F.relu(torch.bmm(self.t4v.expand(n,-1,-1), A.t().unsqueeze(1))).sum(dim=2)
        u3v   = torch.bmm(self.t3v.expand(n,-1,-1), relu.unsqueeze(2)).squeeze(2).t()

        term3 = torch.cat((u3c, u3v), 1)
        if self.verbose:
            print('\tterm1:  %g ' % (float(torch.max(torch.max(torch.abs(term1))))))
            print('\tterm2:  %g ' % (float(torch.max(torch.max(torch.abs(term2))))))
            print('\tu3c:  %g ' % (float(torch.max(torch.max(torch.abs(u3c))))))
            print('\tu3v:  %g ' % (float(torch.max(torch.max(torch.abs(u3v))))))
            print('\tterm3:  %g ' % (float(torch.max(torch.max(torch.abs(term3))))))

        #return torch.clamp(F.relu(term1 + term2 + term3), -10.00, 10.00)
        return F.relu(term1 + term2 + term3)

    def _forward_bipartite(self, item):

        # get inputs
        c_feats = item['c_feats'] # is_inequality, rhs, is_bound
        v_feats = item['v_feats'] # obj 
        e_feats = item['e_feats'] # i, coeffs
        in_loss = item['in_loss']
        dims    = item['dims']

        m, n    = int(dims['m']), int(dims['n']) 
        C       = 1/float(m+n)

        # mu
        mu = Variable(torch.zeros(self.p,m+n).type(self.dtype))
        if self.on_cuda:
            mu = mu.cuda()

        # squeeze
        c_feats = Variable(c_feats)
        v_feats = Variable(v_feats)

        # e_feats
        to_int  = lambda f : [int(f[i]) for i in range(len(f))]
        i       = torch.LongTensor([to_int(f) for f in e_feats['i']])
        v       = torch.FloatTensor([float(f) for f in e_feats['coeffs']])
        A       = Variable(torch.sparse.FloatTensor(i.t(), v, torch.Size([m,n])).to_dense())

        # adjacency
        nnz     = len(e_feats['coeffs'])
        v       = torch.FloatTensor([1] * nnz)
        adj     = Variable(torch.sparse.FloatTensor(i.t(), v, torch.Size([m,n])).to_dense())

        if self.on_cuda:
            c_feats = c_feats.cuda()
            v_feats = v_feats.cuda()
            A       = A.cuda()
            adj     = adj.cuda()

        # Normalise
        Ab           = F.normalize(torch.cat((A, -c_feats[:,1].unsqueeze(1)), 1), p=2, dim=1)
        A            = Ab[:,:n]
        c_feats[:,1] = -Ab[:,n]

        ## Add cosine and cosine/b to features.
        cosines = torch.matmul(A, v_feats[:,0])
        c_feats = torch.cat((c_feats, cosines.unsqueeze(1)), 1)

        # structure2vec
        #   Create embeddings
        for t in range(self.T):
            mu = self._s2v_bipartite(mu, c_feats, v_feats, A, adj)

        # Features
        Cm, Cn  = 1/float(math.sqrt(m)), 1/float(math.sqrt(n))
        Cm, Cn  = 1/float(m), 1/float(n)
        ic, iv  = list(range(0,m)), list(range(m,m+n))

        u6      = self.t6c.matmul(Cm * mu[:,ic].sum(dim=1)) + self.t6v.matmul(Cn * mu[:,iv].sum(dim=1))
        embed   = F.relu(torch.cat((u6.unsqueeze(1).expand(-1,len(in_loss)), self.t7.matmul(mu[:,in_loss])),0))
        embed   = torch.cat((embed, c_feats[in_loss,:].t()), 0)
        scores  = self.t8.matmul(embed).t()

        self.probs = self.softmax(scores)
        if self.verbose:
            print('\tu6:  %g ' % (float(torch.max(torch.max(torch.abs(u6))))))
            print('\tembed:  %g ' % (float(torch.max(torch.max(torch.abs(embed))))))
            print('\tscores:  %g ' % (float(torch.max(torch.max(torch.abs(scores))))))
            print('\tmin probs:  %g ' % (float(torch.min(torch.min(self.probs)))))
            print('\tmax probs:  %g ' % (float(torch.max(torch.max(self.probs)))))

        return self.logsoftmax(scores)

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

