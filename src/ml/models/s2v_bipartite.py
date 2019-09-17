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

        # Some functions
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax    = nn.Softmax(dim=1)

        # -------------------
        # theta0: Bias term
        # -------------------
        self.theta0 = nn.Parameter(torch.randn(p,1).type(self.dtype), requires_grad=True)

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

    def _s2v(mu, node_features, edge_features):

        return mu

    def _bipartite_forward(lp_data):

        # get inputs
        c_features = lp_data['c_features']
        v_features = lp_data['v_features']
        e_features = lp_data['e_features']

        # mu
        mu = Variable(torch.zeros(self.p,m+n).type(self.dtype))
        if self.with_cuda:
            mu = mu.cuda()

        # s2v
        for t in range(self.T):
            mu = self._s2v(mu, node_features, edge_features)

        return scores_out

    def forward(self, lp_data):
        return self._bipartite_forward(lp_data)

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

