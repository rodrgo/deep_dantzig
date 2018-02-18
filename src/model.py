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

    def __init__(self, m, n, p):
        super(Model, self).__init__()
        # Types
        # @params
        # Problem is of dimension (m,n)
        # and 'p' is dimension of feature space
        self.m = m
        self.n = n
        self.p = p
        # Variance reductor
        scale = lambda x : math.sqrt(1/x)
        # theta's
        self.theta1 = nn.Parameter(scale(n) * torch.randn(p,n).type(self.dtype), requires_grad=True)
        self.theta2 = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.theta3 = nn.Parameter(scale(p) * torch.randn(p,1).type(self.dtype), requires_grad=True)
        # c's
        self.theta2c = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.theta3c = nn.Parameter(scale(p) * torch.randn(p,1).type(self.dtype), requires_grad=True)
        # out's
        self.theta4a = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        self.theta4b = nn.Parameter(scale(p) * torch.randn(p,p).type(self.dtype), requires_grad=True)
        # Number of steps in greedy algorithm
        # There are 'm' nodes for the constraints, 
        # '1' node for the cost, and we do one
        # more step to close the propagation loop
        self.T = m + 2
        # Declare parameters
        self.mu = None

    def forward(self, A, b, c, z):
        '''
        Features are created using a variation of the
        structure2vec transform presented in
            Dai et al. - "Learning Combinatorial Optimization
            Algorithms over Graphs"
            https://arxiv.org/pdf/1704.01665.pdf
        '''

        # -----------------
        # structure2vec
        # -----------------

        def f(u):
            if u < self.m:
                return torch.cat((A[:,u,0:self.n], b[:,1]), 1)
            else:
                return torch.cat((c, b[:,1]), 1)
        N   = lambda v    : [w for w in range(self.m) if w != v]
        w   = lambda u, v : float(f(u).dot(f(v)).data)
        wc  = lambda u    : float(c.dot(A[:,u,0:self.n]).data)
        row = lambda u    : A[:,u,:]

        # Scaling factor, otherwise blows up
        K = 0.01

        # Indices [0, ..., m-1] are for constraints, [m] is for cost
        mu     = Variable(torch.randn(p, self.m+1).type(self.dtype))
        z_int  = int(z.data)
        for t in range(int(self.m/2)): #range(self.T):
            mu_tmp = mu # TODO: Check whether mu.clone() is necessary
            # mu_tmp = mu.clone()
            for v in range(self.m+1):
                # Nv
                Nv = N(v)
                # term1: Isolated term
                term1  = K * torch.mm(self.theta1, torch.transpose(row(z_int), 0, 1))
                # term2: Linear transformations
                term2  = K * torch.mm(self.theta2, mu_tmp[:, Nv].sum(dim=1).unsqueeze(1))
                term2c = K * torch.mm(self.theta2c, mu_tmp[:, [self.m]].sum(dim=1).unsqueeze(1))
                # term3: Non-linear transformations
                for u in Nv:
                    if u == Nv[0]:
                        term3 = F.relu(self.theta3 * w(v,u))
                    else:
                        term3 = term3 + F.relu(self.theta3 * w(v,u))
                term3  = K * term3
                term3c = K * F.relu(self.theta3c * wc(u))
                #term3c = term3c.div(term3c.norm(p=2, dim=1, keepdim=True)) # normalize
                # Aggregate
                mu[:,v] = F.relu(term1 + term2 + term2c + term3 + term3c)
        # Pool embeddings        
        term4a = self.theta4a.mm(torch.transpose(torch.sum(mu[:,N(self.m)], dim=1).unsqueeze(0), 0, 1))
        term4b = self.theta4b.mm(torch.transpose(mu[:,z_int].unsqueeze(0), 0, 1))
        feature_z = torch.transpose(torch.cat((term4a, term4b), 0), 0, 1)
        return feature_z

if __name__ == '__main__':
    import torch.optim as optim
    from utils import LPDataset
    dtype = torch.FloatTensor

    # Params
    N = 300
    n = 5 
    m = 10
    p = 10

    trainset = LPDataset(m=m, n=n, seed=3231, N=N, num_lps=1)
    testset  = LPDataset(m=m, n=n, seed=3231, N=N, num_lps=1)

    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=1, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,
            batch_size=1, shuffle=False, num_workers=2)

    # Initialise net
    net = Model(m, n, p)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
            lr=0.001, momentum=0.9)

    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get inputs
            A       = Variable(data['x']['A'].type(dtype))
            b       = Variable(data['x']['b'].type(dtype))
            c       = Variable(data['x']['c'].type(dtype))
            z       = Variable(data['x']['i'].type(dtype))
            labels  = Variable(data['y'].type(torch.LongTensor))

            print('i=%d, label=%d' % (z.data, labels.data))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(A, b, c, z)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9:
                print('[%d, %5d] loss: %.7f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished training')

    # ----------------
    # Test dataset
    # ----------------

    correct = 0
    total   = 0
    num     = 0
    for data in testloader:
        A       = Variable(data['x']['A'].type(dtype))
        b       = Variable(data['x']['b'].type(dtype))
        c       = Variable(data['x']['c'].type(dtype))
        z       = Variable(data['x']['i'].type(dtype))
        labels  = Variable(data['y'].type(torch.LongTensor))
        outputs = net(A, b, c, z)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        predicted = predicted.numpy()
        labels = labels.data.numpy()
        print('predicted = %d, real = %d' % (predicted[0], labels[0]))
        correct += (predicted == labels).sum()
        num += 1

    print('Accuracy of the network on the %d test LP instances: %d %%' % (num, 100 * correct / total))

