from torch.autograd import Variable
import torch
import numpy as np

def to_batches(data, batch_size):
    batched = []
    batch   = 0
    for batch in range(batch_size):
        dp = {}
        dp['x']      = {}   
        dp['x']['A'] = data['x']['A'][batch, :, :].unsqueeze(0)
        dp['x']['b'] = data['x']['b'][batch, :].unsqueeze(0)
        dp['x']['c'] = data['x']['c'][batch, :].unsqueeze(0)
        dp['x']['i'] = torch.LongTensor([data['x']['i'][batch]])
        dp['y']      = torch.LongTensor([data['y'][batch]])
        batched.append(dp)
    return batched

def train_net(net, criterion, optimizer, trainloader, num_epochs, batch_size, verbose=False):
    verbose=True
    losses  = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # batch the data
            batched = to_batches(data, batch_size)

            # zero the parameter gradients
            optimizer.zero_grad()
            for dp in batched:
                # x: input, y: output
                inputs = dp['x']
                label  = Variable(dp['y'].type(torch.LongTensor))

                # forward + backward + optimize
                outputs = net(inputs)
                loss    = criterion(outputs, label)
                loss.backward()
            optimizer.step()

            # Loss and running loss
            loss_val        = float(loss.data[0])
            losses          = np.append(losses, loss_val)
            running_loss    += loss_val

            # Stats
            if i % 1000 == 0 and verbose:
                print('[%d, %5d] loss=%.7f' % (epoch+1, i+1, loss_val))
    if verbose:
        print('Finished training')
    return losses

