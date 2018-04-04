import torch

def LongTensor(var, cuda=False):
    if cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    var = var.type(dtype)
    if cuda:
        var = var.cuda()
    return var

