import torch

def batched(data, batch_size, graph_structure):
    if graph_structure == 'complete':
        for batch in range(batch_size):
            x = {}
            x['A'] = data['lp']['A'][batch, :, :].unsqueeze(0)
            x['b'] = data['lp']['b'][batch, :].unsqueeze(0)
            x['c'] = data['lp']['c'][batch, :].unsqueeze(0)
            x['node_features']  = data['node_features']
            x['in_loss'] = [int(p) for p in data['in_loss']]
            y = data['node_labels'].squeeze(0)[x['in_loss']]
            yield x, y
    elif graph_structure == 'bipartite':
        for batch in range(batch_size):
            x = {}
            x['c_feats'] = data['c_feats'].squeeze(0)
            x['v_feats'] = data['v_feats'].squeeze(0)
            x['e_feats'] = data['e_feats']
            x['dims']    = data['dims']
            x['in_loss'] = [int(p) for p in data['in_loss']]
            y            = data['c_labels'].squeeze(0)[x['in_loss']]
            yield x, y
    else:
        raise(ValueError('graph_structure not recognised'))
