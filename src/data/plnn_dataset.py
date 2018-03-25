
from torch.utils.data.dataset import Dataset
import numpy as np
import math
import random
import sys
import os
import subprocess
import dotenv

from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import load_snapshot_and_simplify

class DatasetPLNN(Dataset):

    def __init__(self, m, n, N, num_lps=1, test=False, seed=3231):
        self.foo        = []

    def __len__(self):
        return len(self.foo)

    def __getitem__(self, idx):
        return None 

def main():
    '''
    $prop = file to prove
    $target = output file
    We only use $prop, are interested in using this file and saving the output of LinearizedNetwork
    Creating a dataset:
        1. Get only the paths of $prop
        2. call load_and_simplify with LinearizedNetwork 
        3. When the LinearizedNetwork is solving the problem, just save the linearised networks before they go into gurobi 
        No need memory or time requirements as we won't be solving the problems
    '''

    # Build relevant paths
    dotenv.load_dotenv(dotenv.find_dotenv())
    root     = os.environ.get('ROOT')

    plnnDir  = os.path.join(root, '../PLNN-verification')
    sys.path.append(plnnDir)

    data_plnn = os.path.join(root, 'data/plnn')

    result  = subprocess.run(['sh', 'plnn_dataset_paths.sh', plnnDir], stdout=subprocess.PIPE)
    result  = result.stdout.decode('utf-8').split('\n')

    # First path in each line is the property that the PLNN wants to prove
    rlvs    = [y for y in [x.split(' ')[0] for x in result] if y]
    for i, rlv in enumerate(rlvs):
        save_params = {'fpath': data_plnn, 'tag': 'problem_%d' % (i)}
        with open(rlv, 'r') as rlv_infile:
            network, domain = load_snapshot_and_simplify(rlv_infile, LinearizedNetwork, save_params)

if __name__ == '__main__':
    main()
