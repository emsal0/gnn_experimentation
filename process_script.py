import sys
import torch
import networkx as nx
import os.path
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
from preprocess_qm9 import DATADIR, PREFIX
from preprocess_qm9 import * 

class QM9Dataset(Dataset):
    def __init__(self, datadir=DATADIR, prefix=PREFIX):
        self.datadir = datadir
        self.prefix = prefix

    def __len__(self):
        return len(os.listdir(self.datadir))

    def __getitem__(self, idx):
        fname = os.path.join(self.datadir, self.prefix + '_' + f"{idx:06}" + '.xyz')
        dat = process(fname)
        nx_graph = rdkit_process(dat)
        return from_networkx(nx_graph)
    
qm9 = QM9Dataset('xyzfiles', 'dsgdb9nsd')
mol = qm9[1000]
print(mol)
# fname = sys.argv[1]
# dat = process(fname)
# print(rdkit_process(dat))
