import sys
import torch
import networkx as nx
import os.path

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
from torch_geometric.nn.conv import GINEConv

from preprocess_qm9 import DATADIR, PREFIX
from preprocess_qm9 import * 

class QM9Dataset(Dataset):
    def __init__(self, datadir=DATADIR, prefix=PREFIX):
        self.datadir = datadir
        self.prefix = prefix

    def __len__(self):
        return len(os.listdir(self.datadir))

    def __getitem__(self, idx):
        if type(idx) is slice:
            res = [self.__getitem__(i) 
                    for i in range(*list(filter(None, [idx.start, idx.stop, idx.step])))]
            return res
        fname = os.path.join(self.datadir, self.prefix + '_' + f"{idx:06}" + '.xyz')
        dat = process(fname)
        nx_graph = rdkit_process(dat)
        return from_networkx(nx_graph)
    
qm9 = QM9Dataset('xyzfiles', 'dsgdb9nsd')
nn = Seq(
        Lin(4, 32), ReLU(), Lin(32, 1))

conv = GINEConv(nn, train_eps=True, edge_dim=1)
for i in range(1,11):
    mol = qm9[i]
    print(mol.bond_type)
    processed = conv(mol.coord.float(), mol.edge_index, mol.bond_type.float().view(-1,1))

    print(processed)
