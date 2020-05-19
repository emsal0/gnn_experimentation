import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, MessagePassing, SAGEConv
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import from_networkx

import preprocess_qm9

class SAGEConv2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding

class QM9Dataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(QM9Dataset, self).__init__(root, transform, pre_transform)
        # self.raw_dir = preprocess_qm9.DIR
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return preprocess_qm9.DIR

    @property
    def raw_file_names(self):
        mol_fname = lambda n: preprocess_qm9.PREFIX + '_' + str(n).zfill(6) + '.xyz'
        return [mol_fname(i) for i in range(1,133886)]

    @property
    def processed_file_names(self):
        fn = lambda n: 'graph_files/{}.pt'.format(n)
        return [fn(i) for i in range(1,133886)]

    def process(self):
        i=1
        for raw_path in self.raw_paths:
            nxg = preprocess_qm9.rdkit_process(preprocess_qm9.process(raw_path))
            data = from_networkx(nxg)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(
                self.processed_dir, 'graph_files/{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, '{}.pt'.format(idx)))
        return data

class Net(nn.Module):
    pass

if __name__ == '__main__':
    mol = preprocess_qm9.process('xyzfiles/dsgdb9nsd_004049.xyz')
    print(mol['smiles'])
    mol_graph = preprocess_qm9.rdkit_process(mol)

    print(mol_graph.edges)
    print(from_networkx(mol_graph))
