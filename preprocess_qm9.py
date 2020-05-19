import numpy as np
import networkx as nx
import pandas
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

DIR = 'xyzfiles'
PREFIX = 'dsgdb9nsd'

def process_line(atom_line):
    dfields = atom_line.split('\t')[1:]
    return [d.replace('*^', 'e').replace('.*^', 'e') for d in dfields]


def process(fname):
    with open(fname) as f:
        mdata = f.read().split('\n')

    n_atoms = int(mdata[0])

    props_meta = ['tag+i', 'A', 'B', 'C', 'mu', 'alpha', 'eps_homo',
                  'eps_lumo', 'eps_gap', 'R2', 'zpve', 'U0', 'U',
                  'H', 'G', 'Cv']
    props_raw = mdata[1].split('\t')
    props_data = dict(zip(props_meta, props_raw))

    atom_data = list(map(process_line, mdata[2:2+n_atoms]))

    freqs = list(map(float, mdata[2+n_atoms].split('\t')))

    smiles = mdata[3+n_atoms].strip().split('\t')
    inchi = mdata[4+n_atoms].split('\t')

    return {
        'properties': props_data,
        'atoms': atom_data,
        'frequencies': freqs,
        'smiles': smiles,
        'InChI': inchi
    }


def rdkit_process(dat):
    mol = Chem.MolFromSmiles(dat['smiles'][0])

    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    feats = factory.GetFeaturesForMol(mol)

    g = nx.Graph()

    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        try:
            g.add_node(i, element_type=atom_i.GetSymbol(),
                   atomic_num=atom_i.GetAtomicNum(),
                   is_donor=0,
                   is_acceptor=0,
                   aromatic=atom_i.GetIsAromatic(),
                   num_h=atom_i.GetTotalNumHs(),
                   coord=np.array(dat['atoms'][i]).astype(np.float))
        except:
            import pdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

    for f in range(len(feats)):
        if feats[f].GetFamily() == 'Donor':
            node_list = feats[f].GetAtomIds()
            for i in node_list:
                g.nodes[i]['is_donor'] = 1
        if feats[f].GetFamily() == 'Acceptor':
            node_list = feats[f].GetAtomIds()
            for i in node_list:
                g.nodes[i]['is_acceptor'] = 1

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            edge = mol.GetBondBetweenAtoms(i, j)
            if edge is not None:
                g.add_edge(i, j, bond_type=edge.GetBondType(),
                           distance=np.linalg.norm(
                               g.nodes[i]['coord'] - g.nodes[j]['coord']))

    return g
