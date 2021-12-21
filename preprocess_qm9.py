import numpy as np
import networkx as nx
import rdkit.Chem as Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

DATADIR = 'xyzfiles'
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

    g = nx.Graph(props=dat['properties'])
    
    hydrogen_count = 0


    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        num_implicit_h = atom_i.GetNumImplicitHs()

        try:
            g.add_node(i, element_type=atom_i.GetSymbol(),
                   atomic_num=atom_i.GetAtomicNum(),
                   is_donor=0,
                   is_acceptor=0,
                   aromatic=atom_i.GetIsAromatic(),
                   num_h=atom_i.GetTotalNumHs(),
                   coord=np.array(dat['atoms'][i]).astype(np.float64))

            for i in range(num_implicit_h):
                h_idx = mol.GetNumAtoms() + hydrogen_count

                g.add_node(h_idx,
                        element_type = 'H',
                        atomic_num = 1,
                        is_donor = 0,
                        is_acceptor = 0,
                        aromatic = False,
                        num_h = 0,
                        coord = np.array(dat['atoms'][h_idx]).astype(np.float64))

                hydrogen_count += 1
                pass
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

    h_cursor = mol.GetNumAtoms()
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        for _ in range(atom_i.GetNumImplicitHs()):
            g.add_edge(i, h_cursor, bond_type = 1.,
                    distance = np.linalg.norm(g.nodes[i]['coord'] - g.nodes[h_cursor]['coord']))
            h_cursor += 1


    return g
