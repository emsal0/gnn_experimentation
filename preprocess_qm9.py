DIR = 'xyzfiles'
PREFIX = 'dsgdb9nsd'


def process_line(atom_line):
    dfields = atom_line.split('\t')
    return dict(zip(['element', 'x', 'y', 'z'], dfields))

def process(fname):
    with open(fname) as f:
        mdata = f.read().split('\n')

    n_atoms = int(mdata[0])

    props_meta = ['tag', 'i', 'A', 'B', 'C', 'mu', 'alpha', 'eps_homo',
                   'eps_lumo', 'eps_gap', 'R2', 'zpve', 'U0', 'U',
                   'H', 'G', 'Cv']
    props_raw = mdata[1].split('\t')
    props_data = dict(zip(props_meta, props_raw))

    atom_data = map(process_line, mdata[2:2+n_atoms])

    freqs = map(float, mdata[3+n_atoms].split('\t'))

    smiles = mdata[4+n_atoms].split('\t')
    inchi = mdata[5+n_atoms].split('\t')

    return {
            'properties': props_data,
            'atoms': atom_data,
            'frequencies': freqs,
            'smiles': smiels,
            'InChI': inchi
            }

