"""
Support utilities for molecular graphs.
Adapted from MMGX (https://github.com/ohuelab/MMGX.git)
Paper: "Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX"
Communications Chemistry (2024), https://doi.org/10.1038/s42004-024-01084-4
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem, RDConfig
from rdkit.Chem import FragmentCatalog

# list of atoms in molecules (up to 100)
periodic_table = Chem.GetPeriodicTable()
ATOMS = [periodic_table.GetElementSymbol(i) for i in range(1, 101)]
# convert atom list into atomic numbers
ATOMIC_NUM = [periodic_table.GetAtomicNumber(x) if x != 'Unknown' else 0 for x in ATOMS]
ATOM_NUM = len(ATOMS)

# bond constraints
BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
         Chem.rdchem.BondType.AROMATIC]

# bond features
BOND_FEATURES = [
    'is_conjugated',
    'is_aromatic',
    'is_in_ring',
]

# bond stereo type
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

# totoal length of bond features
BOND_NUM = (
        1 +  # null bond
        len(BONDS) +
        len(BOND_FEATURES) +
        len(BOND_STEREO) +
        1  # stereo_atoms_present (bool)
)

# for functional graph
fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
FUNCTIONAL_NUM = len(range(fparams.GetNumFuncGroups()))

FUNCBOND = [
    ('C', Chem.rdchem.BondType.SINGLE, 'C'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'C'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'C'),
    ('C', Chem.rdchem.BondType.SINGLE, 'O'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'O'),
    ('C', Chem.rdchem.BondType.SINGLE, 'N'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'N'),
    ('C', Chem.rdchem.BondType.SINGLE, 'S'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('O', Chem.rdchem.BondType.SINGLE, 'O'),
    ('O', Chem.rdchem.BondType.SINGLE, 'N'),
    ('O', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('O', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.SINGLE, 'N'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('N', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.SINGLE, 'S'),
    ('S', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.TRIPLE, 'S'),
    'Unknown'
]
FUNCBOND_NUM = len(FUNCBOND)

# pull in util file
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
file_path = os.path.join(module_dir, 'utils', 'func_ring.csv')
func_ring = pd.read_csv(file_path)

RINGS = list(func_ring['smarts'])
FUNCRING_NUM = len(RINGS)


# smiles to mol
def mol_to_smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles


# mol to smiles
def smiles_to_mol(smi, with_atom_index=True, kekulize=False):
    mol = Chem.MolFromSmiles(smi)
    if with_atom_index:
        mol = mol_with_atom_index(mol)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


# mol with atom index
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def one_hot_encode(val, choices):
    encoding = [int(val == choice) for choice in choices]
    if sum(encoding) == 0:
        encoding[-1] = 1  # 'unknown'
    return encoding


# one-hot encoding (unknown all zeros)
def one_of_k_encoding_none(x, allowable_set):
    # Maps inputs not in the allowable set to zero list
    if x not in allowable_set:
        x = [0 for i in range(len(allowable_set))]
    return list(map(lambda s: x == s, allowable_set))


# one-hot encoding (only allowable set)
def one_of_k_encoding(x, allowable_set):
    # Maps inputs only in the allowable set
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one-hot encoding (unknown to last element)
def one_of_k_encoding_unk(x, allowable_set):
    # Maps inputs not in the allowable set to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# convert mol to topology (atom symbol, no consider bond type)
def topology_checker(mol):
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Add the bonds as edges
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=bond.GetBondType())
        # # Add the bonds as edges
        # for bonded in atom.GetNeighbors():
        #     topology.add_edge(atom.GetIdx(), bonded.GetIdx())
    return topology


# check same atom (atom from nx)
def is_same_atom(a1, a2):
    return a1['symbol'] == a2['symbol']


# check same bond (bond from nx)
def is_same_bond(b1, b2):
    return b1['type'] == b2['type']


# check graph is isomorphic
def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom, edge_match=is_same_bond)


# check graph is isomorphic only atom
def is_isomorphic_atom(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom)


def get_bond_features(bond, use_chirality=True):
    """Get bond features (11)"""

    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    # stereo info
    stereo = bond.GetStereo()
    stereo_feats = [
        stereo == Chem.rdchem.BondStereo.STEREONONE,
        stereo == Chem.rdchem.BondStereo.STEREOANY,
        stereo == Chem.rdchem.BondStereo.STEREOZ,
        stereo == Chem.rdchem.BondStereo.STEREOE,
    ]
    bond_feats.extend(stereo_feats)

    if use_chirality:
        try:
            stereo_atoms = bond.GetStereoAtoms()
            bond_feats.append(len(stereo_atoms) > 0)
        except Exception:
            bond_feats.append(False)

    return np.array(bond_feats, dtype=int)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
