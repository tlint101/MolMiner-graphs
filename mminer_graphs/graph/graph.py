"""
Convert molecules into graph
Adapted from MMGX (https://github.com/ohuelab/MMGX.git)
Paper: "Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX"
Communications Chemistry (2024), https://doi.org/10.1038/s42004-024-01084-4
"""

from mminer_graphs.graph.graph_utils import *
from rdkit.Chem import rdchem
from rdkit import Chem

__all__ = ["MolGraph", "AtomGraph", "GLOBAL_VALUES"]


# print global definition
def GLOBAL_VALUES():
    # atom features
    # print('ATOM Features:', ATOM_NUM) #todo add additional charges into list
    print('ATOM Features: 142')
    for (x, item) in enumerate(ATOMS):
        print(f"  {x}: {item}")
    # degree
    print(f"  Atom Degree: 11")
    print(f"  Atom Valence: 7")
    print(f"  Formal charge: 5")
    print(f"  Hybridization: 6")
    print(f"  Aromaticity: 1")
    print(f"  Number of Hydrogens: 5")
    print(f"  Chirality: 5")
    print(f"  Ring status: 1")
    print(f"  Mass: 1")

    # bond features
    print('BOND Features:', BOND_NUM)
    print('  Bond types:')
    for (x, item) in enumerate(BONDS):
        print(f'    {x}: {item}')

    print('  Bond flags:')
    for (x, item) in enumerate(BOND_FEATURES):
        print(f'    {x + len(BONDS)}: {item}')

    print('  Bond stereo:')
    for (x, item) in enumerate(BOND_STEREO):
        print(f'    {x + len(BONDS) + len(BOND_FEATURES)}: {item}')

    print(f'    {len(BONDS) + len(BOND_FEATURES) + len(BOND_STEREO)}: stereo_atoms_present')

    # features for functional graph
    print('FUNCTIONAL Graph Features:', FUNCTIONAL_NUM)
    for x in range(fparams.GetNumFuncGroups()):
        if x == 27:
            print('SWAP WITH 29')
        if x == 29:
            print('SWAP WITH 27')
        print(f"{x}: {fparams.GetFuncGroup(x).GetProp('_Name')} {Chem.MolToSmarts(fparams.GetFuncGroup(x))}")
    # functional graph bond features
    print('FUNC_BOND Graph Bond Features', FUNCBOND_NUM)
    for (x, item) in enumerate(FUNCBOND):
        print(f"{x}: {item}")

    print('FUNC_RING Functional Graph Rings', FUNCRING_NUM)
    print(func_ring)


class MolGraph:
    def __init__(self, smi=None, with_atom_index=True):
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))  # standardize smi
        self.mol = smiles_to_mol(self.smi)
        self.cliques = None
        self.edges = None
        self.graph_size = 0
        self.node_index = []
        self.node_features = []
        self.edge_index = []
        self.edge_features = []

    def getMoleculeGraph(self):
        return self.mol, self.smi, self.cliques, self.edges, self.graph_size, \
            self.node_index, self.node_features, self.edge_index, self.edge_features,

    def __str__(self):
        if not isinstance(self, AtomGraph):
            return "Molecule Graph: " + self.smi + "\n" + \
                "Node Index:" + str(self.node_index) + "\n" + \
                "Edge Index:" + str(self.edge_index) + "\n" + \
                "Cliques:" + str(self.cliques)
        else:
            return "Molecule Graph: " + self.smi + "\n" + \
                "Node Index:" + str(self.node_index) + "\n" + \
                "Edge Index:" + str(self.edge_index)


# ATOM-based
class AtomGraph(MolGraph):
    def __init__(self, smi, normalize=True, with_atom_index=True):
        super().__init__(smi)
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))
        self.mol = smiles_to_mol(self.smi)
        self.normalize = normalize
        graph_size, node_index, node_attr, edge_index, edge_attr = self.mol_to_atom_graph(self.mol, normalize)
        self.graph_size = graph_size
        self.node_index = node_index
        self.node_features = node_attr
        self.edge_index = edge_index
        self.edge_features = edge_attr
        # atom graph only
        self.cliques = [[x] for x in node_index]
        self.edges = edge_index

    def mol_to_atom_graph(self, mol=None, normalize=True):
        """
        Convert molecules into graph embeddings. Will run default graph featurization in this order - graph_size,
        node_index, node_attr, edge_index, edge_attr.
        :param mol: Chem.rdchem.Mol
            A molecule rendered in RDKit for graph featurization.
        :param normalize: bool
            Convert vector into a distibution where values add up to 1.
        :return:
        """
        if mol is None:
            mol = self.mol

        graph_size = mol.GetNumAtoms()

        node_attr = []
        node_index = []
        for atom in mol.GetAtoms():
            # print(atom.GetAtomicNum())
            node_feature = self._get_atom_feats(atom, chirality=True)
            node_attr.append(list(node_feature))
            node_index.append(atom.GetIdx())

        edge_attr = []
        edge_index = []
        # bidirectional edges
        for bond in mol.GetBonds():
            edge_feature = self._get_bond_feats(bond)
            # from A -> B
            edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

            # from B -> A
            edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

        return graph_size, node_index, node_attr, edge_index, edge_attr

    @staticmethod
    def _get_atom_feats(atom, chirality: bool = True):
        """
        Get atom features (Total: 142). The features are ordered as follows:
        - atomic number (100)
        - degree (11)
        - implicit valence (7)
        - formal charge (5)
        - hybridization (6)
        - aromaticity (1)
        - number of hydrogens (5)
        - chirality (5)
        - ring status (1)
        - mass (1)
        """
        atom_feats = []

        # atomic number
        atom_feats += one_hot_encode(atom.GetAtomicNum(), ATOMIC_NUM)

        # atom degree (number of directly bonded atoms)
        atom_feats += one_hot_encode(atom.GetDegree(), list(range(11)))

        # implicit valence
        atom_feats += one_hot_encode(atom.GetValence(Chem.ValenceType.IMPLICIT), list(range(7)))

        # formal charge
        formal_charges = [-2, -1, 0, 1, 2]
        atom_feats += one_hot_encode(atom.GetFormalCharge(), formal_charges)

        # hybridization
        hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'Unknown'
        ]
        atom_feats += one_hot_encode(atom.GetHybridization(), hybridizations)

        # aromatic
        atom_feats += [int(atom.GetIsAromatic())]

        # number of Hs
        atom_feats += one_hot_encode(atom.GetTotalNumHs(), list(range(5)))

        # chirality
        if chirality:
            try:
                chiral_tags = [
                    rdchem.ChiralType.CHI_UNSPECIFIED,
                    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    rdchem.ChiralType.CHI_OTHER
                ]
                atom_feats += one_hot_encode(atom.GetChiralTag(), chiral_tags)
            except:
                atom_feats += [0, 0]
            atom_feats += [int(atom.HasProp('_ChiralityPossible'))]

        # check if atom is in ring
        atom_feats += [np.multiply(atom.IsInRing(), 1)]

        # atom mass
        atom_feats += [atom.GetMass() / 100]  # normalize atomic mass between 0 and 1

        return np.array(atom_feats, dtype=np.float32)

    @staticmethod
    def _get_bond_feats(bond, chirality: bool = True):
        """
        Get bond features (Total: 11). The features are ordered as follows:
        - numm bond (1)
        - bond type (4)
        - conjugated (1)
        - aromatic (1)
        - ring (1)
        - stereo info (7)
        - stereo atom present (1)
        """
        # bond features
        bond_feats = []

        # null bond
        bond_feats += [int(bond is None)]
        if bond is None:
            return np.zeros(16, dtype=np.float32)

        # bond info
        bond_type = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        bond_feats += one_hot_encode(bond.GetBondType(), bond_type)

        # bond conjugation and ring status
        bond_feats += [int(bond.GetIsConjugated())]
        bond_feats += [int(bond.GetIsAromatic())]
        bond_feats += [int(bond.IsInRing())]

        # stereo info
        stereo_feats = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            'Unknown'
        ]
        bond_feats += one_hot_encode(bond.GetStereo(), stereo_feats)

        if chirality:
            try:
                stereo_atoms = bond.GetStereoAtoms()
                bond_feats.append(len(stereo_atoms) > 0)
            except Exception:
                bond_feats.append(False)

        return np.array(bond_feats, dtype=np.float32)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
