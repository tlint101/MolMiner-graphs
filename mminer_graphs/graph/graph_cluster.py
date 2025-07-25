"""
Convert molecules into graph
Adapted from MMGX (https://github.com/ohuelab/MMGX.git)
Paper: "Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX"
Communications Chemistry (2024), https://doi.org/10.1038/s42004-024-01084-4
"""
from mminer_graphs.graph.graph import MolGraph
from mminer_graphs.graph.graph_utils import *
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

__all__ = ["JunctionTreeGraph", "ClusterGraph"]

# for junction tree and cluster, Maximum Spanning Tree
MST_MAX_WEIGHT = 100


# JUNCTIONTREE-based
# ref: https://arxiv.org/pdf/1802.04364.pdf
class JunctionTreeGraph(MolGraph):
    def __init__(self, smi, normalize=True, with_atom_index=True):
        super().__init__(smi)
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))
        self.mol = smiles_to_mol(self.smi)
        self.normalize = normalize
        cliques, edges = self.get_reduced_junction_graph(self.mol)
        self.cliques = cliques
        self.edges = edges
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_junction_graph(self.mol,
                                                                                                               cliques,
                                                                                                               edges,
                                                                                                               normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    # form a junction tree graph with intersection as a node
    #     mol              graph
    #      A                AC
    #      |      >>>       |
    #  B - C - D       BC - C - DC
    @staticmethod
    def get_reduced_junction_graph(mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        atom_clique_neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
        for x in range(len(cliques)):
            for atom in cliques[x]:
                atom_clique_neighbors[atom].append(x)

        # merge rings that share more than 2 atoms as they form bridged compounds.
        for x in range(len(cliques)):
            if len(cliques[x]) <= 2: continue
            for atom in cliques[x]:
                for j in atom_clique_neighbors[atom]:
                    if x >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[x]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[x].extend(cliques[j])
                        cliques[x] = list(set(cliques[x]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]

        # build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(atom_clique_neighbors[atom]) <= 1:
                continue
            clique_neighbors = atom_clique_neighbors[atom]
            # number of bond clusters that the atom lies in.
            bonds = [c for c in clique_neighbors if len(cliques[c]) == 2]
            # number of ring clusters that the atom lies in.
            rings = [c for c in clique_neighbors if len(cliques[c]) > 4]
            # in general, if len(cnei) >= 3, a singleton should be added,
            # but 1 bond + 2 ring is currently not dealt with.
            if len(bonds) > 2 or (len(bonds) == 2 and len(clique_neighbors) > 2):
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in clique_neighbors:
                    edges[(c1, c2)] = 1
            # at least 1 bond connect to at least 2 rings
            elif len(rings) >= 2 and len(bonds) >= 1:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in clique_neighbors:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            # multiple (n>2) complex rings
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in clique_neighbors:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            else:
                for x in range(len(clique_neighbors)):
                    for j in range(x + 1, len(clique_neighbors)):
                        c1, c2 = clique_neighbors[x], clique_neighbors[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            # clique_neighbors[i] < clique_neighbors[j] by construction ?
                            edges[(c1, c2)] = len(inter)
                            edges[(c2, c1)] = len(inter)

        # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in atom_clique_neighbors[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                atom_clique_neighbors[atom.GetIdx()].append(len(cliques) - 1)

        edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = []
        for x in range(len(row)):
            edges.append((row[x], col[x]))
            edges.append((col[x], row[x]))
        return cliques, edges

    def get_junction_clique_features(self, clique, edges, clique_idx, mol):
        # number of node features (83)
        NUM_ATOM_DICT = {a: 0 for a in ATOMS}
        NUM_BOND_DICT = {b: 0 for b in BONDS}

        # number of atoms
        num_atoms = len(clique)
        # number of edges
        num_edges = 0
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                num_edges += 1
        # number of Hs
        num_hs = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                num_hs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NUM_ATOM_DICT:
                    NUM_ATOM_DICT[sb] += 1
                else:
                    NUM_ATOM_DICT['Unknown'] += 1
        # is ring
        is_ring = 0
        if len(clique) > 2:
            is_ring = 1
        # is bond
        is_bond = 0
        if len(clique) == 2:
            is_bond = 1
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bt = bond.GetBondType()
                NUM_BOND_DICT[bt] += 1

                # convert number of each atom
        if sum(list(NUM_ATOM_DICT.values())) != 0 and self.normalize:
            num_each_atom = [float(x) / sum(list(NUM_ATOM_DICT.values())) for x in list(NUM_ATOM_DICT.values())]
        else:
            num_each_atom = [int(x) for x in list(NUM_ATOM_DICT.values())]
            # convert number of each bond
        if sum(list(NUM_BOND_DICT.values())) != 0 and self.normalize:
            num_each_bond = [float(x) / sum(list(NUM_BOND_DICT.values())) for x in list(NUM_BOND_DICT.values())]
        else:
            num_each_bond = [int(x) for x in list(NUM_BOND_DICT.values())]

        return np.array(
            one_of_k_encoding_unk(num_atoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(num_edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(num_hs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            [is_ring] +
            [is_bond] +
            num_each_atom +
            num_each_bond
        )

    @staticmethod
    # todo add features for size, aromaticity, nad ring classification
    def get_junction_clique_edge_features(edge, clique):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])

        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else:
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else:
            end_type = 'bond'

        EDGE_TYPE = {('atom', 'atom'): 0,
                     ('atom', 'bond'): 1,
                     ('atom', 'ring'): 2,
                     ('bond', 'atom'): 1,
                     ('bond', 'bond'): 3,
                     ('bond', 'ring'): 4,
                     ('ring', 'atom'): 2,
                     ('ring', 'bond'): 4,
                     ('ring', 'ring'): 5}
        edge_type = EDGE_TYPE[(begin_type, end_type)]

        return np.array(one_of_k_encoding(edge_type, list(set(EDGE_TYPE.values()))))

    def mol_to_junction_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.get_junction_clique_features(cliques[idx], edges, idx, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(clique_features / sum(clique_features))
            else:
                clique_attr.append(clique_features)

        clique_edge_index = edges
        clique_edge_attr = []
        for idx in range(len(edges)):
            clique_edge_features = self.get_junction_clique_edge_features(edges[idx], cliques)
            if normalize and sum(clique_edge_features) != 0:
                clique_edge_attr.append(clique_edge_features / sum(clique_edge_features))
            else:
                clique_edge_attr.append(clique_edge_features)

        return clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr


# CLUSTER-based
# each cluster (cliques) of atoms will encode rings or functional groups as nodes.
class ClusterGraph(MolGraph):
    def __init__(self, smi, normalize=True, with_atom_index=True):
        super().__init__(smi)
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))
        self.mol = smiles_to_mol(self.smi)
        self.normalize = normalize
        cliques, edges = self.get_reduced_cluster_graph(self.mol)
        self.cliques = cliques
        self.edges = edges
        clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr = self.mol_to_cluster_graph(
            self.mol,
            cliques,
            edges,
            normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = clique_edge_index
        self.edge_features = clique_edge_attr

    # form a cluster graph with intersection as a cycle
    #     mol             graph
    #      A               AC
    #      |      >>>    /    \
    #  B - C - D       BC ---- DC
    @staticmethod
    def get_reduced_cluster_graph(mol):
        n_atoms = mol.GetNumAtoms()

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        cliques = [c for c in cliques if len(c) > 0]
        atom_clique_neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
        for x in range(len(cliques)):
            for atom in cliques[x]:
                atom_clique_neighbors[atom].append(x)

        # build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(atom_clique_neighbors[atom]) <= 1:
                continue
            clique_neighbors = atom_clique_neighbors[atom]

            for x in range(len(clique_neighbors)):
                for j in range(x + 1, len(clique_neighbors)):
                    c1, c2 = clique_neighbors[x], clique_neighbors[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)
                        edges[(c2, c1)] = len(inter)

        # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in atom_clique_neighbors[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                atom_clique_neighbors[atom.GetIdx()].append(len(cliques) - 1)

        edges = [e for e in edges]
        if len(edges) == 0:
            return cliques, edges

        return cliques, edges

    # todo add MolWt, TPSA, mass, and aromaticity
    def get_cluster_clique_features(self, clique, edges, clique_idx, mol):
        # number of node (atom) features (44 total)
        NUM_ATOM_DICT = {x: 0 for x in ATOMS}

        # number of atoms
        num_atoms = len(clique)
        # number of edges
        num_edges = 0
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                num_edges += 1
        # number of Hs
        num_hs = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                num_hs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NUM_ATOM_DICT:
                    NUM_ATOM_DICT[sb] += 1
                else:
                    NUM_ATOM_DICT['Unknown'] += 1
        # is ring
        is_ring = 0
        if len(clique) > 2:
            is_ring = 1
        # is bond
        is_bond = 0
        if len(clique) == 2:
            is_bond = 1

        num_each_bond_dict = defaultdict(int)
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bond_type = bond.GetBondType()
                num_each_bond_dict[bond_type] += 1

                # convert number of each atom
        if sum(list(NUM_ATOM_DICT.values())) != 0 and self.normalize:
            num_each_atom = [float(x) / sum(list(NUM_ATOM_DICT.values())) for x in list(NUM_ATOM_DICT.values())]
        else:
            # convert number of each bond
            num_each_atom = [int(x) for x in list(NUM_ATOM_DICT.values())]
        if sum(list(num_each_bond_dict.values())) != 0 and self.normalize:
            num_each_bond = [float(x) / sum(list(num_each_bond_dict.values())) for x in
                             list(num_each_bond_dict.values())]
        else:
            num_each_bond = [int(x) for x in list(num_each_bond_dict.values())]

        return np.array(
            one_of_k_encoding_unk(num_atoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(num_edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(num_hs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            [is_ring] +
            [is_bond] +
            num_each_atom +
            num_each_bond
        )

    @staticmethod
    def get_cluster_clique_edge_features(edge, clique):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])

        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else:
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else:
            end_type = 'bond'

        EDGE_TYPE = {('atom', 'atom'): 0,
                     ('atom', 'bond'): 1,
                     ('atom', 'ring'): 2,
                     ('bond', 'atom'): 1,
                     ('bond', 'bond'): 3,
                     ('bond', 'ring'): 4,
                     ('ring', 'atom'): 2,
                     ('ring', 'bond'): 4,
                     ('ring', 'ring'): 5}
        edge_type = EDGE_TYPE[(begin_type, end_type)]

        return np.array(one_of_k_encoding(edge_type, list(set(EDGE_TYPE.values()))))

    def mol_to_cluster_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.get_cluster_clique_features(cliques[idx], edges, idx, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(clique_features / sum(clique_features))
            else:
                clique_attr.append(clique_features)

        clique_edge_index = edges
        clique_edge_attr = []
        for idx in range(len(edges)):
            clique_edge_features = self.get_cluster_clique_edge_features(edges[idx], cliques)
            if normalize and sum(clique_edge_features) != 0:
                clique_edge_attr.append(clique_edge_features / sum(clique_edge_features))
            else:
                clique_edge_attr.append(clique_edge_features)

        return clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr


if __name__ == "__main__":
    import doctest

    doctest.testmod()
