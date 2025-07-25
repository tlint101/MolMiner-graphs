"""
Convert molecules into graph
Adapted from MMGX (https://github.com/ohuelab/MMGX.git)
Paper: "Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX"
Communications Chemistry (2024), https://doi.org/10.1038/s42004-024-01084-4
"""
from mminer_graphs.graph.graph import MolGraph
from mminer_graphs.graph.graph_utils import *
from rdkit.Chem import rdReducedGraphs
from collections import defaultdict

__all__ = ["PharmacophoreGraph", "FunctionalGraph"]


# ref: https://github.com/rdkit/rdkit/blob/b208da471f8edc88e07c77ed7d7868649ac75100/Code/GraphMol/ReducedGraphs/ReducedGraphs.cpp
class PharmacophoreGraph(MolGraph):
    def __init__(self, smi, normalize=True, with_atom_index=True):
        super().__init__(smi)
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))
        self.mol = smiles_to_mol(self.smi)
        self.normalize = normalize
        cliques, edges, cliques_prop = self.get_reduced_pharmacophore_graph(self.mol)
        self.cliques = cliques
        self.edges = edges
        self.cliques_prop = cliques_prop
        clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr = self.mol_to_pharm_graph(cliques,
                                                                                                              edges,
                                                                                                              cliques_prop,
                                                                                                              normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = clique_edge_index
        self.edge_features = clique_edge_attr

    @staticmethod
    def get_reduced_pharmacophore_graph(mol):
        mol = mol_with_atom_index(mol)
        mol_g = rdReducedGraphs.GenerateMolExtendedReducedGraph(mol)
        mol_g.UpdatePropertyCache(False)
        mapping_atom = {a.GetAtomMapNum(): x for x, a in enumerate(mol_g.GetAtoms())}

        cliques = []
        cliques_prop = []

        ring_8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) < 8]
        ring_B8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) >= 8]

        # add more 8-atom ring
        if len(ring_B8) > 0:
            rwmol_g = Chem.RWMol(mol_g)
            for rb8 in ring_B8:
                new_a = rwmol_g.AddAtom(Chem.Atom(0))
                rwmol_g.GetAtomWithIdx(new_a).SetProp('_ErGAtomTypes', '')
                for rb8_a in rb8:
                    if rb8_a in mapping_atom:
                        rwmol_g.AddBond(new_a, mapping_atom[rb8_a], Chem.BondType.SINGLE)
            mol_g = rwmol_g

        # display(mol_g)
        num_ring_8 = 0
        num_ring_B8 = 0

        for atom in mol_g.GetAtoms():
            if atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 5 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 4 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*':
                cliques.append(ring_B8[num_ring_B8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_B8 += 1
            else:
                cliques.append([atom.GetAtomMapNum()])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))

        edges = []
        for bond in mol_g.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edges.append((a1, a2))
            edges.append((a2, a1))

        return cliques, edges, cliques_prop

    @staticmethod
    # todo add structural info (mass), polarity, size, aromaticity, then modify and add clique_features for return in
    # todo get_reduced_pharmacophore_graph and pass into mol_to_pharm_graph
    def get_clique_features_pharmacophore(cliques_prop):
        # number of node features (6)
        pharmacophore = np.zeros(6)
        for p in cliques_prop:
            pharmacophore[p] = 1

        return np.array(pharmacophore)

    @staticmethod
    def get_clique_edge_features_pharmacophore(edge, cliques_prop):
        # number of edge features (3)
        begin = cliques_prop[edge[0]]
        end = cliques_prop[edge[1]]

        if len(begin) == 0:
            begin_type = 'none'
        else:
            begin_type = 'phar'

        if len(end) == 0:
            end_type = 'none'
        else:
            end_type = 'phar'

        definedEdgeType = {('none', 'none'): 0,
                           ('none', 'phar'): 1,
                           ('phar', 'none'): 1,
                           ('phar', 'phar'): 2, }
        EdgeType = definedEdgeType[(begin_type, end_type)]

        return np.array(one_of_k_encoding(EdgeType, list(set(definedEdgeType.values()))))

    def mol_to_pharm_graph(self, cliques, edges, cliques_prop, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.get_clique_features_pharmacophore(cliques_prop[idx])
            if normalize and sum(clique_features) != 0:
                clique_attr.append(clique_features / sum(clique_features))
            else:
                clique_attr.append(clique_features)

        clique_edge_index = edges
        clique_edge_attr = []
        for idx in range(len(edges)):
            clique_edge_features = self.get_clique_edge_features_pharmacophore(edges[idx], cliques_prop)
            if normalize and sum(clique_edge_features) != 0:
                clique_edge_attr.append(clique_edge_features / sum(clique_edge_features))
            else:
                clique_edge_attr.append(clique_edge_features)

        return clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr


# FUNCTIONAL-based
# ref: https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt
class FunctionalGraph(MolGraph):
    def __init__(self, smi, normalize=True, with_atom_index=True):
        super().__init__(smi)
        self.smi = mol_to_smiles(smiles_to_mol(smi, with_atom_index=with_atom_index))
        self.mol = smiles_to_mol(self.smi)
        self.normalize = normalize
        cliques, edges, cliques_func, cliques_ring = self.get_reduced_functional_graph(self.mol)
        self.cliques = cliques
        self.edges = edges
        self.cliques_func = cliques_func
        self.cliques_ring = cliques_ring
        clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr = self.mol_to_functional_graph(
            self.mol, cliques,
            edges,
            cliques_func,
            cliques_ring,
            normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = clique_edge_index
        self.edge_features = clique_edge_attr

    @staticmethod
    def get_reduced_functional_graph(mol):
        n_atoms = mol.GetNumAtoms()

        # functional group
        functional_group_dict = dict()
        for x in range(fparams.GetNumFuncGroups()):
            functional_group_dict[x] = list(mol.GetSubstructMatches(fparams.GetFuncGroup(x)))

        # edit #27 <-> #29
        temp = functional_group_dict[27]
        functional_group_dict[27] = functional_group_dict[29]
        functional_group_dict[29] = temp

        cliques = []
        cliques_ring = {}  # node group in ring
        cliques_func = {}  # node group in func
        seen_func = {}  # node seen in func
        group_num = 0

        # extract functional from substructure match
        for f in functional_group_dict:
            for l in functional_group_dict[f]:
                if not (all(ll in seen_func for ll in l)):
                    cliques.append(list(l))
                    for ll in l:
                        if ll in seen_func or ll in cliques_func:
                            cliques_func[ll].append(f)
                            seen_func[ll].append(group_num)
                        else:
                            cliques_func[ll] = [f]
                            seen_func[ll] = [group_num]
                group_num += 1

        # extract bond which not in functional
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if not bond.IsInRing():
                if (a1 not in seen_func) or (a2 not in seen_func):
                    cliques.append([a1, a2])
                elif a1 in seen_func and a2 in seen_func and len(set(seen_func[a1]) & set(seen_func[a2])) == 0:
                    cliques.append([a1, a2])

        # extract ring
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        cliques_ring = ssr

        atom_clique_neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
        for x in range(len(cliques)):
            for atom in cliques[x]:
                atom_clique_neighbors[atom].append(x)

        cliques = [c for c in cliques if len(c) > 0]
        atom_clique_neighbors = [[] for _ in range(n_atoms)]
        for x in range(len(cliques)):
            for atom in cliques[x]:
                atom_clique_neighbors[atom].append(x)

        # build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(atom_clique_neighbors[atom]) <= 1:
                continue
            clique_neighbors = atom_clique_neighbors[atom]
            # number of bond clusters that the atom lies in.
            bonds = [c for c in clique_neighbors if len(cliques[c]) == 2]
            # number of ring clusters that the atom lies in.
            funcring = [c for c in clique_neighbors if len(cliques[c]) > 2]

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

        edges = [i for i in edges]

        return cliques, edges, cliques_func, cliques_ring

    @staticmethod
    # todo add features for group mass, TPSA, pKa, polarity
    def get_functional_clique_features(clique, cliques_func, cliques_ring, mol):
        # number of node features (115)
        func_type = [0 for f in range(len(range(fparams.GetNumFuncGroups())))]  # no unknown
        func_ring_type_list = range(len(RINGS))  # (only aromatic)
        func_ring_type_other_list = range(len(RINGS))  # (other bonds)
        func_ring_type_size_list = [3, 4, 5, 6, 7, 8, 9, 10]  # unknown ring size 3-9 and >9
        func_bond_type_list = range(len(FUNCBOND))  # included unknown
        # atomTypeList = range(len(definedAtom)) # included unknown

        ring_type = one_of_k_encoding_none(None, func_ring_type_list)
        ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
        func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)
        # atomType = atomType = one_of_k_encoding_none(None, atomTypeList)

        # functional group
        func_found = False
        if all(c in cliques_func for c in clique):
            funcGroup = [cliques_func[c] for c in clique]
            intersect = funcGroup[0]
            for f in funcGroup:
                intersect = set(set(intersect) & set(f))
            for x in list(intersect):
                func_type[x] = 1
                func_found = True
            if func_found:  # func, not ring not bond
                ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)

        # ring type
        if len(clique) > 2 and not func_found:
            if clique in cliques_ring:
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # not kekulize
                smarts = Chem.MolFragmentToSmarts(new_mol, clique)
                ring_found = False
                for ring in RINGS:
                    mol_ring = Chem.MolFromSmarts(ring)
                    mol_smart = Chem.MolFromSmarts(smarts)
                    t1 = topology_checker(mol_ring)
                    t2 = topology_checker(mol_smart)
                    if len(mol_smart.GetSubstructMatches(mol_ring)) != 0:
                        ring_type = one_of_k_encoding_unk(RINGS.index(ring), func_ring_type_list)
                        ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                        func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)
                        ring_found = True
                        break
                    elif is_isomorphic(t1, t2):
                        ring_type = one_of_k_encoding_unk(RINGS.index(ring), func_ring_type_list)
                        ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                        func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)
                        ring_found = True
                        break
                    elif is_isomorphic_atom(t1, t2):
                        ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                        ring_type_other = one_of_k_encoding_unk(RINGS.index(ring), func_ring_type_other_list)
                        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                        func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)
                        ring_found = True
                        break
                if not ring_found:  # unknown ring
                    mol_smart = Chem.MolFromSmarts(smarts)
                    ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                    ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                    ring_type_size = one_of_k_encoding_unk(mol_smart.GetNumAtoms(), func_ring_type_size_list)
                    func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)
            else:  # not ring not bond
                ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                func_bond_type = one_of_k_encoding_none(None, func_bond_type_list)

        # bond type
        if len(clique) == 2 and not func_found:
            bond_found = False
            for bond in mol.GetBonds():
                b = bond.GetBondType()
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                a1 = atom1.GetIdx()
                a2 = atom2.GetIdx()
                a1_s = atom1.GetSymbol()
                a2_s = atom2.GetSymbol()
                if [a1, a2] == clique or [a2, a1] == clique:
                    if (a1_s, b, a2_s) in FUNCBOND:
                        func_bond_type = one_of_k_encoding_unk(FUNCBOND.index((a1_s, b, a2_s)), func_bond_type_list)
                        ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                        ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                        bond_found = True
                        break
                    elif (a2_s, b, a1_s) in FUNCBOND:
                        func_bond_type = one_of_k_encoding_unk(FUNCBOND.index((a2_s, b, a1_s)), func_bond_type_list)
                        ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                        ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                        ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                        bond_found = True
                        break
            if not bond_found:  # unknown bond
                ring_type = one_of_k_encoding_none(None, func_ring_type_list)
                ring_type_other = one_of_k_encoding_none(None, func_ring_type_other_list)
                ring_type_size = one_of_k_encoding_none(None, func_ring_type_size_list)
                func_bond_type = one_of_k_encoding_unk(None, func_bond_type_list)

        return np.array(func_type + ring_type + ring_type_other + ring_type_size + func_bond_type)

    @staticmethod
    def get_functional_clique_edge_features(edge, clique, cliques_func):
        # number of edge features (10)+10 = (20) # (10)+12 = (22)
        begin = clique[edge[0]]
        end = clique[edge[1]]

        if all(c in cliques_func for c in begin):
            begin_type = 'func'
        elif len(begin) > 2:
            begin_type = 'ring'
        elif len(begin) == 1:
            begin_type = 'atom'
        else:
            begin_type = 'bond'

        if all(c in cliques_func for c in end):
            end_type = 'func'
        elif len(end) > 2:
            end_type = 'ring'
        elif len(end) == 1:
            end_type = 'atom'
        else:
            end_type = 'bond'

        intersect = len(set(begin) & set(end))

        begin_atom = 0
        end_atom = 0
        if intersect == 1:
            a1 = list(set(begin) & set(end))[0]
            a2 = list(set(begin) & set(end))[0]
            begin_atom = sorted(begin).index(a1) + 1
            end_atom = sorted(end).index(a2) + 1
        # in case, more than 2 common atoms
        else:
            begin_atom = 0
            end_atom = 0

        EDGE_TYPE = {('atom', 'atom'): 0,
                     ('atom', 'bond'): 1,
                     ('atom', 'ring'): 2,
                     ('atom', 'func'): 3,
                     ('bond', 'atom'): 1,
                     ('bond', 'bond'): 4,
                     ('bond', 'ring'): 5,
                     ('bond', 'func'): 6,
                     ('ring', 'atom'): 2,
                     ('ring', 'bond'): 5,
                     ('ring', 'ring'): 7,
                     ('ring', 'func'): 8,
                     ('func', 'atom'): 3,
                     ('func', 'bond'): 6,
                     ('func', 'ring'): 8,
                     ('func', 'func'): 9}
        edge_type = EDGE_TYPE[(begin_type, end_type)]

        # return np.array(one_of_k_encoding(edgeType,list(set(definedEdgeType.values())))+
        #                 one_of_k_encoding_unk(intersect,list(range(10)))+[begin_atom]+[end_atom])
        return np.array(one_of_k_encoding(edge_type, list(set(EDGE_TYPE.values()))) +
                        one_of_k_encoding_unk(intersect, list(range(10))))

    def mol_to_functional_graph(self, mol, cliques, edges, cliques_func, cliques_ring, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.get_functional_clique_features(cliques[idx], cliques_func, cliques_ring, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(clique_features / sum(clique_features))
            else:
                clique_attr.append(clique_features)

        clique_edge_index = edges
        clique_edge_attr = []
        for idx in range(len(edges)):
            clique_edge_features = self.get_functional_clique_edge_features(edges[idx], cliques, cliques_func)
            if normalize and sum(clique_edge_features) != 0:
                clique_edge_attr.append(clique_edge_features / sum(clique_edge_features))
            else:
                clique_edge_attr.append(clique_edge_features)

        return clique_size, clique_index, clique_attr, clique_edge_index, clique_edge_attr


if __name__ == "__main__":
    import doctest

    doctest.testmod()
