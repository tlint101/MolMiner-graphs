import numpy as np
import pytest
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from mminer_graphs.graph import (
    AtomGraph,
    ClusterGraph,
    FunctionalGraph,
    JunctionTreeGraph,
    PharmacophoreGraph,
)
from mminer_graphs.graph.graph_utils import BOND_NUM, BOND_STEREO
from mminer_graphs.model.model_utils import GRAPH_EDGE_DIMS, GRAPH_NODE_DIMS, mol_to_graph


def test_bond_num_matches_atom_graph(bace_df):
    assert BOND_NUM == 16
    assert len(BOND_STEREO) == 7
    g = AtomGraph(bace_df["smiles"].iloc[0])
    assert len(g.edge_features[0]) == BOND_NUM == 16


@pytest.mark.parametrize(
    "graph_name,graph_cls",
    [
        ("AtomGraph", AtomGraph),
        ("PharmacophoreGraph", PharmacophoreGraph),
        ("FunctionalGraph", FunctionalGraph),
        ("JunctionTreeGraph", JunctionTreeGraph),
        ("ClusterGraph", ClusterGraph),
    ],
)
def test_all_five_graph_types_on_bace(bace_df, graph_name, graph_cls):
    expected_node_dim = GRAPH_NODE_DIMS[graph_name]
    expected_edge_dim = GRAPH_EDGE_DIMS[graph_name]

    for smi in bace_df["smiles"].iloc[:10]:
        g = graph_cls(smi)
        nf = np.array(g.node_features)
        assert nf.ndim == 2 and nf.shape[1] == expected_node_dim
        if len(g.edge_features) > 0:
            ef = np.array(g.edge_features)
            assert ef.ndim == 2 and ef.shape[1] == expected_edge_dim


@pytest.mark.parametrize(
    "smi",
    [
        "C1=CCCCC1C",                            # unsaturated non-aromatic ring
        "CC(=O)O.[Na+]",                         # salt with isolated ion
        "C1C2CC3CC1CC(C2)C3",                    # bridged ring (adamantane)
        "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",    # bridged drug (atropine)
        "[Md]C",                                 # atomic number > 100
        "*CC",                                   # dummy atom
        "C[NH2]->[Pt]",                          # dative bond
        "C",                                     # single atom
    ],
)
def test_cluster_and_junction_tree_edge_cases(smi):
    for gtype, cls in [("JunctionTreeGraph", JunctionTreeGraph), ("ClusterGraph", ClusterGraph)]:
        g = cls(smi)
        arr = np.array(g.node_features)
        assert arr.ndim == 2 and arr.shape[1] == 139
        data = mol_to_graph(smi, label=1, graph_type=gtype)
        assert data.x.shape[1] == 139


def test_junction_tree_vs_cluster_topology():
    # Isobutane: JunctionTreeGraph adds a singleton center (4 nodes); ClusterGraph keeps a 3-node cycle
    jt = JunctionTreeGraph("CC(C)C")
    cg = ClusterGraph("CC(C)C")
    assert jt.graph_size == 4 and len(jt.edges) == 6
    assert cg.graph_size == 3 and len(cg.edges) == 6


def test_atom_graph_chirality_fallback_length(bace_df):
    mol = Chem.MolFromSmiles(bace_df["smiles"].iloc[0])
    atom = mol.GetAtomWithIdx(0)

    class BrokenChiralAtom:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def GetChiralTag(self):
            raise RuntimeError("simulated chirality failure")

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    feats_normal = AtomGraph._get_atom_feats(atom, chirality=True)
    feats_fallback = AtomGraph._get_atom_feats(BrokenChiralAtom(atom), chirality=True)
    assert feats_normal.shape == (142,)
    assert feats_fallback.shape == (142,)


@pytest.mark.parametrize(
    "graph_type,zero_edge_smi",
    [
        ("AtomGraph", "C"),
        ("PharmacophoreGraph", "O"),
        ("FunctionalGraph", "c1ccccc1"),
        ("JunctionTreeGraph", "c1ccccc1"),
        ("ClusterGraph", "c1ccccc1"),
    ],
)
def test_mol_to_graph_zero_edge_batching(bace_df, graph_type, zero_edge_smi):
    expected_e = GRAPH_EDGE_DIMS[graph_type]
    g0 = mol_to_graph(zero_edge_smi, label=np.int64(0), graph_type=graph_type)
    g1 = mol_to_graph(bace_df["smiles"].iloc[0], label=np.int64(1), graph_type=graph_type)
    assert g0.edge_attr.shape == (0, expected_e)
    assert g1.edge_attr.shape[1] == expected_e
    batch = next(iter(DataLoader([g0, g1], batch_size=2)))
    assert batch.edge_attr.shape[1] == expected_e


def test_mol_to_graph_label_types_and_add_feats(bace_df):
    smi = bace_df["smiles"].iloc[0]
    for lbl, expected_dtype in [
        (1, torch.long),
        (np.int64(1), torch.long),
        (np.int32(0), torch.long),
        (True, torch.long),
        (np.bool_(False), torch.long),
        (5.5, torch.float32),
        (np.float32(5.5), torch.float32),
        (np.float64(5.5), torch.float32),
    ]:
        data = mol_to_graph(smi, label=lbl, graph_type="AtomGraph")
        assert data.y is not None and data.y.dtype == expected_dtype

    assert mol_to_graph(smi, label=None).y is None

    with pytest.raises(TypeError):
        mol_to_graph(smi, label="not_a_number")

    extra = [1.0, 2.0, 3.0]
    data_extra = mol_to_graph(smi, label=1, add_feats=extra, add_smi=True)
    assert data_extra.add_feats.shape == (1, 3)
    assert isinstance(data_extra.smi, str) and len(data_extra.smi) > 0
