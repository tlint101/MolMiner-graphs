import numpy as np
import random
from typing import Optional, Union
import torch
from torch_geometric.data import Data
from mminer_graphs.graph import AtomGraph, PharmacophoreGraph, FunctionalGraph, JunctionTreeGraph, ClusterGraph

graph_type_dict = {
    "AtomGraph": AtomGraph,
    "PharmacophoreGraph": PharmacophoreGraph,
    "FunctionalGraph": FunctionalGraph,
    "JunctionTreeGraph": JunctionTreeGraph,
    "ClusterGraph": ClusterGraph,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
}

CRITERIONS = {
    # Classification
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "bce_with_logits": torch.nn.BCEWithLogitsLoss,
    "nll": torch.nn.NLLLoss,
    "poisson": torch.nn.PoissonNLLLoss,
    "kl_divergence": torch.nn.KLDivLoss,
    # Regression
    "mse": torch.nn.MSELoss,
    "l1": torch.nn.L1Loss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "huber": torch.nn.HuberLoss,
    # Others
    "cosine_embedding": torch.nn.CosineEmbeddingLoss,
    "triplet_margin": torch.nn.TripletMarginLoss,
}


def mol_to_graph(smi: str, label: Union[int, float] = None, graph_type: str = "AtomGraph", add_smi: bool = False,
                 add_feats: Optional[Union[np.array, list]] = None, with_atom_index: bool = False):
    """
    Support function to convert molecules into graph for torch, used inside class MolToGraph.
    :param smi: str
        Molecule as a smiles string.
    :param label: pd.series
        Associated label for each molecule.
    :param graph_type: str
        The molecular graph type to generate from smiles string. Can be 'AtomGraph', 'PharmacophoreGraph'
        'FunctionalGraph', 'JunctionTreeGraph', or 'ClusterGraph'.
    :param add_feats: Optional[Union[np.array, list]]
        Include additional information to include with the molecular graph tensor. Must be given as a np.array or a
        list.
    :param with_atom_index: bool
        To include the smi string with atom index. Defaults to False.
    :return:
    """
    if graph_type in graph_type_dict.keys():
        graph = graph_type_dict[graph_type](smi, with_atom_index=with_atom_index)
    elif graph_type is None:
        raise TypeError("Must specify 'graph_type'!!!")
    else:
        raise TypeError(
            "Only 'AtomGraph', 'PharmacophoreGraph', 'FunctionalGraph', 'JunctionTreeGraph', or 'ClusterGraph' are supported!!")

    node = torch.tensor(graph.node_features, dtype=torch.float)
    # convert node features into a vector for GNN
    if node.ndim == 1:
        node = node.unsqueeze(1)

    # edge_index: bond connections
    edge_index = graph.edge_index
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # bond features
    bond = graph.edge_features
    bond = torch.tensor(bond, dtype=torch.float).contiguous()

    # append smi string
    smiles = graph.smi

    if label is not None:
        y = torch.tensor([label], dtype=torch.long)
    else:
        y = None

    if add_smi and add_feats:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y, smi=smiles, add_feats=add_feats)
    elif add_smi:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y, smi=smiles)
    elif add_feats:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y, add_feats=add_feats)
    else:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y)


# support function to determine if gpu, mps, or cpu will be used for torch models
def get_device(device_type="auto"):
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_type == "gpu":
        device = torch.device("cuda")
    elif device_type == "cuda":
        device = torch.device("cuda")
    elif device_type == "mps":
        device = torch.device("mps")
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            f"Trying to run device on {device_type}. Only 'auto', 'cuda', 'gpu'(cuda), 'mps' or 'cpu' are supported!!")

    return device


# set seed for pytorch, numpy and python
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    import doctest

    doctest.testmod()
