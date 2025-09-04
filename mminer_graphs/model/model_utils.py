import random
import torch
import numpy as np
from typing import Optional, Union
import torch.nn.functional as F
from torch_geometric.data import Data
from mminer_graphs.graph import AtomGraph, PharmacophoreGraph, FunctionalGraph, JunctionTreeGraph, ClusterGraph

# optional import based on workstation
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as optim
except ImportError:
    mx = None
    nn = None
    optim = None

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

F_ACT = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
    "gelu": F.gelu,
    "elu": F.elu,
    "silu": F.silu,  # Also known as Swish
    "tanh": F.tanh,
}

if optim is not None:
    MLX_OPTI = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.AdaDelta,
        "adamax": optim.Adamax,
    }

    MLX_ACT = {
        "relu": mlx_nn.relu,
        "leaky_relu": lambda x: mlx_nn.leaky_relu(x, negative_slope=0.01),
        "gelu": mlx_nn.gelu,
        "elu": mlx_nn.elu,
        "silu": mlx_nn.silu,  # Also known as Swish
        "tanh": mlx_nn.tanh,
    }

    MLX_CRITERIONS = {
        # Classification
        "cross_entropy": mlx_nn.losses.cross_entropy,
        "nll": mlx_nn.losses.nll_loss,
        "kl_divergence": mlx_nn.losses.kl_div_loss,
        # "bce_with_logits": mlx_nn.losses.BCEWithLogitsLoss,
        # "poisson": mlx_nn.PoissonNLLLoss,
        # Regression
        "mse": mlx_nn.losses.mse_loss,
        "l1": mlx_nn.losses.l1_loss,
        "smooth_l1": mlx_nn.losses.smooth_l1_loss,
        "huber": mlx_nn.losses.huber_loss,
        # Others
        "cosine_similarity": mlx_nn.losses.cosine_similarity_loss,
        "triplet_margin": mlx_nn.losses.triplet_loss,
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

    node = torch.tensor(np.array(graph.node_features), dtype=torch.float)
    # convert node features into a vector for GNN
    if node.ndim == 1:
        node = node.unsqueeze(1)

    # edge_index: bond connections
    edge_index = graph.edge_index
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # bond features
    bond = graph.edge_features
    bond = torch.tensor(np.array(graph.edge_features), dtype=torch.float).contiguous()

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
def get_torch_device(device_type="auto"):
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


# support function to get apple silicon gpu, cpu, or cuda
def get_mlx_device(device_type="auto"):
    if device_type == "auto":
        device = mx.gpu
    elif device_type == "gpu":
        device = mx.gpu
    elif device_type == "cpu":
        device = mx.cpu
    elif device_type == "cuda":  # todo test on cuda enabled machine
        if not mx.cuda.is_available():
            raise ValueError("CUDA not available!")
        else:
            device = mx.cuda
    else:
        raise ValueError(
            f"Trying to run device on {device_type}. Only 'auto', 'cuda', 'gpu'(cuda), 'mps' or 'cpu' are supported!!")
    return device


# run to see what is available on machine
def check_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Device: {device_name}")
    elif torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available. Device: Apple GPU")
    else:
        print("Only CPU is available. Check CUDA configurations!")


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
