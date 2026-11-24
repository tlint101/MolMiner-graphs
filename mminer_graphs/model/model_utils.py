import random
import torch
import subprocess
import numpy as np
from typing import Optional, Union
import torch.nn.functional as F
from torch_geometric.data import Data
from mminer_graphs.graph import AtomGraph, PharmacophoreGraph, FunctionalGraph, JunctionTreeGraph, ClusterGraph

graph_type_dict = {
    "AtomGraph": AtomGraph,
    "PharmacophoreGraph": PharmacophoreGraph,
    "FunctionalGraph": FunctionalGraph,
    "JunctionTreeGraph": JunctionTreeGraph,
    "ClusterGraph": ClusterGraph,
}

GRAPH_EDGE_DIMS = {
    "AtomGraph": 16,
    "PharmacophoreGraph": 3,
    "FunctionalGraph": 20,
    "JunctionTreeGraph": 6,
    "ClusterGraph": 6,
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

TORCH_CLASSIFICATION = {
    # Classification
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "bce_with_logits": torch.nn.BCEWithLogitsLoss(),
    "nll": torch.nn.NLLLoss(),
    "poisson": torch.nn.PoissonNLLLoss(),
    "kl_divergence": torch.nn.KLDivLoss(),
}

TORCH_REGRESSION = {
# Regression
    "mse": torch.nn.MSELoss(),
    "l1": torch.nn.L1Loss(),
    "smooth_l1": torch.nn.SmoothL1Loss(),
    "huber": torch.nn.HuberLoss(),
    # Others
    # "cosine_embedding": torch.nn.CosineEmbeddingLoss(),
    # "triplet_margin": torch.nn.TripletMarginLoss(),
}

# combine criterion dict
TORCH_CRITERIONS = {**TORCH_CLASSIFICATION, **TORCH_REGRESSION}

F_ACT = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
    "gelu": F.gelu,
    "elu": F.elu,
    "silu": F.silu,  # Also known as Swish
    "tanh": F.tanh,
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

    # convert to np.array() first then to torch.tensor
    node_np = np.array(graph.node_features)
    node = torch.tensor(node_np, dtype=torch.float)
    # convert node features into a vector for GNN
    if node.ndim == 1:
        node = node.unsqueeze(1)

    # edge_index: bond connections
    edge_index = graph.edge_index
    # if single atom
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    # full mol
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # bond features
    bond = np.array(graph.edge_features)
    # if single atom or single-clique graph (0 edges)
    if len(bond) == 0:
        edge_dim = GRAPH_EDGE_DIMS[graph_type]
        bond = torch.empty((0, edge_dim), dtype=torch.float)
    # full mol
    else:
        bond = torch.tensor(bond, dtype=torch.float).contiguous()

    # append smi string
    smiles = graph.smi

    if label is not None:
        if isinstance(label, (bool, np.bool_)):
            y = torch.tensor([int(label)], dtype=torch.long)
        elif isinstance(label, (int, np.integer)):
            y = torch.tensor([int(label)], dtype=torch.long)
        elif isinstance(label, (float, np.floating)):
            y = torch.tensor([float(label)], dtype=torch.float32)
        else:
            raise TypeError(
                f"Unsupported label type {type(label).__name__}; expected int, float, numpy scalar, or None."
            )
    else:
        y = None

    # for additional feats,  store as [1, features] tensor for DataLoader to stack batches [batch, features]
    if add_feats is not None:
        add_feats = torch.tensor(np.asarray(add_feats), dtype=torch.float32).reshape(1, -1)

    if add_smi and add_feats is not None:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y, smi=smiles, add_feats=add_feats)
    elif add_smi:
        return Data(x=node, edge_index=edge_index, edge_attr=bond, y=y, smi=smiles)
    elif add_feats is not None:
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


# run to see what is available on machine
def check_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Device: {device_name}")
    elif torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available. Device: Apple GPU")
    else:
        print("Only CPU is available. Check CUDA configurations!")


def state_apple_silicon_device():
    output = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                            capture_output=True, text=True)
    return output.stdout.strip()


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


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNG deterministically in PyTorch DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch.Generator for reproducible DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


if __name__ == "__main__":
    import doctest

    doctest.testmod()
