from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mminer_graphs.model.model_utils import mol_to_graph

REPO_ROOT = Path(__file__).resolve().parent.parent
BACE_CSV = REPO_ROOT / "tutorials" / "dataset" / "bace.csv"
DESCRIPTOR_COLS = ["MW", "AlogP", "PSA", "HBA", "HBD", "RB", "HeavyAtomCount", "RingCount"]


@pytest.fixture(scope="session")
def bace_df() -> pd.DataFrame:
    """Balanced 40-row slice of tutorials/dataset/bace.csv with 2 clean CV folds."""
    df_full = pd.read_csv(BACE_CSV)
    neg = df_full[df_full["Class"] == 0].iloc[:20]
    pos = df_full[df_full["Class"] == 1].iloc[:20]
    # Interleave classes so slicing [:30] / [30:] always contains both classes
    df = pd.concat([neg, pos]).iloc[
        [idx for pair in zip(range(20), range(20, 40)) for idx in pair]
    ].reset_index(drop=True)

    # Explicit balanced 2-fold CV columns (30 train / 10 test, divisible by batch_size=10)
    df["CV0"] = ["test" if i < 10 else "train" for i in range(40)]
    df["CV1"] = ["test" if 10 <= i < 20 else "train" for i in range(40)]
    return df


@pytest.fixture(scope="session")
def bace_cls_graphs(bace_df):
    """Pre-featurized PyG AtomGraph train/test split for binary classification."""
    graphs = [
        mol_to_graph(smi, int(y), graph_type="AtomGraph")
        for smi, y in zip(bace_df["smiles"], bace_df["Class"])
    ]
    return graphs[:30], graphs[30:]


@pytest.fixture(scope="session")
def bace_reg_graphs(bace_df):
    """Pre-featurized PyG AtomGraph train/test split for pIC50 regression."""
    graphs = [
        mol_to_graph(smi, float(y), graph_type="AtomGraph")
        for smi, y in zip(bace_df["smiles"], bace_df["pIC50"])
    ]
    return graphs[:30], graphs[30:]


@pytest.fixture(scope="session")
def bace_tabular(bace_df):
    """Standardized BACE 2D descriptor matrix split into train (30) / test (10)."""
    X = bace_df[DESCRIPTOR_COLS].to_numpy(dtype=np.float32)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    y_cls = bace_df["Class"].to_numpy(dtype=np.int64)
    y_reg = bace_df["pIC50"].to_numpy(dtype=np.float32)
    return {
        "X_train": X[:30],
        "X_test": X[30:],
        "y_cls_train": y_cls[:30],
        "y_cls_test": y_cls[30:],
        "y_reg_train": y_reg[:30],
        "y_reg_test": y_reg[30:],
    }
