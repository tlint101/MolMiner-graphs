from unittest.mock import patch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader

from mminer_graphs.model.crossval import CrossVal
from mminer_graphs.model.graph_modeling import GCNNet, MolToGraph
from mminer_graphs.model.model_utils import make_generator, seed_worker


def test_crossval_sliced_and_string_indexed_with_add_feats(bace_df):
    add_feats = bace_df[["MW", "AlogP", "PSA"]].to_numpy(dtype=np.float32)

    # Shifted non-0-based integer index (100..139)
    df_shifted = bace_df.copy()
    df_shifted.index = range(100, 140)
    cv1 = CrossVal(
        model=GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2, add_feats=3),
        dataset=df_shifted,
        estop=5,
    )
    splits = cv1.get_cv_split(df_shifted, labels="CV", cv=2)
    for tr_idx, te_idx in splits:
        assert tr_idx.min() >= 0 and tr_idx.max() < len(df_shifted)
        assert te_idx.min() >= 0 and te_idx.max() < len(df_shifted)
    scores1 = cv1.get_cv_scores(
        cv=2, smi_col="smiles", activity_col="Class", add_feats=add_feats,
        batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
    )
    assert len(scores1) == 2

    # String index ("mol_0".."mol_39")
    df_str = bace_df.copy()
    df_str.index = [f"mol_{i}" for i in range(len(df_str))]
    cv2 = CrossVal(
        model=GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2, add_feats=3),
        dataset=df_str,
        estop=5,
    )
    scores2 = cv2.get_cv_scores(
        cv=2, smi_col="smiles", activity_col="Class", add_feats=add_feats,
        batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
    )
    assert len(scores2) == 2


def test_crossval_model_override(bace_df):
    model_init = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    model_override = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    cv = CrossVal(model=model_init, dataset=bace_df, estop=5)

    scores = cv.get_cv_scores(
        model=model_override, cv=2, smi_col="smiles", activity_col="pIC50",
        criterion="mse", batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
    )
    assert model_override.is_fitted is True
    assert model_init.is_fitted is False
    assert {"r2", "MAE", "RMSE", "Loss"}.issubset(scores.columns)


def test_crossval_featurizes_once(bace_df):
    calls = 0
    orig_init = MolToGraph.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_init(self, *args, **kwargs)

    with patch.object(MolToGraph, "__init__", counting_init):
        model = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
        cv = CrossVal(model=model, dataset=bace_df, estop=5)
        cv.get_cv_scores(
            cv=2, smi_col="smiles", activity_col="Class",
            batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
        )

    assert calls == len(bace_df)


def test_make_generator_reproducible_without_clobbering_rng():
    np.random.seed(999)
    _ = np.random.rand()
    state_before = np.random.get_state()[1].copy()

    dl1 = TorchDataLoader(list(range(20)), batch_size=5, shuffle=True,
                          worker_init_fn=seed_worker, generator=make_generator(42))
    dl2 = TorchDataLoader(list(range(20)), batch_size=5, shuffle=True,
                          worker_init_fn=seed_worker, generator=make_generator(42))

    assert np.array_equal(np.random.get_state()[1], state_before)
    assert [b.tolist() for b in dl1] == [b.tolist() for b in dl2]
