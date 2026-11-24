from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest


def _mlx_available() -> bool:
    try:
        import mlx.core as mx
        import mlx_graphs  # noqa: F401

        mx.eval(mx.array([1.0]))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mlx_available(),
    reason="MLX / Metal GPU not available in this environment",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_mlx_graphs(bace_df, activity_col="Class", graph_type="AtomGraph", add_feats=None, add_smi=False):
    from mminer_graphs.mlx_models import MolToGraph

    labels = bace_df[activity_col]
    cast = int if activity_col == "Class" else float
    if add_feats is not None:
        return [
            MolToGraph(smi=s, label=cast(y), graph_type=graph_type, add_feats=f, add_smi=add_smi)
            for s, y, f in zip(bace_df["smiles"], labels, add_feats)
        ]
    return [
        MolToGraph(smi=s, label=cast(y), graph_type=graph_type, add_smi=add_smi)
        for s, y in zip(bace_df["smiles"], labels)
    ]


def test_mlx_mol_to_graph_add_smi_and_dataloader(bace_df):
    from mlx_graphs.loaders import Dataloader

    wrapped = _build_mlx_graphs(bace_df.iloc[:6], activity_col="Class", add_smi=True)
    assert all(isinstance(m.smi, str) and len(m.smi) > 0 for m in wrapped)
    loader = Dataloader([m.graph_data for m in wrapped], batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2


def test_mlx_gnns_edge_features_eval_dropout_and_dims(bace_df):
    import mlx.core as mx
    from mlx_graphs.loaders import Dataloader
    from mminer_graphs.mlx_models import GATNet, GCNNet, GINENet, GSAGENet

    graphs = [m.graph_data for m in _build_mlx_graphs(bace_df.iloc[:10], activity_col="Class")]
    batch = next(iter(Dataloader(graphs, batch_size=10, shuffle=False)))

    # All 4 architectures with reduce_graph=True (and multi-head GATNet)
    models = [
        GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=2, reduce_graph=True, graph_dropout=0.5, mlp_dropout=0.5),
        GINENet(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, mlp_layers=2, reduce_graph=True, graph_dropout=0.5, mlp_dropout=0.5),
        GSAGENet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=2, reduce_graph=True, graph_dropout=0.5, mlp_dropout=0.5),
        GATNet(node_dim=142, edge_feat_dim=16, hidden_dim=16, heads=2, graph_layers=2, mlp_layers=2, reduce_graph=True, graph_dropout=0.5, mlp_dropout=0.5),
    ]
    for model in models:
        model.eval()
        args = (
            (batch.edge_index, batch.node_features, batch.edge_features, batch.batch_indices)
            if isinstance(model, (GINENet, GATNet))
            else (batch.edge_index, batch.node_features, batch.batch_indices)
        )
        out1 = np.array(model(*args))
        out2 = np.array(model(*args))
        # Dropout must be disabled during eval() -> deterministic outputs
        assert out1.shape == (10, 2)
        assert np.allclose(out1, out2, atol=1e-6)

    # GINENet must actually use edge_features (changing edge_features changes output)
    gine = models[1]
    gine.eval()
    out_orig = np.array(gine(batch.edge_index, batch.node_features, batch.edge_features, batch.batch_indices))
    zero_edges = mx.zeros_like(batch.edge_features)
    out_zero = np.array(gine(batch.edge_index, batch.node_features, zero_edges, batch.batch_indices))
    assert not np.allclose(out_orig, out_zero, atol=1e-5)


def test_mlx_gnn_fit_restore_best_multiclass_and_save_load(tmp_path, bace_df):
    from mminer_graphs.mlx_models import EarlyStopping, GCNNet, MolToGraph

    reg_graphs = [m.graph_data for m in _build_mlx_graphs(bace_df, activity_col="pIC50")]
    model = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    model.fit(reg_graphs[:30], reg_graphs[30:], epochs=6, batch_size=10, criterion="mse",
              estop=EarlyStopping(patience=3), pbar=False, device_warn=False, restore_best=True)
    expected = model.predict(reg_graphs[30:], batch_size=10, return_type="array")

    # Save / load round-trip + mismatch guard
    reg_path = tmp_path / "mlx_gnn_reg.npz"
    model.save(str(reg_path))
    loaded = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    loaded.load(str(reg_path))
    assert np.allclose(expected, loaded.predict(reg_graphs[30:], batch_size=10, return_type="array"), atol=1e-5)

    clf_wrong = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    with pytest.raises(ValueError, match="Mismatch"):
        clf_wrong.load(str(reg_path))

    # Re-fit clears metrics
    model.fit(reg_graphs[:30], reg_graphs[30:], epochs=3, batch_size=10, criterion="mse",
              pbar=False, device_warn=False)
    assert len(model.metrics.test_loss) == 3

    # 3-class classification
    y3 = pd.qcut(bace_df["pIC50"], q=3, labels=[0, 1, 2]).astype(int).to_numpy()
    cls3_graphs = [MolToGraph(s, int(y)).graph_data for s, y in zip(bace_df["smiles"], y3)]
    clf3 = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=3)
    clf3.fit(cls3_graphs[:30], cls3_graphs[30:], epochs=2, batch_size=10, pbar=False, device_warn=False)
    preds3 = clf3.predict(cls3_graphs[30:], batch_size=10, binary="multi", return_type="array")
    assert preds3.shape == (10,) and set(preds3.tolist()).issubset({0, 1, 2})


def test_mlx_mlp_mixed_inputs_and_save_load(tmp_path, bace_tabular):
    from mminer_graphs.mlx_models import MLP

    mlp = MLP(input_size=8, hidden_size=16, layers=2, num_classes=None)
    assert "model" not in mlp and not hasattr(mlp, "train_loss")

    mlp.fit(
        [row for row in bace_tabular["X_train"]],
        pd.Series(bace_tabular["y_reg_train"], index=range(50, 80)),
        pd.DataFrame(bace_tabular["X_test"]),
        bace_tabular["y_reg_test"].tolist(),
        epochs=3, batch_size=10, criterion="mse", pbar=False, device_warn=False,
    )
    preds = np.array(mlp.predict(pd.DataFrame(bace_tabular["X_test"]), return_type="list"))
    path = tmp_path / "mlx_mlp_reg.npz"
    mlp.save(str(path))
    loaded = MLP(input_size=8, hidden_size=16, layers=2, num_classes=None).load(str(path))
    preds_loaded = np.array(loaded.predict(bace_tabular["X_test"], return_type="list"))
    assert np.allclose(preds, preds_loaded, atol=1e-5)


def test_mlx_crossval_featurize_once_and_drop_last(bace_df):
    from mminer_graphs.mlx_models import CrossVal, GCNNet, MolToGraph

    df_shifted = bace_df.copy()
    df_shifted.index = range(100, 140)
    add_feats = bace_df[["MW", "AlogP", "PSA"]].to_numpy(dtype=np.float32)

    calls = 0
    orig_init = MolToGraph.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_init(self, *args, **kwargs)

    with patch.object(MolToGraph, "__init__", counting_init):
        model = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=2, num_classes=2, add_feats=3)
        cv = CrossVal(model=model, dataset=df_shifted, estop=5)
        # 30 train rows % batch_size=29 == 1 -> drop_last=True avoids BatchNorm size-1 crash!
        scores = cv.get_cv_scores(
            cv=2, smi_col="smiles", activity_col="Class", add_feats=add_feats,
            batch_size=29, epochs=2, drop_last=True, pbar=False, device_warn=False,
        )

    assert len(scores) == 2
    assert calls == len(df_shifted)


def test_mlx_plots_and_opt(tmp_path, bace_df):
    from mminer_graphs.hyperparams.mlx_opt import MLX_opt
    from mminer_graphs.mlx_models import GCNNet, GINENet
    from mminer_graphs.plot import PlotsMLX

    reg_graphs = [m.graph_data for m in _build_mlx_graphs(bace_df, activity_col="pIC50")]
    reg = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    reg.fit(reg_graphs[:30], reg_graphs[30:], epochs=2, batch_size=10, criterion="mse", pbar=False, device_warn=False)
    p_reg = PlotsMLX(model=reg, data_loader=reg_graphs[30:], batch_size=10)
    with patch.object(plt, "show"):
        p_reg.curves(metrics=["mae"])
        p_reg.regression()
    plt.close("all")

    cls_graphs = [m.graph_data for m in _build_mlx_graphs(bace_df, activity_col="Class")]
    clf = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    clf.fit(cls_graphs[:30], cls_graphs[30:], epochs=2, batch_size=10, pbar=False, device_warn=False)
    p_cls = PlotsMLX(model=clf, data_loader=cls_graphs[30:], batch_size=10)
    with patch.object(plt, "show") as mock_show:
        # Passing list of graphs directly into confusion_matrix & classification_report
        p_cls.confusion_matrix(data=cls_graphs[30:], batch_size=10)
        p_cls.classification_report(data=cls_graphs[30:], batch_size=10, savepath=str(tmp_path / "mlx_cr.png"))
        p_cls.roc_curve(savepath=str(tmp_path / "mlx_roc.png"))
        p_cls.pr_curve(savepath=str(tmp_path / "mlx_pr.png"))
        assert mock_show.call_count == 4
    plt.close("all")

    # MLX_opt regression with GINENet on JunctionTreeGraph (auto node_dim=139, edge_dim=6)
    mopt = MLX_opt(model=GINENet, direction="maximize")
    study = mopt.opt_model(
        n_trials=1, dataset=bace_df, smi_col="smiles", activity_col="pIC50", num_classes=None,
        graph_type="JunctionTreeGraph", graph_layers=2, mlp_layers=1, hidden_dims=(16, 16),
        graph_dropout=(0.1, 0.1), mlp_dropout=(0.1, 0.1),
        epochs=(2, 2), batch_size=(10, 10), optimizer=["adam"],
        lr=(1e-3, 1e-3), wt_decay=(1e-5, 1e-5), criterion="mse",
        cv=2, progress=False, n_jobs=1, seed=42,
    )
    assert 0.0 < study.best_value <= 1.0
