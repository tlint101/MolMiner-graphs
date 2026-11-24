import subprocess
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest


def _mlx_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx, mlx_graphs; mx.eval(mx.array([1.0]))"],
            capture_output=True,
            timeout=15,
        )
        return proc.returncode == 0
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


@pytest.mark.parametrize(
    "enc_name,extra_kwargs,expected_dim,uses_edges",
    [
        ("GCN", {"reduce_graph": False}, 16, False),
        ("GCN", {"reduce_graph": True}, 8, False),
        ("GINE", {"edge_feat_dim": 16, "reduce_graph": True}, 8, True),
        ("GSAGE", {"reduce_graph": True}, 8, False),
        ("GAT", {"edge_feat_dim": 16, "heads": 2, "reduce_graph": True}, 16, True),
        ("GAT", {"edge_feat_dim": None, "heads": 2, "reduce_graph": False}, 32, True),
    ],
)
def test_mlx_latent_space_encoders(bace_df, enc_name, extra_kwargs, expected_dim, uses_edges):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mminer_graphs.mlx_models import GAT, GCN, GINE, GSAGE

    enc_map = {"GCN": GCN, "GINE": GINE, "GSAGE": GSAGE, "GAT": GAT}
    graphs = [m.graph_data for m in _build_mlx_graphs(bace_df.iloc[:20], activity_col="Class")]
    enc = enc_map[enc_name](node_dim=142, hidden_dim=16, graph_layers=2, **extra_kwargs)

    assert enc.out_dim == expected_dim
    assert enc._has_param("edge_features") is uses_edges

    tx, ty, vx, vy = enc.split_graph_space(graphs[:12], graphs[12:], batch_size=6)
    sp = enc.extract_graph_space(graphs[12:], batch_size=6)
    assert tx.shape == (12, expected_dim) and vx.shape == (8, expected_dim)
    assert sp.shape == (8, expected_dim) and len(ty) == 12 and len(vy) == 8

    # store_initial_params / reset_params restores weights without polluting parameters()
    enc.store_initial_params()
    param_keys = {k for k, _ in tree_flatten(enc.parameters())}
    assert not any("_initial_params" in k for k in param_keys)
    orig_first_key, orig_first_val = tree_flatten(enc.parameters())[0]
    orig_np = np.array(orig_first_val)
    # Perturb parameter and verify reset restores it
    perturbed = [(k, v + 1.0) if k == orig_first_key else (k, v) for k, v in tree_flatten(enc.parameters())]
    enc.update(tree_unflatten(perturbed))
    mx.eval(enc.parameters())
    assert not np.allclose(np.array(dict(tree_flatten(enc.parameters()))[orig_first_key]), orig_np)
    enc.reset_params()
    assert np.allclose(np.array(dict(tree_flatten(enc.parameters()))[orig_first_key]), orig_np)


def test_mlx_multitarget_mol_to_graph_and_dataloader(bace_df):
    import mlx.core as mx
    from mlx_graphs.loaders import Dataloader
    from mminer_graphs.mlx_models import MolToGraph

    smis = bace_df["smiles"].iloc[:4].tolist()
    targets = [
        [1, np.nan, 45.2],
        np.array([0.0, 2.5, np.nan]),
        pd.Series([1, 3.1, 60.0]),
        [None, 1.8, 52.0],
    ]
    feats = bace_df[["MW", "AlogP"]].iloc[:4].to_numpy(dtype=np.float32)
    wrapped = [
        MolToGraph(s, label=t, graph_type="AtomGraph", add_feats=f, add_smi=True)
        for s, t, f in zip(smis, targets, feats)
    ]
    assert all(isinstance(m.smi, str) for m in wrapped)
    g0 = wrapped[0].graph_data
    assert g0.y.shape == (1, 3) and g0.y.dtype == mx.float32
    assert g0.y_mask.shape == (1, 3) and g0.y_mask.dtype == mx.bool_

    batch = next(iter(Dataloader([m.graph_data for m in wrapped], batch_size=4, shuffle=False)))
    assert batch.y.shape == (4, 3) and batch.y.dtype == mx.float32
    assert batch.y_mask.shape == (4, 3) and batch.y_mask.dtype == mx.bool_
    y_np = np.array(batch.y)
    mask_np = np.array(batch.y_mask)
    assert not np.isnan(y_np).any()
    assert mask_np.tolist() == [
        [True, False, True],
        [True, True, False],
        [True, True, True],
        [False, True, True],
    ]
    assert np.isclose(y_np[2, 1], 3.1, atol=1e-5)

    # Single-task graphs must still omit y_mask
    single = MolToGraph(smis[0], label=1, graph_type="AtomGraph").graph_data
    assert not hasattr(single, "y_mask")


@pytest.mark.parametrize("weighting", ["equal", "manual", "uncertainty"])
def test_mlx_multitask_gnn_end_to_end(tmp_path, bace_df, weighting):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mminer_graphs.mlx_models import (
        EarlyStopping,
        GAT,
        GCN,
        GINE,
        GSAGE,
        MolToGraph,
        MultiTaskGNN,
        MultiTaskMetrics,
        summary,
    )

    cols = ["Class", "AlogP", "PSA"]
    y_mat = bace_df[cols].to_numpy(dtype=np.float32).copy()
    # Sprinkle NaNs + make task 0 all-NaN in the first batch of 10
    for r in range(10):
        y_mat[r, 0] = np.nan
    for r in range(0, 40, 4):
        y_mat[r, 1] = np.nan

    feats = bace_df[["MW", "HBD"]].to_numpy(dtype=np.float32)
    graphs = [
        MolToGraph(smi=s, label=y_mat[i], graph_type="AtomGraph", add_feats=feats[i]).graph_data
        for i, s in enumerate(bace_df["smiles"])
    ]
    tr_g, te_g = graphs[:30], graphs[30:]

    encoder = GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, reduce_graph=True)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={
            "Class": {"type": "classification", "num_classes": 2, "weight": 2.0},
            "AlogP": {"type": "regression", "weight": 1.0},
            "PSA": {"type": "regression", "weight": 0.5, "criterion": "huber"},
        },
        add_feats=2,
        shared_mlp_layers=1,
        head_mlp_layers=2,
        mlp_dropout=0.2,
        loss_weighting=weighting,
    )

    # Non-parameter metadata must not leak into MLX parameters()
    param_keys = {k for k, _ in tree_flatten(model.parameters())}
    assert not any(k.startswith("_") for k in param_keys)
    assert ("log_vars" in param_keys) is (weighting == "uncertainty")

    model.fit(
        tr_g,
        te_g,
        epochs=4,
        batch_size=10,
        lr=0.02,
        estop=EarlyStopping(patience=3),
        pbar=False,
        device_warn=False,
        restore_best=True,
    )

    assert isinstance(model.metrics, MultiTaskMetrics)
    assert all(np.isfinite(x) for x in model.metrics.train_loss)
    assert all(np.isfinite(x) for x in model.metrics.test_loss)
    assert set(model.metrics.tasks.keys()) == set(cols)

    # Deterministic eval predictions (dropout disabled in eval)
    preds1 = model.predict(te_g, batch_size=10, binary=False, return_type="dict")
    preds2 = model.predict(te_g, batch_size=10, binary=False, return_type="dict")
    for c in cols:
        assert np.allclose(preds1[c], preds2[c], atol=1e-6)

    preds_df = model.predict(te_g, batch_size=10, return_type="dataframe")
    assert isinstance(preds_df, pd.DataFrame) and list(preds_df.columns) == cols

    eval_df = model.evaluate(te_g, batch_size=10)
    assert list(eval_df.index) == cols
    assert eval_df.loc["AlogP", "N_Observed"] == int((~np.isnan(y_mat[30:, 1])).sum())

    # summary() works on MLX MultiTaskGNN
    s_text = str(summary(model))
    assert "MultiTaskGNN" in s_text

    # Save / load round-trip + mismatch guard
    ckpt = tmp_path / f"mlx_mt_{weighting}.npz"
    model.save(str(ckpt))
    reloaded = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, reduce_graph=True),
        tasks={
            "Class": {"type": "classification", "num_classes": 2, "weight": 2.0},
            "AlogP": {"type": "regression", "weight": 1.0},
            "PSA": {"type": "regression", "weight": 0.5, "criterion": "huber"},
        },
        add_feats=2,
        shared_mlp_layers=1,
        head_mlp_layers=2,
        mlp_dropout=0.2,
        loss_weighting=weighting,
    ).load(str(ckpt))
    preds_loaded = reloaded.predict(te_g, batch_size=10, binary=False, return_type="dict")
    for c in cols:
        assert np.allclose(preds1[c], preds_loaded[c], atol=1e-5)

    wrong = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, reduce_graph=True),
        tasks=["pIC50", "AlogP", "PSA"],
        default_task="regression",
        add_feats=2,
        shared_mlp_layers=1,
        head_mlp_layers=2,
        loss_weighting=weighting,
    )
    with pytest.raises(ValueError, match="Mismatch"):
        wrong.load(str(ckpt))

    # reset_params restores initial weights and clears scalers
    model.reset_params()
    assert model.is_fitted is False and model.y_means == {} and model.y_stds == {}

    # Smoke-test remaining MLX encoders (GCN, GSAGE, GAT) with homogeneous regression list shorthand
    if weighting == "equal":
        reg_graphs = [
            MolToGraph(smi=s, label=row).graph_data
            for s, row in zip(bace_df["smiles"].iloc[:20], bace_df[["pIC50", "AlogP", "PSA"]].iloc[:20].to_numpy())
        ]
        for enc in [
            GCN(node_dim=142, hidden_dim=16, graph_layers=2, reduce_graph=True),
            GSAGE(node_dim=142, hidden_dim=16, graph_layers=2, reduce_graph=True),
            GAT(node_dim=142, edge_feat_dim=16, hidden_dim=16, heads=2, graph_layers=2, reduce_graph=True),
        ]:
            mt = MultiTaskGNN(encoder=enc, tasks=["pIC50", "AlogP", "PSA"], default_task="regression")
            mt.fit(reg_graphs[:12], reg_graphs[12:], epochs=2, batch_size=6, pbar=False, device_warn=False)
            out = mt.predict(reg_graphs[12:], batch_size=6, return_type="dict")
            assert out["pIC50"].shape == (8,)


def test_mlx_multitask_crossval_and_plots(tmp_path, bace_df):
    from mminer_graphs.mlx_models import CrossVal, GCN, GINE, MolToGraph, MultiTaskGNN
    from mminer_graphs.plot import PlotsMLX

    cols = ["Class", "AlogP", "PSA"]
    df_mt = bace_df.copy()
    df_mt.index = [f"mol_{i}" for i in range(len(df_mt))]
    df_mt.loc[df_mt.index[0], "AlogP"] = np.nan

    calls = 0
    orig_init = MolToGraph.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_init(self, *args, **kwargs)

    model = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
    )
    cv = CrossVal(model=model, dataset=df_mt, estop=5)
    with patch.object(MolToGraph, "__init__", counting_init):
        scores = cv.get_cv_scores(
            cv=2, smi_col="smiles", activity_col=cols,
            batch_size=10, epochs=2, pbar=False, device_warn=False,
        )

    assert calls == len(df_mt)
    assert scores.index.names == ["fold", "Task"]
    assert list(cv.summary.index) == cols
    assert cv.macro_summary is not None and list(cv.macro_summary.index) == ["mean", "std"]

    # PlotsMLX across all 7 plot methods with sparse NaN in test split
    y_mat = bace_df[cols].to_numpy(dtype=np.float32).copy()
    y_mat[30, 1] = np.nan
    y_mat[31, 0] = np.nan
    graphs = [MolToGraph(s, label=y_mat[i]).graph_data for i, s in enumerate(bace_df["smiles"])]
    tr_g, te_g = graphs[:30], graphs[30:]

    m1 = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
    )
    m1.fit(tr_g, te_g, epochs=3, batch_size=10, pbar=False, device_warn=False)
    m2 = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
    )
    m2.fit(tr_g, te_g, epochs=3, batch_size=10, pbar=False, device_warn=False)

    p = PlotsMLX(model=m1, data_loader=te_g, batch_size=10)
    with patch.object(plt, "show") as mock_show:
        p.curves(metrics=["loss"])
        p.curves(metrics=["loss", "accuracy"], task="Class")
        p.curves(metrics=["loss", "mae"], task="AlogP")
        p.regression(task="AlogP", savepath=str(tmp_path / "mlx_mt_reg.png"))
        p.residuals(train_data=tr_g, task="PSA", batch_size=10)
        p.confusion_matrix(task="Class", batch_size=10)
        p.classification_report(task="Class", batch_size=10)
        p.roc_curve(task="Class")
        p.roc_curve(model=[m1, m2], model_names=["GINE", "GCN"], color=["blue", "green"], task="Class")
        p.pr_curve(task="Class", chance_level=True)
        p.pr_curve(model=[m1, m2], model_names=["GINE", "GCN"], color=["blue", "green"], task="Class")
        assert mock_show.call_count == 11
    plt.close("all")

    with pytest.raises(ValueError, match="Please specify task"):
        p.regression()
    with pytest.raises(ValueError, match="expected 'regression'"):
        p.regression(task="Class")


def test_mlx_opt_multitask_and_build(tmp_path, bace_df):
    from mminer_graphs.hyperparams.mlx_opt import MLX_opt
    from mminer_graphs.mlx_models import GINE, MolToGraph, MultiTaskGNN

    mixed_tasks = {
        "Class": {"type": "classification", "num_classes": 2},
        "AlogP": {"type": "regression"},
        "PSA": {"type": "regression", "criterion": "huber"},
    }
    mopt = MLX_opt(model=MultiTaskGNN, direction="maximize")
    study = mopt.opt_model(
        encoder_cls=GINE,
        tasks=mixed_tasks,
        primary_task="Class",
        n_trials=2,
        dataset=bace_df,
        smi_col="smiles",
        activity_col=["Class", "AlogP", "PSA"],
        graph_layers=2,
        hidden_dims=(16, 16),
        graph_dropout=(0.1, 0.1),
        mlp_dropout=(0.1, 0.1),
        shared_mlp_layers=(0, 1),
        head_mlp_layers=(1, 2),
        loss_weighting=["equal", "uncertainty"],
        epochs=(2, 2),
        batch_size=(10, 10),
        optimizer=["adam"],
        lr=(1e-3, 1e-3),
        wt_decay=(1e-5, 1e-5),
        cv=2,
        progress=False,
        n_jobs=1,
        seed=42,
    )
    assert 0.0 < study.best_value <= 1.0
    json_path = tmp_path / "mlx_mt_best.json"
    best_params = mopt.best_solutions(study, savepath=str(json_path))

    y_mat = bace_df[["Class", "AlogP", "PSA"]].to_numpy(dtype=np.float32)
    graphs = [MolToGraph(s, label=row).graph_data for s, row in zip(bace_df["smiles"], y_mat)]
    built = mopt.build_opt_model(
        model=MultiTaskGNN,
        encoder_cls=GINE,
        tasks=mixed_tasks,
        train_data=graphs[:30],
        test_data=graphs[30:],
        params=str(json_path),
        estop=5,
    )
    assert isinstance(built, MultiTaskGNN) and built.is_fitted is True

    # Also verify MLX_opt.build_model reloads saved MultiTaskGNN checkpoint + JSON params
    ckpt_path = tmp_path / "mlx_mt_opt.npz"
    built.save(str(ckpt_path))
    rebuilt = mopt.build_model(
        model=MultiTaskGNN,
        encoder_cls=GINE,
        tasks=mixed_tasks,
        model_path=str(ckpt_path),
        params=str(json_path),
    )
    p1 = built.predict(graphs[30:], batch_size=10, binary=False, return_type="dict")
    p2 = rebuilt.predict(graphs[30:], batch_size=10, binary=False, return_type="dict")
    for k in ["Class", "AlogP", "PSA"]:
        assert np.allclose(p1[k], p2[k], atol=1e-5)
