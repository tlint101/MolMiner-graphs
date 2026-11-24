from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from mminer_graphs import MultiTaskGNN as TopLevelMultiTaskGNN
from mminer_graphs.model import (
    CrossVal,
    EarlyStopping,
    GAT,
    GCN,
    GIN,
    GINE,
    GSAGE,
    MolToGraph,
    MultiTaskGNN,
    MultiTaskMetrics,
    summary,
)
from mminer_graphs.model.model_utils import set_seed
from mminer_graphs.plot import Plots


def _build_bace_multitask_graphs(bace_df, cols, nan_positions=None, add_feats=None):
    y_mat = bace_df[cols].to_numpy(dtype=np.float32).copy()
    if nan_positions:
        for r, c in nan_positions:
            y_mat[r, c] = np.nan

    graphs = []
    for i, smi in enumerate(bace_df["smiles"]):
        extra = add_feats[i] if add_feats is not None else None
        graphs.append(MolToGraph(smi=smi, label=y_mat[i], graph_type="AtomGraph", add_feats=extra).graph_data)
    return graphs, y_mat


@pytest.mark.parametrize(
    "enc_cls,enc_kwargs",
    [
        (GCN, {"node_dim": 142, "hidden_dim": 16, "graph_layers": 2, "reduce_graph": True}),
        (GIN, {"node_dim": 142, "hidden_dim": 16, "graph_layers": 2, "reduce_graph": True}),
        (GINE, {"node_dim": 142, "edge_feat_dim": 16, "hidden_dim": 16, "graph_layers": 2, "reduce_graph": True}),
        (GSAGE, {"node_dim": 142, "hidden_dim": 16, "graph_layers": 2, "reduce_graph": True}),
        (GAT, {"node_dim": 142, "edge_feat_dim": 16, "hidden_dim": 16, "heads": 2, "graph_layers": 2, "reduce_graph": True}),
    ],
)
def test_multitask_all_five_encoders_mixed_tasks(bace_df, enc_cls, enc_kwargs):
    assert TopLevelMultiTaskGNN is MultiTaskGNN
    cols = ["Class", "AlogP", "PSA"]
    graphs, _ = _build_bace_multitask_graphs(bace_df, cols)
    tr_g, te_g = graphs[:30], graphs[30:]

    encoder = enc_cls(**enc_kwargs)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression", "criterion": "huber"},
        },
        shared_mlp_layers=1,
        head_mlp_layers=2,
        mlp_dropout=0.1,
    )
    assert model.task == "MultiTask"
    assert model.task_names == cols
    model.fit(tr_g, te_g, epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)

    assert isinstance(model.metrics, MultiTaskMetrics)
    assert len(model.metrics.train_loss) == 2
    assert len(model.metrics.test_loss) == 2
    assert set(model.metrics.tasks.keys()) == set(cols)
    assert len(model.metrics.tasks["Class"].test_accuracy) == 2
    assert len(model.metrics.tasks["AlogP"].test_rmse) == 2
    assert model.metrics.tasks["PSA"].best_epoch == model.metrics.best_epoch

    preds_dict = model.predict(te_g, batch_size=10, device="cpu", return_type="dict")
    assert set(preds_dict.keys()) == set(cols)
    assert preds_dict["Class"].shape == (10,) and set(preds_dict["Class"].tolist()).issubset({0, 1})
    assert preds_dict["AlogP"].shape == (10,) and preds_dict["PSA"].shape == (10,)

    preds_df = model.predict(te_g, batch_size=10, device="cpu", return_type="dataframe")
    assert isinstance(preds_df, pd.DataFrame) and list(preds_df.columns) == cols
    assert len(preds_df) == 10

    eval_df = model.evaluate(te_g, batch_size=10, device="cpu")
    assert list(eval_df.index) == cols
    assert eval_df.loc["Class", "TaskType"] == "Classification"
    assert eval_df.loc["AlogP", "TaskType"] == "Regression"
    assert np.isfinite(eval_df.loc["AlogP", "RMSE"])
    assert np.isnan(eval_df.loc["AlogP", "Accuracy"])
    assert np.isnan(eval_df.loc["Class", "RMSE"])


def test_multitask_homogeneous_list_shorthand_and_multiclass(bace_df):
    # Homogeneous regression list shorthand: ["pIC50", "AlogP", "PSA"]
    cols = ["pIC50", "AlogP", "PSA"]
    graphs, _ = _build_bace_multitask_graphs(bace_df, cols)
    model_r = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks=cols,
        default_task="regression",
    )
    model_r.fit(graphs[:30], graphs[30:], epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    assert set(model_r.y_means.keys()) == set(cols)
    assert set(model_r.y_stds.keys()) == set(cols)

    # Mixed binary (BCE + CE) + 3-class classification + regression
    df_m = bace_df.copy()
    df_m["Class3"] = pd.qcut(df_m["pIC50"], q=3, labels=[0, 1, 2]).astype(int)
    mcols = ["Class", "Class3", "AlogP"]
    mgraphs, _ = _build_bace_multitask_graphs(df_m, mcols)

    model_m = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2, "criterion": "bce_with_logits"},
            "Class3": {"type": "classification", "num_classes": 3, "criterion": "cross_entropy"},
            "AlogP": {"type": "regression", "criterion": "l1"},
        },
    )
    model_m.fit(mgraphs[:30], mgraphs[30:], epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)

    proba_dict = model_m.predict(mgraphs[30:], batch_size=10, device="cpu", binary="proba", return_type="dict")
    assert proba_dict["Class"].shape == (10, 2)
    assert proba_dict["Class3"].shape == (10, 3)
    assert np.allclose(proba_dict["Class"].sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(proba_dict["Class3"].sum(axis=1), 1.0, atol=1e-5)

    proba_df = model_m.predict(mgraphs[30:], batch_size=10, device="cpu", binary="proba", return_type="dataframe")
    assert "Class3_class_2" in proba_df.columns and "AlogP" in proba_df.columns

    multi_dict = model_m.predict(mgraphs[30:], batch_size=10, device="cpu", binary="multi", return_type="dict")
    assert set(multi_dict["Class3"].tolist()).issubset({0, 1, 2})


def test_multitask_sparse_nan_masking_and_all_nan_batch(bace_df):
    cols = ["Class", "AlogP", "PSA"]
    # Make task 0 completely NaN in the first 10 training rows (an entire mini-batch of size 10!)
    # and sprinkle NaNs across train & test for tasks 1 and 2
    nan_positions = [(r, 0) for r in range(10)] + [(r, 1) for r in range(0, 40, 4)] + [(r, 2) for r in range(1, 40, 5)]
    graphs, y_mat = _build_bace_multitask_graphs(bace_df, cols, nan_positions=nan_positions)
    tr_g, te_g = graphs[:30], graphs[30:]

    model = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
        loss_weighting="uncertainty",
    )
    model.fit(tr_g, te_g, epochs=3, batch_size=10, device="cpu", pbar=False, device_warn=False)

    # Losses and parameters must remain completely finite despite NaNs and an all-NaN batch
    assert all(np.isfinite(x) for x in model.metrics.train_loss)
    assert all(np.isfinite(x) for x in model.metrics.test_loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())

    # Regression scalers must equal mean/std of observed training entries only
    obs_alogp = y_mat[:30, 1][~np.isnan(y_mat[:30, 1])]
    assert np.isclose(model.y_means["AlogP"], float(np.mean(obs_alogp)), atol=1e-5)
    assert np.isclose(model.y_stds["AlogP"], float(np.std(obs_alogp, ddof=0)), atol=1e-5)

    # evaluate() counts only observed test entries per task
    eval_df = model.evaluate(te_g, batch_size=10, device="cpu")
    assert eval_df.loc["AlogP", "N_Observed"] == int((~np.isnan(y_mat[30:, 1])).sum())
    assert eval_df.loc["PSA", "N_Observed"] == int((~np.isnan(y_mat[30:, 2])).sum())


@pytest.mark.parametrize("weighting", ["equal", "manual", "uncertainty"])
def test_multitask_loss_weightings_and_add_feats(bace_df, weighting):
    cols = ["Class", "AlogP", "PSA"]
    feats = bace_df[["MW", "HBD", "HBA"]].to_numpy(dtype=np.float32)
    graphs, _ = _build_bace_multitask_graphs(bace_df, cols, add_feats=feats)

    model = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2, "weight": 2.0},
            "AlogP": {"type": "regression", "weight": 1.0},
            "PSA": {"type": "regression", "weight": 0.5, "criterion": "smooth_l1"},
        },
        add_feats=3,
        loss_weighting=weighting,
    )
    if weighting == "uncertainty":
        assert model.log_vars is not None and model.log_vars.shape == (3,)
        init_lv = model.log_vars.detach().clone()
    else:
        assert model.log_vars is None

    model.fit(graphs[:30], graphs[30:], epochs=4, batch_size=10, lr=0.05, device="cpu", pbar=False, device_warn=False)
    if weighting == "uncertainty":
        # Learnable log_vars should update during backprop
        assert not torch.allclose(model.log_vars.detach(), init_lv)

    preds = model.predict(graphs[30:], batch_size=10, device="cpu", add_feats=3, return_type="dict")
    assert preds["AlogP"].shape == (10,)


def test_multitask_restore_best_reset_and_save_load(tmp_path, bace_df):
    cols = ["Class", "AlogP", "PSA"]
    graphs, _ = _build_bace_multitask_graphs(bace_df, cols)
    tr_g, te_g = graphs[:30], graphs[30:]

    set_seed(0)
    model = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
        loss_weighting="uncertainty",
    )
    model.fit(
        tr_g,
        te_g,
        epochs=12,
        batch_size=10,
        lr=0.05,
        device="cpu",
        estop=EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        pbar=False,
        device_warn=False,
        restore_best=True,
    )

    # Save / load round-trip preserves exact predictions and regression scalers
    expected = model.predict(te_g, batch_size=10, device="cpu", binary=False, return_type="dict")
    ckpt = tmp_path / "mt_model.pth"
    model.save(str(ckpt))

    reloaded = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
        loss_weighting="uncertainty",
    ).load(str(ckpt), device="cpu")
    loaded_preds = reloaded.predict(te_g, batch_size=10, device="cpu", binary=False, return_type="dict")
    for name in cols:
        assert np.allclose(expected[name], loaded_preds[name], atol=1e-6)
    assert reloaded.y_means == model.y_means and reloaded.y_stds == model.y_stds

    # Mismatch guard: wrong task list raises ValueError
    wrong_tasks = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks=["pIC50", "AlogP", "PSA"],
        default_task="regression",
        loss_weighting="uncertainty",
    )
    with pytest.raises(ValueError, match="Mismatch"):
        wrong_tasks.load(str(ckpt), device="cpu")

    # Mismatch guard: incompatible loss_weighting raises ValueError
    wrong_w = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
        loss_weighting="equal",
    )
    with pytest.raises(ValueError, match="Mismatch"):
        wrong_w.load(str(ckpt), device="cpu")

    # reset_params resets encoder, heads, log_vars, and clears scalers
    trained_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    model.reset_params()
    assert model.is_fitted is False and model.y_means == {} and model.y_stds == {}
    assert torch.allclose(model.log_vars, torch.zeros_like(model.log_vars))
    changed = [k for k, v in model.named_parameters() if "weight" in k and not torch.allclose(v, trained_params[k])]
    assert len(changed) >= 3

    # Also check special characters in task names (dots, hyphens, spaces) + summary()
    dotted = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks=["pIC50.v1", "NR-AR", "Log P"],
        default_task="regression",
    )
    assert dotted.get_head("pIC50.v1") is dotted.heads[0]
    s = summary(dotted, data=tr_g, batch_size=5)
    assert isinstance(s, str) and "MultiTaskGNN" in s


def test_multitask_crossval_featurizes_once_and_summaries(bace_df):
    cols = ["Class", "AlogP", "PSA"]
    df_mt = bace_df.copy()
    df_mt.index = [f"mol_{i}" for i in range(len(df_mt))]
    # Inject sparse NaNs to confirm CrossVal handles missing assays
    df_mt.loc[df_mt.index[0], "AlogP"] = np.nan
    df_mt.loc[df_mt.index[5], "PSA"] = np.nan

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
            batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
        )

    assert calls == len(df_mt)
    assert scores.index.names == ["fold", "Task"]
    assert len(scores) == 2 * len(cols)
    assert list(cv.summary.index) == cols
    assert cv.macro_summary is not None and list(cv.macro_summary.index) == ["mean", "std"]

    # Homogeneous regression CrossVal with activity_col=None (defaults to model.task_names)
    reg_cols = ["pIC50", "AlogP", "PSA"]
    reg_mt = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks=reg_cols,
        default_task="regression",
    )
    cv_reg = CrossVal(model=reg_mt, dataset=bace_df, estop=5)
    reg_scores = cv_reg.get_cv_scores(
        cv=2, smi_col="smiles", activity_col=None,
        batch_size=10, epochs=2, device="cpu", pbar=False, device_warn=False,
    )
    assert list(reg_scores.columns) == ["r2", "MAE", "RMSE", "Loss"]
    assert list(cv_reg.summary.index) == reg_cols


def test_multitask_plots_all_methods_with_sparse_nans(tmp_path, bace_df):
    cols = ["Class", "AlogP", "PSA"]
    # Add sparse NaNs in test split too so plotting filters observed samples cleanly
    nan_positions = [(30, 1), (31, 2), (32, 0)]
    graphs, _ = _build_bace_multitask_graphs(bace_df, cols, nan_positions=nan_positions)
    tr_g, te_g = graphs[:30], graphs[30:]

    m1 = MultiTaskGNN(
        encoder=GINE(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
    )
    m1.fit(tr_g, te_g, epochs=3, batch_size=10, device="cpu", pbar=False, device_warn=False)
    m2 = MultiTaskGNN(
        encoder=GCN(node_dim=142, hidden_dim=16, graph_layers=2),
        tasks={
            "Class": {"type": "classification", "num_classes": 2},
            "AlogP": {"type": "regression"},
            "PSA": {"type": "regression"},
        },
    )
    m2.fit(tr_g, te_g, epochs=3, batch_size=10, device="cpu", pbar=False, device_warn=False)

    p = Plots(model=m1, data_loader=te_g, device="cpu")
    with patch.object(plt, "show") as mock_show:
        p.curves(metrics=["loss"])  # aggregate loss without task
        p.curves(metrics=["loss", "accuracy"], task="Class")
        p.curves(metrics=["loss", "mae"], task="AlogP")
        p.regression(task="AlogP", savepath=str(tmp_path / "mt_reg.png"))
        p.residuals(train_data=tr_g, task="PSA")
        p.confusion_matrix(task="Class")
        p.classification_report(task="Class")
        p.roc_curve(task="Class")
        p.roc_curve(model=[m1, m2], model_names=["GINE", "GCN"], color=["blue", "green"], task="Class")
        p.pr_curve(task="Class", chance_level=True)
        p.pr_curve(model=[m1, m2], model_names=["GINE", "GCN"], color=["blue", "green"], task="Class")
        assert mock_show.call_count == 11
    plt.close("all")

    # Guardrails: missing task or wrong task type raises clear ValueError
    with pytest.raises(ValueError, match="Please specify task"):
        p.regression()
    with pytest.raises(ValueError, match="expected 'regression'"):
        p.regression(task="Class")
    with pytest.raises(ValueError, match="expected 'classification'"):
        p.roc_curve(task="AlogP")
    with pytest.raises(ValueError, match="Unknown task"):
        p.curves(metrics=["loss"], task="DoesNotExist")
