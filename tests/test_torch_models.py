import warnings
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch_geometric.loader import DataLoader

from mminer_graphs.model.graph_modeling import (
    EarlyStopping,
    GATNet,
    GCNNet,
    GINENet,
    GINNet,
    GSAGENet,
    MLP,
    test_c as eval_cls,
    test_r as eval_reg,
)
from mminer_graphs.model.latent_space import GAT, GCN, GINE, GSAGE
from mminer_graphs.model.model_utils import mol_to_graph, set_seed


@pytest.mark.parametrize(
    "model_cls,extra_kwargs",
    [
        (GCNNet, {"reduce_graph": False}),
        (GCNNet, {"reduce_graph": True}),
        (GINNet, {"reduce_graph": True}),
        (GINENet, {"edge_feat_dim": 16, "reduce_graph": True}),
        (GSAGENet, {"reduce_graph": True}),
        (GATNet, {"edge_feat_dim": 16, "heads": 2, "reduce_graph": True}),
    ],
)
def test_all_gnns_classification_and_regression(bace_cls_graphs, bace_reg_graphs, model_cls, extra_kwargs):
    tr_c, te_c = bace_cls_graphs
    clf = model_cls(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=2, num_classes=2, **extra_kwargs)
    assert not hasattr(clf, "train_loss")
    clf.fit(tr_c, te_c, epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    preds = clf.predict(te_c, device="cpu", binary=True, return_type="array")
    assert preds.shape == (len(te_c),)

    tr_r, te_r = bace_reg_graphs
    reg = model_cls(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None, **extra_kwargs)
    reg.fit(tr_r, te_r, epochs=2, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False)
    rpreds = reg.predict(te_r, device="cpu", return_type="array")
    assert rpreds.shape[0] == len(te_r)
    assert isinstance(reg.y_mean, float) and isinstance(reg.y_std, float)


def test_gnn_with_add_feats(bace_df):
    feats = bace_df[["MW", "AlogP", "PSA"]].to_numpy(dtype=np.float32)
    graphs = [
        mol_to_graph(smi, int(y), graph_type="AtomGraph", add_feats=f)
        for smi, y, f in zip(bace_df["smiles"], bace_df["Class"], feats)
    ]
    model = GINENet(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, mlp_layers=2,
                    num_classes=2, add_feats=3)
    model.fit(graphs[:30], graphs[30:], epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    preds = model.predict(graphs[30:], device="cpu", add_feats=3, return_type="list")
    assert len(preds) == 10


def test_multiclass_classification_and_predict_modes(bace_df, bace_tabular):
    # 3-class GNN (bin pIC50 into 3 quantile classes)
    y3 = pd.qcut(bace_df["pIC50"], q=3, labels=[0, 1, 2]).astype(int).to_numpy()
    graphs = [mol_to_graph(s, int(y)) for s, y in zip(bace_df["smiles"], y3)]
    gnn = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=3)
    gnn.fit(graphs[:30], graphs[30:], epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)

    multi_preds = gnn.predict(graphs[30:], device="cpu", binary="multi", return_type="array")
    proba_preds = gnn.predict(graphs[30:], device="cpu", binary="proba", return_type="array")
    assert multi_preds.shape == (10,) and set(multi_preds.tolist()).issubset({0, 1, 2})
    assert proba_preds.shape == (10, 3) and np.allclose(proba_preds.sum(axis=1), 1.0, atol=1e-5)

    # 3-class MLP
    mlp = MLP(input_size=8, hidden_size=16, layers=2, num_classes=3)
    mlp.fit(bace_tabular["X_train"], y3[:30], bace_tabular["X_test"], y3[30:],
            epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    mlp_multi = mlp.predict(bace_tabular["X_test"], device="cpu", binary="multi", return_type="array")
    assert mlp_multi.shape == (10,) and set(mlp_multi.tolist()).issubset({0, 1, 2})


def test_regression_rmse_and_classification_roc_auc(bace_cls_graphs, bace_reg_graphs):
    # Regression RMSE must equal sqrt(MSE)
    tr_r, te_r = bace_reg_graphs
    reg = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    reg.fit(tr_r, te_r, epochs=3, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False)
    preds = reg.predict(te_r, device="cpu", return_type="array").ravel()
    labels = np.array([g.y.item() for g in te_r])
    expected_rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    assert abs(reg.metrics.test_rmse[reg.metrics.best_epoch] - expected_rmse) < 1e-5

    # Classification ROC-AUC must use softmax probabilities, not hard classes
    tr_c, te_c = bace_cls_graphs
    clf = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    clf.fit(tr_c, te_c, epochs=3, batch_size=10, device="cpu", pbar=False, device_warn=False)
    probs = clf.predict(te_c, device="cpu", binary=False, return_type="array")
    y_true = np.array([g.y.item() for g in te_c])
    expected_auc = float(roc_auc_score(y_true, probs))
    assert abs(clf.metrics.roc_auc[clf.metrics.best_epoch] - expected_auc) < 1e-5


def test_restore_best_and_metrics_reset(bace_reg_graphs, bace_cls_graphs):
    tr_r, te_r = bace_reg_graphs
    set_seed(0)
    reg = GCNNet(node_dim=142, hidden_dim=32, graph_layers=2, mlp_layers=2, num_classes=None)
    reg.fit(tr_r, te_r, epochs=20, batch_size=10, lr=0.05, criterion="mse", device="cpu",
            estop=EarlyStopping(monitor="val_loss", patience=4, mode="min"),
            pbar=False, device_warn=False, restore_best=True)
    te_loader = DataLoader(te_r, batch_size=10)
    eval_loss, eval_mae, _, _ = eval_reg(reg, te_loader, reg.criterion, torch.device("cpu"), reg.y_mean, reg.y_std)
    best_ep = reg.metrics.best_epoch
    assert abs(eval_loss - reg.metrics.test_loss[best_ep]) < 1e-5
    assert abs(eval_mae - reg.metrics.test_mae[best_ep]) < 1e-5

    # Re-fitting resets self.metrics and restore_best=False keeps final epoch
    reg.fit(tr_r, te_r, epochs=4, batch_size=10, lr=0.05, criterion="mse", device="cpu",
            pbar=False, device_warn=False, restore_best=False)
    assert len(reg.metrics.test_loss) == 4
    last_loss, _, _, _ = eval_reg(reg, te_loader, reg.criterion, torch.device("cpu"), reg.y_mean, reg.y_std)
    assert abs(last_loss - reg.metrics.test_loss[-1]) < 1e-5

    # EarlyStopping(monitor='val_acc', mode='max') aligns best_epoch and weights
    tr_c, te_c = bace_cls_graphs
    clf = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    clf.fit(tr_c, te_c, epochs=15, batch_size=10, lr=0.05, device="cpu",
            estop=EarlyStopping(monitor="val_acc", patience=4, mode="max"),
            pbar=False, device_warn=False, restore_best=True)
    _, eval_acc, *_ = eval_cls(clf, DataLoader(te_c, batch_size=10), clf.criterion, torch.device("cpu"))
    assert clf.metrics.best_epoch == int(np.argmax(clf.metrics.test_accuracy))
    assert abs(eval_acc - clf.metrics.test_accuracy[clf.metrics.best_epoch]) < 1e-6


def test_reset_params_resets_all_layers(bace_reg_graphs):
    tr_r, te_r = bace_reg_graphs
    model = GINENet(node_dim=142, edge_feat_dim=16, hidden_dim=16, graph_layers=2, mlp_layers=2, num_classes=None)
    model.fit(tr_r, te_r, epochs=3, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False)
    assert model.is_fitted is True and model.y_mean is not None

    trained_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    model.reset_params()
    assert model.is_fitted is False and model.y_mean is None and model.y_std is None
    # Conv weights and linear weights must change; BatchNorm running_mean resets to 0
    changed = [
        k for k, v in model.named_parameters()
        if "weight" in k and not torch.allclose(v, trained_params[k])
    ]
    assert len(changed) >= 3
    assert torch.allclose(model.normalization[0].running_mean, torch.zeros_like(model.normalization[0].running_mean))


def test_save_load_roundtrip_and_mismatch_guards(tmp_path, bace_reg_graphs, bace_cls_graphs, bace_tabular):
    # GNN Regression round-trip + mismatch guard
    tr_r, te_r = bace_reg_graphs
    gnn_r = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    gnn_r.fit(tr_r, te_r, epochs=2, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False)
    expected = gnn_r.predict(te_r, device="cpu", return_type="array")

    reg_path = tmp_path / "gnn_reg.pth"
    gnn_r.save(str(reg_path))
    gnn_r2 = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=None)
    gnn_r2.load(str(reg_path), device="cpu")
    assert np.allclose(expected, gnn_r2.predict(te_r, device="cpu", return_type="array"), atol=1e-6)

    # Loading regression weights into classification model raises ValueError
    gnn_c_wrong = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    with pytest.raises(ValueError, match="Mismatch"):
        gnn_c_wrong.load(str(reg_path), device="cpu")

    # GNN Classification checkpoint into Regression model raises ValueError
    tr_c, te_c = bace_cls_graphs
    gnn_c = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    gnn_c.fit(tr_c, te_c, epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    cls_path = tmp_path / "gnn_cls.pth"
    gnn_c.save(str(cls_path))
    with pytest.raises(ValueError, match="Mismatch"):
        gnn_r2.load(str(cls_path), device="cpu")

    # MLP Regression round-trip
    mlp_r = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None)
    mlp_r.fit(bace_tabular["X_train"], bace_tabular["y_reg_train"],
              bace_tabular["X_test"], bace_tabular["y_reg_test"],
              epochs=2, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False)
    mlp_expected = mlp_r.predict(bace_tabular["X_test"], device="cpu", return_type="array")
    mlp_path = tmp_path / "mlp_reg.pth"
    mlp_r.save(str(mlp_path))
    mlp_r2 = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None).load(str(mlp_path), device="cpu")
    assert np.allclose(mlp_expected, mlp_r2.predict(bace_tabular["X_test"], device="cpu", return_type="array"), atol=1e-6)


def test_mlp_mixed_inputs(bace_tabular):
    x_tr_list = [row for row in bace_tabular["X_train"]]
    y_tr_series = pd.Series(bace_tabular["y_cls_train"], index=range(100, 130))
    x_te_df = pd.DataFrame(bace_tabular["X_test"])
    y_te_list = bace_tabular["y_cls_test"].tolist()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        mlp = MLP(input_size=8, hidden_size=8, layers=2, num_classes=2)
        mlp.fit(x_tr_list, y_tr_series, x_te_df, y_te_list, epochs=2, batch_size=10,
                device="cpu", pbar=False, device_warn=False)
        p_df = mlp.predict(x_te_df, device="cpu", return_type="array")
        p_t = mlp.predict(torch.tensor(bace_tabular["X_test"], dtype=torch.float64), device="cpu", return_type="array")
        assert np.array_equal(p_df, p_t)
        slow_warns = [x for x in w if "list of numpy.ndarrays" in str(x.message)]
        assert len(slow_warns) == 0


@pytest.mark.parametrize(
    "enc_cls,extra_kwargs,expected_dim",
    [
        (GCN, {"reduce_graph": True}, 8),
        (GINE, {"edge_feat_dim": 16, "reduce_graph": True}, 8),
        (GSAGE, {"reduce_graph": True}, 8),
        (GAT, {"edge_feat_dim": 16, "heads": 2, "reduce_graph": True}, 16),
    ],
)
def test_latent_space_encoders(bace_cls_graphs, enc_cls, extra_kwargs, expected_dim):
    tr_g, te_g = bace_cls_graphs
    enc = enc_cls(node_dim=142, hidden_dim=16, graph_layers=2, **extra_kwargs)
    assert not hasattr(enc, "train_loss")
    tx, ty, vx, vy = enc.split_graph_space(tr_g, te_g, batch_size=10, device="cpu")
    sp = enc.extract_graph_space(te_g, batch_size=10, device="cpu")
    assert tx.shape == (30, expected_dim) and vx.shape == (10, expected_dim)
    assert sp.shape == (10, expected_dim) and len(ty) == 30 and len(vy) == 10
