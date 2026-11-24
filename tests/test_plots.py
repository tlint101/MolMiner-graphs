import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.loader import DataLoader

from mminer_graphs.model.graph_modeling import GCNNet, MLP
from mminer_graphs.plot.plot import Plots


def test_plots_init_numpy_pandas_auto_device_and_float_labels(bace_tabular):
    mlp_r = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None)
    mlp_r.fit(
        bace_tabular["X_train"], bace_tabular["y_reg_train"],
        bace_tabular["X_test"], bace_tabular["y_reg_test"],
        epochs=2, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        p = Plots(
            model=mlp_r,
            x_data=pd.DataFrame(bace_tabular["X_test"]),
            y_data=pd.Series(bace_tabular["y_reg_test"]),
            device="auto",
        )
        assert len(w) == 0

    _, y_batch = next(iter(p.data))
    assert y_batch.dtype == torch.float32
    assert np.allclose(y_batch.cpu().numpy(), bace_tabular["y_reg_test"], atol=1e-6)


def test_plots_curves_regression_metrics(bace_tabular):
    mlp_r = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None)
    mlp_r.fit(
        bace_tabular["X_train"], bace_tabular["y_reg_train"],
        bace_tabular["X_test"], bace_tabular["y_reg_test"],
        epochs=3, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False,
    )
    p = Plots(model=mlp_r, x_data=bace_tabular["X_test"], y_data=bace_tabular["y_reg_test"], device="cpu")
    with patch.object(plt, "show") as mock_show:
        p.curves(metrics=["mae"])
        p.curves(metrics=["mae", "rmse"])
        p.curves(metrics=["r2"])
        assert mock_show.call_count == 3
    plt.close("all")

    with pytest.raises(ValueError, match="No recorded history"):
        p.curves(metrics=["accuracy"])


def test_plots_regression_and_residuals_on_loaded_model(tmp_path, bace_tabular):
    mlp_r = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None)
    mlp_r.fit(
        bace_tabular["X_train"], bace_tabular["y_reg_train"],
        bace_tabular["X_test"], bace_tabular["y_reg_test"],
        epochs=2, batch_size=10, criterion="mse", device="cpu", pbar=False, device_warn=False,
    )
    wpath = tmp_path / "mlp_reg.pth"
    imgpath = tmp_path / "reg.png"
    mlp_r.save(str(wpath))

    loaded = MLP(input_size=8, hidden_size=8, layers=2, num_classes=None).load(str(wpath), device="cpu")
    assert loaded.metrics.best_epoch is None

    p = Plots(model=loaded, x_data=bace_tabular["X_test"], y_data=bace_tabular["y_reg_test"], device="cpu")
    with patch.object(plt, "show") as mock_show:
        p.regression(savepath=str(imgpath))
        p.residuals()
        assert mock_show.call_count == 2
        assert imgpath.exists()
    plt.close("all")


def test_plots_tabular_multibatch_cm_cr_and_cuda_cutoff(bace_tabular):
    # Use 30 rows with batch_size=8 -> 4 batches to verify all batches are kept (not just last batch)
    X = bace_tabular["X_train"]
    y = bace_tabular["y_cls_train"]
    mlp_c = MLP(input_size=8, hidden_size=8, layers=2, num_classes=2)
    mlp_c.fit(X, y, bace_tabular["X_test"], bace_tabular["y_cls_test"],
              epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)

    p = Plots(model=mlp_c, x_data=X, y_data=y, batch_size=8, device="cpu")
    with patch.object(plt, "show"):
        p.confusion_matrix()
        p.classification_report()
    plt.close("all")

    # Simulated CUDA device with float cutoff=0.5 must not crash calling .detach() on a list
    p_cuda = Plots(model=mlp_c, x_data=X, y_data=y, batch_size=8, device="cpu")
    p_cuda.device = "cuda"
    with patch.object(mlp_c, "predict", return_value=np.linspace(0.0, 1.0, len(y))):
        yt, yp = p_cuda._get_prediction_labels(None, y.tolist(), 0.5, X, mlp_c, return_type="tensor")
        assert isinstance(yp, np.ndarray) and len(yt) == len(yp) == len(y)


def test_plots_pr_curve_multi_model_and_single_show_on_savepath(tmp_path, bace_cls_graphs):
    tr_g, te_g = bace_cls_graphs
    gnn1 = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    gnn1.fit(tr_g, te_g, epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)
    gnn2 = GCNNet(node_dim=142, hidden_dim=16, graph_layers=2, mlp_layers=1, num_classes=2)
    gnn2.fit(tr_g, te_g, epochs=2, batch_size=10, device="cpu", pbar=False, device_warn=False)

    te_loader = DataLoader(te_g, batch_size=10)
    pg = Plots(model=gnn1, data_loader=te_loader, device="cpu")

    # Verify multi-model pr_curve plots exact (recall, precision) & AP, NOT ROC
    labels = [g.y.item() for g in te_g]
    expected_probs = gnn1.predict(te_loader, binary=False, device="cpu", return_type="list")
    expected_prec, expected_rec, _ = precision_recall_curve(labels, expected_probs)
    expected_ap = average_precision_score(labels, expected_probs)

    plotted = []
    orig_plot = plt.plot

    def spy_plot(*args, **kwargs):
        plotted.append((args, kwargs))
        return orig_plot(*args, **kwargs)

    with patch.object(plt, "plot", side_effect=spy_plot), patch.object(plt, "show") as mock_show:
        pg.pr_curve(
            model=[gnn1, gnn2],
            model_names=["GNN1", "GNN2"],
            color=["blue", "green"],
            savepath=str(tmp_path / "pr.png"),
        )
        pg.roc_curve(savepath=str(tmp_path / "roc.png"))
        pg.classification_report(savepath=str(tmp_path / "cr.png"))
        assert mock_show.call_count == 3  # exactly 1 plt.show() per plot when savepath is set
    plt.close("all")

    assert np.allclose(plotted[0][0][0], expected_rec)
    assert np.allclose(plotted[0][0][1], expected_prec)
    assert f"{expected_ap:.3f}" in plotted[0][1]["label"]
