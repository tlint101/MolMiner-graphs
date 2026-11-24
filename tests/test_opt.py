import optuna
import pytest

from mminer_graphs import GATNet, GCNNet, GINENet, GINNet, GSAGENet, MultiTaskGNN
from mminer_graphs.hyperparams.torch_opt import ENCODER_PARAM_KEYS, MODEL_PARAM_KEYS, Torch_opt
from mminer_graphs.model import GAT, GCN, GIN, GINE, GSAGE
from mminer_graphs.model.model_utils import mol_to_graph

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("model_cls", [GCNNet, GINNet, GINENet, GSAGENet, GATNet])
def test_torch_opt_all_gnn_classes_and_dropouts(bace_df, model_cls):
    assert "graph_dropout" in MODEL_PARAM_KEYS[model_cls]
    assert "mlp_dropout" in MODEL_PARAM_KEYS[model_cls]

    topt = Torch_opt(model=model_cls, direction="maximize")
    seen_models = []
    orig_get_score = topt._get_score

    def spy_get_score(model, *args, **kwargs):
        seen_models.append(model)
        return orig_get_score(model, *args, **kwargs)

    topt._get_score = spy_get_score
    study = topt.opt_model(
        n_trials=1, dataset=bace_df, smi_col="smiles", activity_col="Class", num_classes=2,
        graph_layers=2, mlp_layers=2, hidden_dims=(16, 16),
        graph_dropout=(0.25, 0.25), mlp_dropout=(0.35, 0.35),
        epochs=(2, 2), batch_size=(10, 10), optimizer=["adam"],
        lr=(1e-3, 1e-3), wt_decay=(1e-5, 1e-5), criterion="cross_entropy",
        device="cpu", cv=2, progress=False, n_jobs=1, seed=42,
    )
    assert len(seen_models) == 1
    assert abs(seen_models[0].graph_dropout - 0.25) < 1e-6
    if len(seen_models[0].mlp) > 0:
        assert abs(seen_models[0].mlp[0][3].p - 0.35) < 1e-6
    assert 0.0 <= study.best_value <= 1.0


def test_torch_opt_regression_non_atom_graphs_and_build(bace_df):
    # Maximize 1 / (1 + loss) on JunctionTreeGraph with GINENet (auto node_dim=139, edge_dim=6)
    topt = Torch_opt(model=GINENet, direction="maximize")
    study = topt.opt_model(
        n_trials=1, dataset=bace_df, smi_col="smiles", activity_col="pIC50", num_classes=None,
        graph_layers=2, mlp_layers=1, hidden_dims=(16, 16),
        graph_dropout=(0.1, 0.1), mlp_dropout=(0.1, 0.1),
        epochs=(2, 2), batch_size=(10, 10), optimizer=["adam"],
        lr=(1e-3, 1e-3), wt_decay=(1e-5, 1e-5), criterion="mse",
        device="cpu", cv=2, progress=False, n_jobs=1, seed=42,
        graph_type="JunctionTreeGraph",
    )
    assert 0.0 < study.best_value <= 1.0

    best_params = dict(study.best_params)
    snapshot = dict(best_params)
    graphs = [mol_to_graph(s, float(y), graph_type="JunctionTreeGraph") for s, y in zip(bace_df["smiles"], bace_df["pIC50"])]
    built = topt.build_opt_model(
        model=GINENet, train_data=graphs[:30], test_data=graphs[30:], params=best_params,
        node_dim=139, edge_feat_dim=6, num_classes=None, criterion="mse", device="cpu", estop=5,
    )
    assert built.task == "Regression"
    assert best_params == snapshot

    # Minimize raw loss on PharmacophoreGraph with huber criterion
    topt_min = Torch_opt(model=GCNNet, direction="minimize")
    study_min = topt_min.opt_model(
        n_trials=1, dataset=bace_df, smi_col="smiles", activity_col="pIC50", num_classes=None,
        graph_layers=1, mlp_layers=1, hidden_dims=(16, 16),
        graph_dropout=(0.1, 0.1), mlp_dropout=(0.1, 0.1),
        epochs=(2, 2), batch_size=(10, 10), optimizer=["adam"],
        lr=(1e-3, 1e-3), wt_decay=(1e-5, 1e-5), criterion="huber",
        device="cpu", cv=2, progress=False, n_jobs=1, seed=42,
        graph_type="PharmacophoreGraph",
    )
    assert study_min.best_value >= 0.0


@pytest.mark.parametrize("enc_cls", [GCN, GIN, GINE, GSAGE, GAT])
def test_torch_opt_multitask_all_encoders_and_build(tmp_path, bace_df, enc_cls):
    assert enc_cls in ENCODER_PARAM_KEYS
    mixed_tasks = {
        "Class": {"type": "classification", "num_classes": 2},
        "AlogP": {"type": "regression"},
        "PSA": {"type": "regression", "criterion": "huber"},
    }
    topt = Torch_opt(model=MultiTaskGNN, direction="maximize")
    extra_opt = {"heads": (1, 2)} if enc_cls is GAT else {}
    study = topt.opt_model(
        encoder_cls=enc_cls,
        tasks=mixed_tasks,
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
        device="cpu",
        cv=2,
        progress=False,
        n_jobs=1,
        seed=42,
        **extra_opt,
    )
    assert 0.0 < study.best_value <= 1.0
    assert {"shared_mlp_layers", "head_mlp_layers", "loss_weighting"}.issubset(study.best_params)

    # Save best params JSON and build + fit the optimized MultiTaskGNN
    if enc_cls is GINE:
        json_path = tmp_path / "mt_best.json"
        best_params = topt.best_solutions(study, savepath=str(json_path))
        snapshot = dict(best_params)
        y_mat = bace_df[["Class", "AlogP", "PSA"]].to_numpy(dtype=float)
        graphs = [mol_to_graph(s, row, graph_type="AtomGraph") for s, row in zip(bace_df["smiles"], y_mat)]
        built = topt.build_opt_model(
            model=MultiTaskGNN,
            encoder_cls=GINE,
            tasks=mixed_tasks,
            train_data=graphs[:30],
            test_data=graphs[30:],
            params=str(json_path),
            device="cpu",
            estop=5,
        )
        assert isinstance(built, MultiTaskGNN) and built.is_fitted is True
        assert best_params == snapshot
        preds = built.predict(graphs[30:], batch_size=10, device="cpu", return_type="dataframe")
        assert list(preds.columns) == ["Class", "AlogP", "PSA"]


def test_torch_opt_multitask_primary_task_and_guards(bace_df):
    mixed_tasks = {
        "Class": {"type": "classification", "num_classes": 2},
        "AlogP": {"type": "regression"},
        "PSA": {"type": "regression"},
    }
    # Primary classification task with accuracy+loss blend (activity_col omitted -> auto-inferred from tasks)
    topt = Torch_opt(model=MultiTaskGNN, precision_weight=0.2, loss_weight=0.5, direction="maximize")
    study = topt.opt_model(
        encoder_cls=GCN,
        tasks=mixed_tasks,
        primary_task="Class",
        n_trials=1,
        dataset=bace_df,
        smi_col="smiles",
        graph_layers=2,
        hidden_dims=(16, 16),
        graph_dropout=(0.0, 0.0),
        mlp_dropout=(0.0, 0.0),
        shared_mlp_layers=1,
        head_mlp_layers=1,
        loss_weighting="uncertainty",
        epochs=(2, 2),
        batch_size=(10, 10),
        optimizer=["adam"],
        lr=(1e-3, 1e-3),
        wt_decay=(1e-5, 1e-5),
        device="cpu",
        cv=2,
        progress=False,
        n_jobs=1,
        seed=42,
    )
    assert 0.0 <= study.best_value <= 1.0
    assert study.best_params["loss_weighting"] == "uncertainty"

    with pytest.raises(ValueError, match="Unknown primary_task"):
        topt.opt_model(encoder_cls=GCN, tasks=mixed_tasks, primary_task="Nope", dataset=bace_df)
    with pytest.raises(ValueError, match="encoder_cls"):
        topt.opt_model(encoder_cls=None, tasks=mixed_tasks, dataset=bace_df)
