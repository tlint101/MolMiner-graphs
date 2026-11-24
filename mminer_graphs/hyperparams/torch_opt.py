"""
Script to run sklearn and xgb hyperparameter optimizations.
"""
import pandas as pd
import numpy as np
import json
import math
import optuna
from optuna.samplers import TPESampler
from torch import nn
from mminer_graphs import CrossVal, GINNet, GINENet, GCNNet, GSAGENet, GATNet, EarlyStopping, MultiTaskGNN
from mminer_graphs.model.latent_space import GCN, GIN, GINE, GSAGE, GAT
from mminer_graphs.model.multitask import _parse_tasks, VALID_LOSS_WEIGHTINGS
from mminer_graphs.model.model_utils import set_seed, seed_worker, make_generator, GRAPH_NODE_DIMS, GRAPH_EDGE_DIMS
from mminer_graphs.hyperparams.opt_plot import Optimization
from typing import Optional, Union, Callable, Iterable, Type
from torch_geometric.loader import DataLoader

# params when calling the neural network
_BASE_MODEL_PARAM_KEYS = {
    "graph_layers", "mlp_layers", "hidden_dim", "node_dim",
    "graph_dropout", "mlp_dropout", "num_classes", "add_feats",
}
MODEL_PARAM_KEYS = {
    GCNNet: _BASE_MODEL_PARAM_KEYS,
    GINNet: _BASE_MODEL_PARAM_KEYS,
    GINENet: _BASE_MODEL_PARAM_KEYS | {"edge_feat_dim"},
    GSAGENet: _BASE_MODEL_PARAM_KEYS,
    GATNet: _BASE_MODEL_PARAM_KEYS | {"edge_feat_dim", "heads"},
}
_BASE_ENCODER_PARAM_KEYS = {"node_dim", "hidden_dim", "graph_layers", "graph_dropout"}
ENCODER_PARAM_KEYS = {
    GCN: _BASE_ENCODER_PARAM_KEYS,
    GIN: _BASE_ENCODER_PARAM_KEYS,
    GINE: _BASE_ENCODER_PARAM_KEYS | {"edge_feat_dim"},
    GSAGE: _BASE_ENCODER_PARAM_KEYS,
    GAT: _BASE_ENCODER_PARAM_KEYS | {"edge_feat_dim", "heads"},
}


def _resolve_int_range(val: Union[int, tuple, list], default_min: int, name: str) -> tuple[int, int]:
    if isinstance(val, (tuple, list)):
        if len(val) != 2:
            raise ValueError(f"{name} tuple must have length 2 (got {val!r}).")
        lo, hi = int(val[0]), int(val[1])
    elif isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        lo, hi = default_min, int(val)
    else:
        raise TypeError(f"{name} must be an int or 2-tuple of ints (got {type(val).__name__}).")
    if lo < default_min or hi < lo:
        raise ValueError(f"{name} range ({lo}, {hi}) must satisfy {default_min} <= lo <= hi.")
    return lo, hi


def _resolve_loss_weighting_choices(loss_weighting: Union[str, list[str], tuple[str, ...]]) -> list[str]:
    if isinstance(loss_weighting, str):
        choices = [loss_weighting]
    elif isinstance(loss_weighting, (list, tuple)) and len(loss_weighting) > 0:
        choices = list(loss_weighting)
    else:
        raise ValueError(
            f"loss_weighting must be a valid string or non-empty list/tuple from {VALID_LOSS_WEIGHTINGS}."
        )
    for lw in choices:
        if lw not in VALID_LOSS_WEIGHTINGS:
            raise ValueError(f"Invalid loss_weighting {lw!r}; expected one of {VALID_LOSS_WEIGHTINGS}.")
    return choices


class Torch_opt(Optimization):
    def __init__(self, model: Optional[Type[nn.Module]] = None, precision_weight: float = 0, loss_weight: float = 1.0,
                 direction: str = "maximize"):
        """

        :param model: Optional[Type[nn.Module]]
            The neural network class to optimize.
        :param precision_weight: float
            Set the weight for precision score. The value should be between 0 and 1. The corresponding weight for
            accuracy will be calculated as 1 - precision_weight, adjusted to the weight of the loss. Precision weight
            defaults to 0.
        :param loss_weight: float
            Set the weight for loss score. The value should be between 0 and 1. This will adjust the weight for the
            model score as (w_acc * val_acc) + (w_prec * val_prec) + (w_loss * n_loss), with the w_acc and w_prec
            adjusted from the loss_weight.
        :param direction: str
            Set the direction of the model scores for each trial.
        """
        super().__init__()
        self.model = model
        self.precision_w = precision_weight
        self.loss_w = loss_weight
        self.direction = direction

    def opt_model(self, model: Optional[Type[nn.Module]] = None, estop: Optional[int] = 100,
                  n_trials: int = 5, dataset: pd.DataFrame = None, smi_col: str = "smiles",
                  activity_col: Optional[Union[str, list[str], tuple[str, ...]]] = "activity",
                  num_classes: Optional[int] = 2, add_feats: Optional[np.array] = None,
                  graph_layers: int = 2, mlp_layers: int = 2, hidden_dims: tuple = (32, 128),
                  graph_dropout: tuple = (0, 0.5), mlp_dropout: tuple = (0, 0.5), epochs: tuple = (50, 100),
                  batch_size: tuple = (16, 128), optimizer: np.array = None, lr: tuple = (1e-5, 1e-1),
                  wt_decay: tuple = (1e-5, 1e-4), criterion: str = "cross_entropy", device: str = 'auto', cv: int = 2,
                  callbacks: Optional[Iterable[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]] = None,
                  timeout: int = None, progress: bool = True, n_jobs: int = -1, seed: int = 42,
                  graph_type: str = "AtomGraph", node_dim: Optional[int] = None, edge_dim: Optional[int] = None,
                  encoder_cls: Optional[Type[nn.Module]] = None,
                  tasks: Optional[Union[list[str], tuple[str, ...], dict]] = None,
                  default_task: str = "classification",
                  shared_mlp_layers: Union[int, tuple] = (0, 2),
                  head_mlp_layers: Union[int, tuple] = (1, 2),
                  loss_weighting: Union[str, list[str], tuple[str, ...]] = ("equal", "uncertainty"),
                  primary_task: Optional[str] = None,
                  heads: Optional[Union[int, tuple]] = None):
        """
        Optimize the model using Optuna.
        :param model: Optional[Type[nn.Module]]
            The neural network class to optimize.
        :param estop: Optional[int]
            Set the patience level for early stopping. If set to None, it will automatically set early stopping to 100.
        :param n_trials: int
            Set the number of trials for searching the model space.
        :param dataset: pd.DataFrame
            The pd.DataFrame containing the labels and cross-validation information. train_indexes and test_indexes will
            be obtained from the input label pd.DataFrame. Format of pd.DataFrame can be found using mol-miner.
        :param smi_col: str
            Column containing the smiles string.
        :param activity_col: str
            Column containing the activity labels.
        :param num_classes: int
            Number of classes for the output layer.
        :param add_feats: Optional[np.array]
            Input the np.array/list of the additional features. This will be appended to teh latent space after the
            graph layers and used in the mlp layers. This is needed to determine the length of the array and modifying
            the dimensions accordingly.
        :param graph_layers: int
            Maximum number of message passing layers.
        :param mlp_layers: int
            Maximum number of mlp layers.
        :param hidden_dims: tuple
            Set the number range of hidden dimensions for each graph layer. Should be a tuple, where the first is
            the lowest, and the last is the maximum.
        :param graph_dropout: tuple
            Set the dropout rates for each graph layer. Should be a tuple, where the first is the lowest, and the last
            is the maximum.
        :param mlp_dropout: tuple
            Set the dropout rates for each mlp layer. Should be a tuple, where the first is the lowest, and the last is
            the maximum.
        :param epochs: tuple
            Set the number of epochs for the neural network. Should be a tuple, where the first is the lowest, and the
             last is the maximum.
        :param batch_size: tuple
            Set the batch size for the train/test loader for the neural network. Should be a tuple, where the first is
            the lowest, and the last is the maximum.
        :param optimizer: np.array
            A list of optimizers for the model. If set to None, the default settings will be used: ["adam", "sgd",
            "rmsprop", "adagrad", "adadelta", "adamax"]
        :param lr: tuple
            Set the learning rate for the neural networks. Should be a tuple, where the first is the lowest, and the
            last is the maximum.
        :param wt_decay: tuple
            Set the weight decay for the neural networks. Should be a tuple, where the first is the lowest, and the last
            is the maximum.
        :param criterion: str
            Set the model loss function.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param cv: int
            The number of cross-validation folds.
        :param callbacks:
            Set the optuna callbacks during model optimization.
        :param timeout: int
            Add a timeout to stop the model. Time must be in seconds. If set to None, trial will run until trials finish.
        :param progress: bool:
            Set the progress bar during model optimization.
        :param n_jobs: int
            Select number of threads to use for parallelization. Must be an integer >= 1.
        :param seed: int
            Set random seed for reproducibility.
        :return:
        """

        if model is None:
            model = self.model
        is_multitask = model is MultiTaskGNN
        if not is_multitask and model not in MODEL_PARAM_KEYS:
            raise ValueError(
                f"Unsupported model class '{model}'. Supported: {list(MODEL_PARAM_KEYS.keys()) + [MultiTaskGNN]}"
            )
        if graph_type not in GRAPH_NODE_DIMS:
            raise ValueError(f"Unsupported graph_type '{graph_type}'. Supported: {list(GRAPH_NODE_DIMS.keys())}")
        if node_dim is None:
            node_dim = GRAPH_NODE_DIMS[graph_type]
        if edge_dim is None:
            edge_dim = GRAPH_EDGE_DIMS[graph_type]

        mt_config = None
        if is_multitask:
            if encoder_cls not in ENCODER_PARAM_KEYS:
                raise ValueError(
                    f"MultiTaskGNN optimization requires encoder_cls in {list(ENCODER_PARAM_KEYS.keys())} "
                    f"(got {encoder_cls!r})."
                )
            if tasks is None:
                if isinstance(activity_col, (list, tuple)):
                    tasks = list(activity_col)
                else:
                    raise ValueError("MultiTaskGNN optimization requires `tasks` (dict or list of task names).")
            task_names, _ = _parse_tasks(tasks, default_task=default_task)
            if activity_col is None or (activity_col == "activity" and dataset is not None and "activity" not in dataset.columns):
                activity_col = list(task_names)
            if primary_task is not None and primary_task not in task_names:
                raise ValueError(f"Unknown primary_task {primary_task!r}; available tasks: {task_names}.")
            mt_config = {
                "encoder_cls": encoder_cls,
                "tasks": tasks,
                "default_task": default_task,
                "shared_range": _resolve_int_range(shared_mlp_layers, 0, "shared_mlp_layers"),
                "head_range": _resolve_int_range(head_mlp_layers, 1, "head_mlp_layers"),
                "lw_choices": _resolve_loss_weighting_choices(loss_weighting),
                "primary_task": primary_task,
                "heads": heads,
            }

        if optimizer is None:
            optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax"]

        # set seed
        sampler = TPESampler(seed=seed)

        # set study
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        # set trials
        study.optimize(
            lambda trial: self._trial_space(trial, model, dataset, smi_col, activity_col, graph_layers, mlp_layers,
                                            hidden_dims, graph_dropout, mlp_dropout, num_classes, add_feats, epochs,
                                            batch_size, optimizer, lr, wt_decay, criterion, device, cv, estop,
                                            graph_type, node_dim, edge_dim, mt_config),
            n_jobs=n_jobs, n_trials=n_trials, timeout=timeout, show_progress_bar=progress, callbacks=callbacks)

        return study

    def build_opt_model(self, model: Optional[Type[nn.Module]] = None, train_data: Union[np.array, list, DataLoader] = None,
                        test_data: Union[np.array, list, DataLoader] = None, params: Union[dict, str] = None,
                        add_feats: Optional[np.ndarray] = None, node_dim: int = 142, edge_feat_dim: int = 16,
                        num_classes: int = 2, criterion: str = "cross_entropy", device: str = "auto",
                        estop: Optional[Union[EarlyStopping, int]] = 100, seed=42,
                        encoder_cls: Optional[Type[nn.Module]] = None,
                        tasks: Optional[Union[list[str], tuple[str, ...], dict]] = None,
                        default_task: str = "classification",
                        loss_weighting: str = "equal"):
        """
        Create an optimized model from the best parameters.
        :param model: Optional[Type[nn.Module]]
            The neural network class to optimize.
        :param train_data: Union[np.array, list]
            Input a list of the dataset. This will be internally converted into a torch_geometric.loader.DataLoader. If
            a list of additional features is given in add_feats param, this will be appended to the DataLoader.
        :param test_data:
            Input a list of the dataset. This will be internally converted into a torch_geometric.loader.DataLoader. If
            a list of additional features is given in add_feats param, this will be appended to the DataLoader.
        :param params: Union[dict, str]
            The best parameters obtained from optimization. Must be in dict format. If str given, it must point to a
            json file containing the parameters i na dict format.
        :param add_feats: Optional[np.ndarray]
            A list of additional features to add to the graph latent space. Information needed to be given to set the
            correct number of nodes.
        :param node_dim: int
            Set the node dimension for the graph. Will default to 142.
        :param edge_feat_dim: int
            Set the dimensions of the edge features/attributes. Will default to 16.
        :param num_classes: int
            Set the number of classes for the model. Will default to 2..
        :param criterion: str
            Set the model loss function.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param estop: Optional[Union[EarlyStopping, int]]
            Set the patience level for early stopping. the early stopping will be triggered based on the smallest test
            loss for each epoch. If set to None, it will automatically set early stopping to 100.
        :param seed: int
            Set model seed.
        :return:
        """
        # check model instance variable
        if model is None:
            model = self.model
        elif isinstance(model, type) and issubclass(model, nn.Module):
            pass
        else:
            raise ValueError("Issue with the model. Check the type. It must be 'Type[nn.Module]'!")

        # set params (copy so caller's dict is not mutated)
        if not isinstance(params, (dict, str)):
            raise ValueError("Parameters must be given as a dictionary or a string to a json file!")
        elif isinstance(params, str):
            with open(params, 'r') as f:
                params = json.load(f)
        else:
            params = dict(params)

        # additional params not found optimization
        params['node_dim'] = node_dim
        params['edge_feat_dim'] = edge_feat_dim
        params['num_classes'] = num_classes
        params['add_feats'] = add_feats
        params['criterion'] = criterion
        params['device'] = device

        if isinstance(estop, int):
            params['estop'] = EarlyStopping(monitor='val_loss', patience=estop, mode='min')
        elif isinstance(estop, EarlyStopping):
            params['estop'] = estop

        # add data to DataLoader (if not already a DataLoader)
        batch_size = params.get("batch_size")
        train_loader = (
            train_data if isinstance(train_data, DataLoader)
            else DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            worker_init_fn=seed_worker, generator=make_generator(seed))
        )
        test_loader = (
            test_data if isinstance(test_data, DataLoader)
            else DataLoader(test_data, batch_size=batch_size)
        )

        if model is MultiTaskGNN:
            if encoder_cls not in ENCODER_PARAM_KEYS:
                raise ValueError(
                    f"Building optimized MultiTaskGNN requires encoder_cls in {list(ENCODER_PARAM_KEYS.keys())} "
                    f"(got {encoder_cls!r})."
                )
            if tasks is None:
                raise ValueError("Building optimized MultiTaskGNN requires `tasks`.")
            print(f"Building Optimized MultiTaskGNN ({encoder_cls.__name__} encoder)...")
            enc_keys = ENCODER_PARAM_KEYS[encoder_cls]
            enc_params = {k: params[k] for k in enc_keys if k in params}
            encoder = encoder_cls(**enc_params)
            build_model = MultiTaskGNN(
                encoder=encoder,
                tasks=tasks,
                default_task=default_task,
                add_feats=add_feats,
                shared_mlp_layers=params.get("shared_mlp_layers", 0),
                head_mlp_layers=params.get("head_mlp_layers", 1),
                mlp_dropout=params.get("mlp_dropout", 0.0),
                loss_weighting=params.get("loss_weighting", loss_weighting),
            )
            for_fit_params = ["epochs", "optimizer", "lr", "wt_decay", "device", "estop"]
            fit_params = {k: params[k] for k in for_fit_params if k in params}
            fit_params["train_loader"] = train_loader
            fit_params["test_loader"] = test_loader
            build_model.fit(**fit_params)
            return build_model

        # separate params for model and fitting
        if model not in MODEL_PARAM_KEYS:
            raise ValueError(f"Unsupported model class '{model}'. Supported: {list(MODEL_PARAM_KEYS.keys())}")
        print(f"Building Optimized {model.__name__} model...")
        for_model_params = MODEL_PARAM_KEYS[model]
        for_fit_params = ["epochs", "optimizer", "criterion", "lr", "wt_decay", "device", "estop"]

        model_params = {param_key: params[param_key] for param_key in for_model_params if param_key in params}
        # if additional feats, add them to model_params
        if add_feats is not None:
            model_params['add_feats'] = add_feats
        fit_params = {param_key: params[param_key] for param_key in for_fit_params if param_key in params}

        # build model
        build_model = model(**model_params)

        if hasattr(build_model, 'is_fitted'):
            build_model.reset_params()

        # add additional params ot fit_params
        # fit_params['model'] = build_model
        fit_params['train_loader'] = train_loader
        fit_params['test_loader'] = test_loader

        build_model.fit(**fit_params)

        return build_model

    @staticmethod
    def best_solutions(space: optuna.study, savepath: Optional[str] = None):
        """
        Save solutions as txt file.
        :param space: optuna.study
            The search space obtained from opt_model.
        :param savepath: str
            Save path for best solutions. Will be saved as a json file.
        :return: dict of best solutions
        """
        params = space.best_params

        if savepath:
            import json
            with open(savepath, "w") as f:
                json.dump(params, f, indent=4)

        return params

    def _trial_space(self, trial, model, dataset, smi_col, activity_col, graph_layers, mlp_layers, hidden_dims,
                     graph_dropout, mlp_dropout, num_classes, add_feats, epochs, batch_size, optimizer, lr, wt_decay,
                     criterion, device, cv, estop, graph_type="AtomGraph", node_dim=142, edge_dim=16,
                     mt_config: Optional[dict] = None):
        """sample the hyperparameter space"""

        epoch = trial.suggest_int("epochs", epochs[0], epochs[1], step=25)
        batch_size = trial.suggest_int("batch_size", batch_size[0], batch_size[1], step=16)
        optimizer = trial.suggest_categorical("optimizer", optimizer)
        lr = trial.suggest_float("lr", lr[0], lr[1], log=True)
        wt_decay = trial.suggest_float("wt_decay", wt_decay[0], wt_decay[1], log=True)

        if model is MultiTaskGNN:
            enc_cls = mt_config["encoder_cls"]
            sh_lo, sh_hi = mt_config["shared_range"]
            hd_lo, hd_hi = mt_config["head_range"]
            enc_total = {
                "graph_layers": trial.suggest_int("graph_layers", 1, graph_layers),
                "hidden_dim": trial.suggest_int("hidden_dim", hidden_dims[0], hidden_dims[1], step=16),
                "node_dim": node_dim,
                "edge_feat_dim": edge_dim,
            }
            if graph_dropout:
                enc_total["graph_dropout"] = trial.suggest_float("graph_dropout", graph_dropout[0], graph_dropout[1])
            if enc_cls is GAT and mt_config["heads"] is not None:
                h_val = mt_config["heads"]
                if isinstance(h_val, (tuple, list)):
                    enc_total["heads"] = trial.suggest_int("heads", int(h_val[0]), int(h_val[1]))
                else:
                    enc_total["heads"] = trial.suggest_int("heads", int(h_val), int(h_val))

            enc_keys = ENCODER_PARAM_KEYS[enc_cls]
            encoder = enc_cls(**{k: v for k, v in enc_total.items() if k in enc_keys})

            shared_layers = trial.suggest_int("shared_mlp_layers", sh_lo, sh_hi)
            head_layers = trial.suggest_int("head_mlp_layers", hd_lo, hd_hi)
            m_drop = trial.suggest_float("mlp_dropout", mlp_dropout[0], mlp_dropout[1])
            chosen_lw = trial.suggest_categorical("loss_weighting", mt_config["lw_choices"])

            model_instance = MultiTaskGNN(
                encoder=encoder,
                tasks=mt_config["tasks"],
                default_task=mt_config["default_task"],
                add_feats=add_feats,
                shared_mlp_layers=shared_layers,
                head_mlp_layers=head_layers,
                mlp_dropout=m_drop,
                loss_weighting=chosen_lw,
            )
            primary_task = mt_config["primary_task"]
        else:
            total_params = {
                "graph_layers": trial.suggest_int("graph_layers", 1, graph_layers),
                "mlp_layers": trial.suggest_int("mlp_layers", 1, mlp_layers),
                "hidden_dim": trial.suggest_int("hidden_dim", hidden_dims[0], hidden_dims[1], step=16),
                "mlp_dropout": trial.suggest_float("mlp_dropout", mlp_dropout[0], mlp_dropout[1]),
                "node_dim": node_dim,
                "edge_feat_dim": edge_dim,
                "num_classes": num_classes,
                "add_feats": add_feats
            }

            if graph_dropout:
                total_params["graph_dropout"] = trial.suggest_float("graph_dropout", graph_dropout[0], graph_dropout[1])

            keys = MODEL_PARAM_KEYS[model]
            filtered_params = {k: v for k, v in total_params.items() if k in keys}
            model_instance = model(**filtered_params)
            primary_task = None

        # set early stopping
        if estop is None:
            estop = 100
        elif not isinstance(estop, int):
            raise ValueError("estop must be an int or None!")
        early_stopping = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        # get model scores
        try:
            score = self._get_score(model_instance, dataset, early_stopping, cv, smi_col, activity_col, add_feats,
                                    batch_size, epoch, optimizer, lr, wt_decay, criterion, device, graph_type,
                                    primary_task=primary_task)
            if score is None or (isinstance(score, float) and math.isnan(score)):
                print("Invalid score (None or NaN). Skipping trial!!")
                raise optuna.exceptions.TrialPruned()
            return score
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                print("Skipping trial due to BatchNorm batch size issue.")
                raise optuna.exceptions.TrialPruned()
            raise

    def _get_score(self, model, dataset, early_stopping, cv, smi_col, activity_col, add_feats, batch_size, epoch,
                   optimizer, lr, wt_decay, criterion, device, graph_type="AtomGraph",
                   primary_task: Optional[str] = None):
        """support function to get the model cv scores"""
        cval = CrossVal(model=model, dataset=dataset, estop=early_stopping)
        metrics = cval.get_cv_scores(graph_type=graph_type, cv=cv, smi_col=smi_col, activity_col=activity_col,
                                     add_feats=add_feats, batch_size=batch_size, epochs=epoch, optimizer=optimizer,
                                     lr=lr, wt_decay=wt_decay, criterion=criterion, device=device, pbar=False,
                                     device_warn=False)

        if model.task == "MultiTask":
            if primary_task is not None:
                task_df = metrics.xs(primary_task, level="Task")
                loss = float(task_df["Loss"].mean())
                if model.task_configs[primary_task]["type"] == "classification" and self.loss_w < 1.0:
                    acc = float(task_df["Accuracy"].mean())
                    prec = float(task_df["Precision"].mean())
                    n_loss = 1.0 / (1.0 + loss)
                    assert 0.0 <= self.loss_w <= 1.0
                    assert 0.0 <= self.precision_w <= 1.0
                    t_weight = 1.0 - self.loss_w
                    w_prec = t_weight * self.precision_w
                    w_acc = t_weight - w_prec
                    return (w_acc * acc) + (w_prec * prec) + (self.loss_w * n_loss)
                return (1.0 / (1.0 + loss)) if self.direction == "maximize" else loss

            loss = float(metrics["Loss"].mean())
            return (1.0 / (1.0 + loss)) if self.direction == "maximize" else loss

        loss = float(metrics['Loss'].mean())

        if model.task == "Regression":
            if self.direction == "maximize":
                return 1.0 / (1.0 + loss)
            return loss

        # classification: average metrics
        acc = float(metrics['Accuracy'].mean())
        prec = float(metrics['Precision'].mean())

        # convert loss from 0 to 1 scale
        n_loss = 1 / (1 + loss)

        # ensure weight input ranges from 0 to 1
        assert 0.0 <= self.loss_w <= 1.0
        assert 0.0 <= self.precision_w <= 1.0

        # set weights, must total 1
        t_weight = 1.0 - self.loss_w
        w_prec = t_weight * self.precision_w
        w_acc = t_weight - w_prec

        # Final score (range 0 to 1)
        score = (w_acc * acc) + (w_prec * prec) + (self.loss_w * n_loss)

        return score


if __name__ == "__main__":
    import doctest

    doctest.testmod()
