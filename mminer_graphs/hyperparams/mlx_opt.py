"""
Script to run sklearn and xgb hyperparameter optimizations.
"""
import pandas as pd
import numpy as np
import json
import math
import optuna
from optuna.samplers import TPESampler
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mminer_graphs.mlx_models import CrossVal, GINENet, GCNNet, EarlyStopping
from mminer_graphs.mlx_models.mlx_utils import set_seed
from mminer_graphs.hyperparams.opt_plot import Optimization
from typing import Optional, Union, Callable, Iterable, Type

# params when calling the neural network
MODEL_PARAM_KEYS = {
    GINENet: {"graph_layers", "mlp_layers", "hidden_dim", "node_dim", "edge_feat_dim", "num_classes", "add_feats"},
    GCNNet: {"graph_layers", "mlp_layers", "hidden_dim", "node_dim", "num_classes", "add_feats"}
}


class MLX_opt(Optimization):
    def __init__(self, model: Optional[Type[mlx.nn.Module]] = None, precision_weight: float = 0,
                 loss_weight: float = 1.0,
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

    def opt_model(self, model: Optional[Type[mlx.nn.Module]] = None, estop: Optional[int] = 100,
                  n_trials: int = 5, dataset: pd.DataFrame = None, smi_col: str = "smiles",
                  activity_col: str = "activity", num_classes: int = 2, add_feats: Optional[np.array] = None,
                  graph_type: str = "AtomGraph", node_dim: int = 142, edge_dim: int = 16, graph_layers: int = 2,
                  mlp_layers: int = 2, hidden_dims: tuple = (32, 128), graph_dropout: tuple = (0, 0.5),
                  mlp_dropout: tuple = (0, 0.5), epochs: tuple = (50, 100), batch_size: tuple = (16, 128),
                  optimizer: Optional[np.array] = None, lr: tuple = (1e-5, 1e-1), wt_decay: tuple = (1e-5, 1e-4),
                  criterion: str = "cross_entropy", device: str = 'auto', cv: int = 2,
                  callbacks: Optional[Iterable[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]] = None,
                  timeout: int = None, progress: bool = True, n_jobs: int = -1, seed: int = 42, verbose: bool = False):
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
        :param graph_type: str
            Set the graph type. Needed if hte dataset is given as a list and not converted to Dataloader.
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
        :param verbose: bool
            Whether to print out training results for each epoch
        :return:
        """

        if model is None:
            model = self.model

        if optimizer is None:
            optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax"]

        # set seed
        sampler = TPESampler(seed=seed)

        # set study
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        # set trials
        study.optimize(
            lambda trial: self._trial_space(graph_type, node_dim, edge_dim, trial, model, dataset, smi_col,
                                            activity_col, graph_layers, mlp_layers, hidden_dims, graph_dropout,
                                            mlp_dropout, num_classes, add_feats, epochs, batch_size, optimizer, lr,
                                            wt_decay, device, cv, estop, verbose),
            n_jobs=n_jobs,
            n_trials=n_trials, timeout=timeout, show_progress_bar=progress, callbacks=callbacks)

        return study

    def build_opt_model(self, model: Optional[Type[mlx.nn.Module]] = None, train_data: Union[np.array, list] = None,
                        test_data: Union[np.array, list] = None, params: Union[dict, str] = None,
                        add_feats: Optional[np.ndarray] = None, node_dim: int = 142, edge_feat_dim: int = 16,
                        num_classes: int = 2, criterion: str = "cross_entropy", device: str = "auto",
                        estop: Optional[Union[EarlyStopping, int]] = 100, seed=42):
        """
        Create an optimized model from the best parameters.
        :param model: Optional[Type[nn.Module]]
            The neural network class to optimize.
        :param train_data: Union[np.array, list]
            Input a list of the dataset. This will be internally converted into a torch_geometric.loader.Dataloader. If
            a list of additional features is given in add_feats param, this will be appended to the Dataloader.
        :param test_data:
            Input a list of the dataset. This will be internally converted into a torch_geometric.loader.Dataloader. If
            a list of additional features is given in add_feats param, this will be appended to the Dataloader.
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
        elif isinstance(model, type) and issubclass(model, mlx.nn.Module):
            pass
        else:
            raise ValueError("Issue with the model. Check the type. It must be 'Type[nn.Module]'!")

        # set params
        if not isinstance(params, (dict, str)):
            raise ValueError("Parameters must be given as a dictionary or a string to a json file!")
        elif isinstance(params, str):
            with open(params, 'r') as f:
                params = json.load(f)
        elif params is None:
            raise ValueError("Parameters cannot be None!")
        else:
            pass

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

        # separate params for model and fitting
        if model == GCNNet:
            print("Building Optimized Model GCNNet model...")
            for_model_params = ["node_dim", "hidden_dim", "graph_layers", "mlp_layers", "graph_dropout", "mlp_dropout",
                                "add_feats"]
        elif model == GINENet:
            print("Building Optimized Model GINENet model...")
            for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "graph_layers", "mlp_layers",
                                "graph_dropout", "mlp_dropout", "add_feats"]
        else:
            raise ValueError("Issue with the input model param!")
        for_fit_params = ["epochs", "optimizer", "criterion", "lr", "wt_decay", "device", "estop"]

        # add data to Dataloader
        batch_size = params.get("batch_size")
        train_loader = Dataloader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = Dataloader(test_data, batch_size=batch_size)

        model_params = {param_key: params[param_key] for param_key in for_model_params}
        # if additional feats, add them to model_params
        if add_feats is not None:
            model_params['add_feats'] = np.array(add_feats, dtype=np.float32)
        fit_params = {param_key: params[param_key] for param_key in for_fit_params}

        # build model
        build_model = model(**model_params)

        # if model is fitted, reset parameters
        if hasattr(build_model, 'is_fitted') and build_model.is_fitted == True:
            build_model.reset_params()

        # add additional params ot fit_params
        fit_params['model'] = build_model
        fit_params['train_loader'] = train_loader
        fit_params['test_loader'] = test_loader

        build_model.fit(**fit_params)

        return build_model

    def build_model(self, model: Optional[Type[mlx.nn.Module]] = None, model_path: str = None,
                    params: Union[dict, str] = None, add_feats: Optional[np.ndarray] = None, node_dim: int = 142,
                    edge_feat_dim: int = 16, num_classes: int = 2, criterion: str = "cross_entropy",
                    device: str = "auto"):
        """
        Build model from json params and saved model weights.
        Create an optimized model from the best parameters.
        :param model: Optional[Type[nn.Module]]
            The neural network class to optimize.
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
        :return:
        """
        # check model instance variable
        if model is None:
            model = self.model
        elif isinstance(model, type) and issubclass(model, mlx.nn.Module):
            pass
        else:
            raise ValueError("Issue with the model. Check the type. It must be 'Type[nn.Module]'!")

        # set params
        if not isinstance(params, (dict, str)):
            raise ValueError("Parameters must be given as a dictionary or a string to a json file!")
        elif isinstance(params, str):
            with open(params, 'r') as f:
                params = json.load(f)
        elif params is None:
            raise ValueError("Parameters cannot be None!")
        else:
            pass

        # additional params not found optimization
        params['node_dim'] = node_dim
        params['edge_feat_dim'] = edge_feat_dim
        params['num_classes'] = num_classes
        params['add_feats'] = add_feats
        params['criterion'] = criterion
        params['device'] = device

        # separate params for model and fitting
        if model == GCNNet:
            print("Building GCNNet model...")
            for_model_params = ["node_dim", "hidden_dim", "graph_layers", "mlp_layers", "graph_dropout", "mlp_dropout",
                                "add_feats"]
        elif model == GINENet:
            print("Building GINENet model...")
            for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "graph_layers", "mlp_layers",
                                "graph_dropout", "mlp_dropout", "add_feats"]
        else:
            raise ValueError("Issue with the input model param!")

        model_params = {param_key: params[param_key] for param_key in for_model_params}
        # print(model_params)  # for debugging

        # if additional feats, add them to model_params
        if add_feats is not None:
            model_params['add_feats'] = add_feats

        # build model
        model = model(**model_params)  # load params
        build_model = model.load(model_path)  # load weights

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

    def _trial_space(self, graph_type, node_dim, edge_dim, trial, model, dataset, smi_col, activity_col, graph_layers,
                     mlp_layers, hidden_dims, graph_dropout, mlp_dropout, num_classes, add_feats, epochs, batch_size,
                     optimizer, lr, wt_decay, device, cv, estop, verbose):
        """sample the hyperparameter space"""

        mx.reset_peak_memory()  # clears GPU memory
        mx.clear_cache()  # clears any kernel or graph cache

        """get all possible hyperparameters for model class"""
        total_params = {
            "graph_layers": trial.suggest_int("graph_layers", 1, graph_layers),
            "mlp_layers": trial.suggest_int("mlp_layers", 1, mlp_layers),
            "hidden_dim": trial.suggest_int("hidden_dim", hidden_dims[0], hidden_dims[1], step=16),
            "mlp_dropout": trial.suggest_float("mlp_dropout", mlp_dropout[0], mlp_dropout[1]),
            "num_classes": num_classes,
            "add_feats": add_feats
        }

        # additional params based on model
        if graph_dropout:
            total_params["graph_dropout"] = trial.suggest_float("graph_dropout", graph_dropout[0], graph_dropout[1])

        """get all possible parameters for fitting the model"""
        epoch = trial.suggest_int("epochs", epochs[0], epochs[1], step=25)
        batch_size = trial.suggest_int("batch_size", batch_size[0], batch_size[1], step=16)
        optimizer = trial.suggest_categorical("optimizer", optimizer)
        lr = trial.suggest_float("lr", lr[0], lr[1], log=True)
        wt_decay = trial.suggest_float("wt_decay", wt_decay[0], wt_decay[1], log=True)

        # filter params for specific model
        keys = MODEL_PARAM_KEYS[model]
        filtered_params = {k: v for k, v in total_params.items() if k in keys}
        filtered_params["node_dim"] = node_dim
        if model == "GINENet":
            filtered_params["edge_feat_dim"] = edge_dim
        if add_feats is not None:
            filtered_params["add_feats"] = add_feats

        # # troubleshooting
        # print("THIS SHOULD BE FILTERED", filtered_params)

        # build model with params
        model = model(**filtered_params)
        # # debug
        # from mminer_graphs.mlx_models import summary
        # print(summary(model))

        # set early stopping
        if estop is None:
            estop = 100
        elif not isinstance(estop, int):
            raise ValueError("estop must be an int or None!")
        early_stopping = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        # get model scores
        try:
            score = self._get_score(graph_type, model, dataset, early_stopping, cv, smi_col, activity_col, add_feats,
                                    batch_size, epoch, optimizer, lr, wt_decay, device, verbose)

            # if scores invalid
            if score is None or (isinstance(score, float) and math.isnan(score)):
                print("Invalid score (None or NaN). Skipping trial!!")
                raise optuna.exceptions.TrialPruned()
            return score
        # if batch size contains 1 only, skip
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                print("Skipping trial due to BatchNorm batch size issue.")
                raise optuna.exceptions.TrialPruned()
            raise

    def _get_score(self, graph_type, model, dataset, early_stopping, cv, smi_col, activity_col, add_feats, batch_size,
                   epoch, optimizer, lr, wt_decay, device, verbose):
        """support function to get the model cv scores"""
        cval = CrossVal(model=model, dataset=dataset, estop=early_stopping)
        metrics = cval.get_cv_scores(graph_type=graph_type, cv=cv, smi_col=smi_col, activity_col=activity_col,
                                     add_feats=add_feats, batch_size=batch_size, epochs=epoch, optimizer=optimizer,
                                     lr=lr, wt_decay=wt_decay, device=device, verbose=verbose, pbar=False,
                                     device_warn=False)

        # average metrics
        acc = np.mean([x for sub in metrics['Accuracy'] for x in sub])
        prec = np.mean([x for sub in metrics['Precision'] for x in sub])
        loss = np.mean([x for sub in metrics['Loss'] for x in sub])

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
