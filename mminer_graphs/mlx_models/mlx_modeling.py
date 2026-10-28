import inspect
import copy
import json
import random
import pandas as pd
import numpy as np
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mlx_graphs.data import GraphData
from mlx_graphs.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from mlx.utils import tree_flatten, tree_unflatten
from typing import Optional, Union
from tqdm import tqdm
from mminer_graphs.model.model_utils import mol_to_graph
from mminer_graphs.mlx_models.mlx_summary import MLXModelStatistics
from mminer_graphs.mlx_models.mlx_utils import get_device, state_apple_silicon_device, MLX_OPTI, MLX_ACT, \
    MLX_CRITERIONS, MLX_CLASSIFICATION, MLX_REGRESSION, NO_WT_DECAY
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score, \
    mean_squared_error, accuracy_score, precision_score


def get_metrics(model: mlx.nn.Module, test_loader: Dataloader, batch_size: int = None,
                binary: Union[bool, float, str, None] = True, cutoff: float = 0.5, return_type: str = "array",
                add_feats: Optional[Union[np.array, list, int]] = None, pbar: bool = False):
    """
    Support function to quickly get relavent metrics for a fitted model.
    :param model: mlx.nn.Module
        The fitted MLX model.
    :param test_loader: Dataloader
        Data for prediction. Must be converted into a graph and set as a Dataloader
    :param batch_size: int
        Set the batch size for prediction. Only used if the input data is of type list.
    :param binary: Union[bool, float, str]
        Set the output for classification models as binary or multi-class. A boolean value will set predictions to
        be binary or multi-class predictions. Input strings can be "multi" or 'proba', where "multi" will output a
        multi-class labels and "proba" will output a binary probability vector. If 'False' is given, the model will
        output the probability for the final classification label.
    :param cutoff:
        Set the cutoff for binary predictions. This param is only available for binary models and when the binary
        param is set to 'True'.
    :param return_type: bool
        Set the progress bar for predictions.
    :param return_type: str
        Set the return type of the prediction. Can be "array" or "list".
    :param add_feats: Optional[Union[np.array, list, int]]
        The number of additional features to be added. Must be the same as what was given during training.
    :param pbar: bool
         Show progress bar for predictions.
    :return:
    """

    y_pred = model.predict(test_loader, batch_size=batch_size, binary=binary, cutoff=cutoff, return_type=return_type,
                           add_feats=add_feats, pbar=pbar)
    # extract true values
    true_y = mx.concatenate([batch.y for batch in test_loader], axis=0)

    # convert to np array
    y_pred = np.array(y_pred)
    true_y = np.array(true_y)

    if model.type == "Classification":
        acc = accuracy_score(true_y, y_pred)
        prec = precision_score(true_y, y_pred)
        recall = recall_score(true_y, y_pred)
        f1 = f1_score(true_y, y_pred)
        mcc = matthews_corrcoef(true_y, y_pred)
        roc_auc = roc_auc_score(true_y, y_pred)
        metrics_dict = {'Accuracy': [acc], 'Precision': [prec], 'Recall': [recall], 'F1': [f1], 'MCC': [mcc],
                        'ROC_AUC': [roc_auc]}
    elif model.type == "Regression":
        mae = mean_absolute_error(true_y, y_pred)
        mse = mean_squared_error(true_y, y_pred)
        r2 = r2_score(true_y, y_pred)
        metrics_dict = {'MAE': [mae], 'MSE': [mse], "R2": [r2]}
    else:
        raise ValueError("Cannot find model type! Is it fitted?")

    df = pd.DataFrame(metrics_dict)

    return df


def summary(model: mlx.nn.Module):
    """
    Summarize the input mlx.nn.Module in a tabular format. If using graph models, input_size is not needed.
    :param model: nn.Module
        The input torch model.
    :return:
    """
    return MLXModelStatistics(model)


# todo set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


class MolToGraph:
    def __init__(self, smi: str, label: Optional[Union[int, float]] = None, graph_type: str = "AtomGraph",
                 add_smi: bool = False, add_feats: Optional[Union[np.array, list]] = None,
                 with_atom_index: bool = False):
        """
        Generate molecular graphs for use in the graph neural networks. The script pulls information from the .graph
        module with additional params to include information on tensors where needed.
        :param smi: str
            Molecular smiles string for conversion into molecular graph.
        :param label: Union[int,float]
            Label for molecule.
        :param graph_type: str
            The molecular graph type to generate from smiles string. Can be 'AtomGraph', 'PharmacophoreGraph'
            'FunctionalGraph', 'JunctionTreeGraph', or 'ClusterGraph'.
        :param add_smi: bool
            Whether to include the smiles string with the MLX Data object.
        :param add_feats: Optional[Union[np.array, list]]
            Include additional information to include with the molecular graph tensor. Must be given as a np.array or a
            list.
        :param with_atom_index: bool
            To include the smi string with atom index. Defaults to False.
        """
        if isinstance(add_feats, list):
            add_feats = np.array(add_feats)
        graph = mol_to_graph(smi=smi, label=label, graph_type=graph_type, add_smi=add_smi,
                             add_feats=add_feats, with_atom_index=with_atom_index)
        self.graph_data = _torch_to_mlx(graph)


# early_stopping.py
class EarlyStopping:
    def __init__(self, monitor: str = 'val_loss', patience: int = 50, mode: str = 'min'):
        """
        Set the params to trigger early stopping for fitting models.
        :param monitor: str
            Determine metric to monitor model training. Can be 'val_loss' or 'val_acc' or 'val_prec'.
        :param patience: int
            Set the number of epochs before early stopping is triggered.
        :param mode: str
            Determine if the metric should be 'min' or 'max'
        """
        self.monitor = monitor
        self.monitor_op = monitor
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.trigger_times = 0
        self.early_stop = False

        # determine monitor metric
        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < best
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.monitor_op = lambda current, best: current > best
            self.best_score = -float('inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def reset(self):
        """
        To reset the trigger counts. Method should be used during a loop, such as cross-validation.
        :return:
        """
        self.trigger_times = 0
        self.early_stop = False
        # reset best score so each new fit is scored independently
        self.best_score = float('inf') if self.mode == 'min' else -float('inf')

    def __call__(self, current_score):
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.trigger_times = 0
        else:
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                self.early_stop = True


class Metrics:
    def __init__(self):
        self.best_epoch = None
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_precision = []
        self.test_precision = []
        self.mcc = []
        self.recall = []
        self.f1_score = []
        self.roc_auc = []
        self.train_mae = []
        self.test_mae = []
        self.train_rmse = []
        self.test_rmse = []
        self.train_r2 = []
        self.test_r2 = []


class Base(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = None
        self.y_mean = None
        self.y_std = None
        self.type = None
        self.initial_params = None
        # info for plot
        self.metrics = Metrics()

    def fit(self, train_loader: Union[Dataloader, list], test_loader: Union[Dataloader, list], epochs: int = 10,
            optimizer: str = "adam", criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0,
            batch_size: int = 1024, device: str = "auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar=True,
            verbose: bool = False, device_warn: bool = True, seed: int = 42, **kwargs):
        """
        Fit the model.
        :param train_loader: Union[mlx_graphs.loaders.Dataloader, list]
            The data loader for the training data.
        :param test_loader: Union[mlx_graphs.loaders.Dataloader, list]
            The data loader for the test data.
        :param epochs: int
            Set the number of training epochs.
        :param optimizer: str
            Determine the model optimizer.
        :param criterion: str
            Set the model loss criterion.
        :param lr: float
            Set the optimizer learning rate.
        :param wt_decay: float
            Reduce the calculation of weights to the loss function.
        :param batch_size: int
            Set the  batch size during model training. This parameter is only used if the train_loader/test_loader input
            is a list instead of a Dataloader.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "gpu" (Apple Silicon), or "cpu".
        :param estop: Optional[Union[EarlyStopping, int]]
            Trigger EarlyStopping during model fitting. Can manually call the estop criteria, or call an int to
            designate the patience number. If an int is given, the early stopping will be triggered based on the
            smallest test loss for each epoch.
        :param pbar:
            Whether to show a progress bar during model fit for each epoch.
        :param verbose: bool
            Determine whether to output train/test accuracy and loss for each epoch.
        :param device_warn:
            Option to print which tensor/model is on 'cpu', 'cuda' or 'gpu' (Apple Silicon).
        :param seed: int
            Set the seed.
        :param kwargs: dict
            kwargs for a given criterion.
        :return:
        """
        global y_mean, y_std
        # set model to current instance
        model = self
        set_seed(seed)

        # check if on the same device
        fit_device = get_device(device)
        mx.set_default_device(fit_device)

        # set optimizer and criterion
        if optimizer in MLX_OPTI.keys():
            if optimizer in NO_WT_DECAY:
                # for optimizers with no learning rate
                optimizer = MLX_OPTI[optimizer](learning_rate=lr)
            else:
                optimizer = MLX_OPTI[optimizer](learning_rate=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! nly {list(MLX_OPTI.keys())} allowed!")

        # set criterion
        self.criterion = _set_criterion(criterion, self.type, self.classes)

        # check dataset input:
        np.random.seed(seed)  # try to enforce reproducibility
        if isinstance(train_loader, list):
            train_loader = Dataloader(train_loader, batch_size=batch_size,
                                      shuffle=True)  # avoid overfitting to sample order
        if isinstance(test_loader, list):
            test_loader = Dataloader(test_loader, batch_size=batch_size)

        if device_warn:
            if device == "auto" or device == "gpu":
                print(f"Model running on {state_apple_silicon_device()}")
            else:
                print(f"Model running on CPU. May have issues. Run using 'auto' or 'gpu' instead!")

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        # set loss and gradient function
        loss_and_grad_fn = mlx.nn.value_and_grad(model, self._forward_fn)

        mx.eval(model.parameters())

        # classification
        if isinstance(self.classes, int) and self.classes > 1:
            for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
                train_loss, train_acc, train_prec = train_c(model, train_loader, loss_and_grad_fn,
                                                            optimizer)
                test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test_c(model, test_loader,
                                                                                  self.criterion)
                # add metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.test_loss.append(test_loss)
                self.metrics.train_accuracy.append(train_acc)
                self.metrics.test_accuracy.append(test_acc)
                self.metrics.train_precision.append(train_prec)
                self.metrics.test_precision.append(test_prec)
                self.metrics.mcc.append(mcc)
                self.metrics.recall.append(recall)
                self.metrics.f1_score.append(f1)
                self.metrics.roc_auc.append(roc_auc)

                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

                # early stopping
                if estop is not None:
                    # determine monitoring metric
                    if estop.monitor == "val_loss":
                        monitor = test_loss
                    elif estop.monitor == "val_acc":
                        monitor = test_acc
                    else:
                        raise ValueError("Only 'val_loss' or 'val_acc' monitors allowed!")

                    estop(monitor)

                    if estop.early_stop:
                        print(f"Early stopping triggered after {epoch + 1} epochs!")
                        break
        # regression
        elif self.classes == 1 or self.classes is None:
            # for scaling, collect all y labels
            all_y_list = []

            for data in train_loader:
                all_y_list.append(data.y)
            all_y = mx.concatenate(all_y_list, axis=0) if all_y_list else None

            y_mean = all_y.mean()
            y_std = all_y.std()
            if y_std == 0: y_std = 1.0

            for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
                train_loss, train_mae, train_rmse, train_r2 = train_r(model, train_loader, loss_and_grad_fn,
                                                                      optimizer, y_mean, y_std)
                test_loss, test_mae, test_rmse, test_r2 = test_r(model, test_loader, self.criterion, y_mean,
                                                                 y_std)
                # add metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.test_loss.append(test_loss)
                self.metrics.train_mae.append(train_mae)
                self.metrics.test_mae.append(test_mae)
                self.metrics.train_rmse.append(train_rmse)
                self.metrics.test_rmse.append(test_rmse)
                self.metrics.train_r2.append(train_r2)
                self.metrics.test_r2.append(test_r2)

                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train R2: {train_r2:.4f} | "
                          f"Test Loss: {test_loss:.4f} | Test R2: {test_r2:.4f}")

                # early stopping
                if estop is not None:
                    # determine monitoring metric
                    if estop.monitor == "val_loss":
                        monitor = test_loss
                    else:
                        raise ValueError("Only 'val_loss' monitors allowed!")

                    estop(monitor)

                    if estop.early_stop:
                        print(f"Early stopping triggered after {epoch + 1} epochs!")
                        break

        # to unstandardize predictions
        if self.classes == 1 or self.classes is None:
            self.y_mean = y_mean
            self.y_std = y_std

        # var for cross validation
        self.is_fitted = True

        # get epoch
        best_train_epoch = np.argmin(self.metrics.train_loss)
        best_test_epoch = np.argmin(self.metrics.test_loss)
        self.metrics.best_epoch = best_test_epoch.item()  # convert from np to int

        if device_warn:
            if isinstance(self.classes, int) and self.classes > 1:  # classification
                print(f"Best Training Loss: {round(self.metrics.train_loss[best_train_epoch], 3)}")
                print(f"Best Test Loss: {round(self.metrics.test_loss[best_test_epoch], 3)}")
                print(f"Best Training Accuracy: {round(self.metrics.train_accuracy[best_train_epoch], 3)}")
                print(f"Best Test Accuracy: {round(self.metrics.test_accuracy[best_test_epoch], 3)}")
            else:
                print(f"Best Training Loss: {round(self.metrics.train_loss[best_train_epoch], 3)}")
                print(f"Best Test Loss: {round(self.metrics.test_loss[best_test_epoch], 3)}")
                print(f"Best Training MAE: {round(self.metrics.train_mae[best_train_epoch], 3)}")
                print(f"Best Test MAE: {round(self.metrics.test_mae[best_test_epoch], 3)}")

    def predict(self, data: Union[np.array, list, Dataloader], batch_size: int = None,
                binary: Union[bool, float, str, None] = True, cutoff: float = 0.5, return_type: str = "list",
                add_feats: Optional[Union[np.array, list, int]] = None, pbar: bool = False):
        """
        Handle predictions for an input MLX Dataloader.
        :param data: Union[np.array, list, Dataloader]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param batch_size: int
            Set the batch size for prediction. Only used if the input data is of type list.
        :param binary: Union[bool, float, str]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. Input strings can be "multi" or 'proba', where "multi" will output a
            multi-class labels and "proba" will output a binary probability vector. If 'False' is given, the model will
            output the probability for the final classification label.
        :param cutoff:
            Set the cutoff for binary predictions. This param is only available for binary models and when the binary
            param is set to 'True'.
        :param return_type: str
            Set the return type of the prediction. Can be "array" or "list".
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param pbar: bool
             Show progress bar for predictions.
        :return:
        """
        # # check if on the same device
        # fit_device = get_device(device)
        # print(f"Prediction running on {fit_device}")
        # set model to current instance
        global pred_out
        model = self
        if self.add_feats is not None:
            add_feats = self.add_feats
        elif self.add_feats is None and add_feats is not None:
            pass  # use add_feats param

        # set model to evaluation mode
        self.eval()

        # for graph models, check and convert into Dataloader
        if isinstance(data, list) and batch_size is not None:
            data = Dataloader(data, batch_size=batch_size, shuffle=False)
        elif isinstance(data, list) and batch_size is None:
            raise ValueError("Input data is a list. Needs batch_size to be set!")
        elif isinstance(data, Dataloader):
            data = data

        all_preds = []

        for x in tqdm(data, disable=not pbar):
            # forward pass
            # check if edge_features used by model, checks using hte model.__call__
            sig = inspect.signature(model.__call__)
            # set features
            kwargs = {
                "edge_index": getattr(x, "edge_index", None),
                "x": getattr(x, "node_features", None),
                "batch_indices": getattr(x, "batch_indices", None)
            }
            if "edge_features" in sig.parameters and hasattr(x, "edge_features"):
                kwargs["edge_features"] = x.edge_features
            if "add_feats" in sig.parameters and hasattr(x, "add_feats") and add_feats is not None:
                kwargs["add_feats"] = x.add_feats
            elif "add_feats" in sig.parameters and hasattr(x, "add_feats") and add_feats is None:
                raise ValueError("'add_feats' found in input, but not given for prediction!")

            output = model(**kwargs)

            # regression output
            if self.type == "Regression":
                # For regression, we just want the raw output, no softmax/thresholding
                pred_out = output
            # classification output
            elif self.type == "Classification":
                # binary labels
                if binary is True:
                    # grab positive class prob (index 1)
                    probs = mx.softmax(output, axis=1)[:, 1]
                    pred_out = (probs >= cutoff).astype(mx.int32)
                elif binary is False or binary is None:
                    # grab positive class prob (index 1)
                    pred_out = mx.softmax(output, axis=1)[:, 1]
                # multi-class classification
                elif binary == "multi":
                    # Returns the Class Index (0, 1, 2...) not probabilities
                    pred_out = mx.argmax(output, axis=1)
                # probabilities in float
                elif binary == "proba":
                    # Returns probabilities for all classes
                    pred_out = mx.softmax(output, axis=1)
                else:
                    raise ValueError("Only 'multi', 'proba', boolean, is supported for binary param!")
            else:
                raise ValueError("Model is not labeled as type 'Regression' or 'Classification'?")

            mx.eval(pred_out)
            all_preds.append(pred_out)

        # combine predictions from batch, concact and extract mx array
        if self.type == "Regression":
            # squeeze regression output
            output = mx.array(all_preds)
            output_flat = mx.squeeze(output)
            predictions = output_flat
            if hasattr(self, 'y_mean') and hasattr(self, 'y_std'):
                predictions = predictions * self.y_std + self.y_mean
        else:
            predictions = mx.concatenate(all_preds, axis=0)

        # conditional check to output as list or tensor
        if return_type == "list":
            return predictions.tolist()
        elif return_type == "array":
            return np.array(predictions)
        else:
            raise TypeError(f"Type {return_type} not recognized!")

    def save(self, filename: str):
        """
        Save a fitted MLX model.
        :param filename: str
            The filepath for the saved MLX model. Must be in '.npz' format.
        :return:
        """
        model = self

        # flatten model
        params = tree_flatten(model.parameters())
        mx.savez(filename, **dict(params))

    def load(self, model_path: str = None, param_path: Optional[str] = None, add_feats: Optional[np.ndarray] = None,
             node_dim: int = 142, edge_feat_dim: int = 16, criterion: str = "cross_entropy"):
        """
        Load a saved MLX model. This will return a model with the loaded parameters and model architecture.
        :param model_path: str
            Filepath to saved mlx model. Must be in '.npz' format.
        :param param_path: Optional[str]
            Filepath to saved mlx model parameters. Must be in .json format.
        :param add_feats: Optional[np.ndarray]
            Additional features to be added to the saved model.
        :param node_dim: int
            The length of feature vector per node. Must match what was used for the saved mlx model.
        :param edge_feat_dim: int
            The dimensions of the hidden layers in teh graph network. Must match what was used for the saved mlx model.
        :param criterion: str
            Set the model loss function.
        :return:
        """

        if param_path is not None:
            # load params from json file
            with open(param_path, 'r') as f:
                params = json.load(f)

            # additional params not found in json file
            params['node_dim'] = node_dim
            params['edge_feat_dim'] = edge_feat_dim
            params['num_classes'] = self.classes
            params['criterion'] = criterion
            params['add_feats'] = add_feats

            # load params by model
            if model.__class__ == GCNNet:
                for_model_params = ["node_dim", "hidden_dim", "graph_layers", "mlp_layers", "graph_dropout",
                                    "mlp_dropout", "add_feats"]
            elif model.__class__ == GINENet:
                for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "graph_layers", "mlp_layers",
                                    "graph_dropout", "mlp_dropout", "add_feats"]
            elif model.__class__ == GSAGENet:
                for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "graph_layers", "mlp_layers",
                                    "graph_dropout", "mlp_dropout", "add_feats"]
            elif model.__class__ == GATNet:
                for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "heads", "graph_layers", "mlp_layers",
                                    "mlp_dropout", "add_feats"]
            else:
                raise ValueError("Cannot identify model class! Only 'GCNNet' or 'GINENet' supported!")
            model_params = {param_key: params[param_key] for param_key in for_model_params}
            # # to check loading of model
            # print(model_params)
            model = self.__class__(**model_params)
        else:
            model = self

        # load weights
        weights = mx.load(model_path)
        # separate model parameters from Regression scale
        model_params = {k: v for k, v in weights.items() if k not in ['y_mean', 'y_std']}
        model.update(tree_unflatten(list(model_params.items())))

        # set Regression or Classification attributes
        R_CRITERION = ['mse', 'huber', 'l1', 'smooth_l1']
        C_CRITERION = ['cross_entropy', 'nll', 'kl_divergence']
        is_regression = criterion in R_CRITERION and (self.classes is None or self.classes == 1)
        is_classification = criterion in C_CRITERION and (self.classes is not None and self.classes >= 2)

        # check model conditions
        if is_regression:
            # check weight
            if 'y_mean' not in weights or 'y_std' not in weights:
                raise ValueError(f"Mismatch! criterion='{criterion}' is set for 'Regression', "
                                 "but model is configured for 'Classification'?")
            model.type = "Regression"
            model.y_mean = weights['y_mean']
            model.y_std = weights['y_std']

        elif is_classification:
            # check weight
            if 'y_mean' in weights or 'y_std' in weights:
                raise ValueError(f"Mismatch! criterion='{criterion}' is set for 'Classification', "
                                 "but model is configured for 'Regression'?")
            model.type = "Classification"
        else:
            raise ValueError(
                f"Mismatch! criterion='{criterion}', num_classes={self.classes}. "
                "Check model params!"
            )

        # add tags
        model.is_fitted = True
        print("Model loaded!")

        return model

    def reset_params(self):
        """reset model to initial params"""

        if self.initial_params is None:
            raise ValueError("Initial parameters not stored! Call store_initial_params() first!")

        # subfunction to copy nested parameter dictionaries
        def copy_nested_params(params):
            if isinstance(params, dict):
                return {key: copy_nested_params(value) for key, value in params.items()}
            elif isinstance(params, list):
                return [copy_nested_params(item) for item in params]
            else:
                # array/parameter
                return mx.array(params)

        reset_params = copy_nested_params(self.initial_params)
        self.update(reset_params)

    def store_initial_params(self):
        """store initial parameters later reset"""
        if self.initial_params is None:
            self.initial_params = copy.deepcopy(dict(self.parameters()))

    def _loss_fn(self, y_hat, y, classes, parameters=None):
        # normalize shape if number of class is 1
        if classes == 1:
            y = y_hat.reshape(-1, 1)
            y = y.reshape(-1, 1)

        loss = mx.mean(self.criterion(y_hat, y))
        return loss

    def _forward_fn(self, model, graph, labels):
        """forward function"""
        if hasattr(graph, "add_feats"):
            model_args = {
                "edge_index": graph.edge_index,
                "x": graph.node_features,
                "batch_indices": graph.batch_indices,
                "add_feats": graph.add_feats,
            }
        else:
            model_args = {
                "edge_index": graph.edge_index,
                "x": graph.node_features,
                "batch_indices": graph.batch_indices,
            }

        # determine if model accepts additional edge_features
        sig = inspect.signature(model.__call__)
        if "edge_features" in sig.parameters:
            model_args["edge_features"] = graph.edge_features

        # forward pass
        y_output = model(**model_args)
        # compute loss
        loss = self._loss_fn(y_output, labels, self.classes, model.parameters())
        # return loss, y_hat
        return loss, y_output

    def _build_mlp(self, input_dim, mlp_layers, num_classes, dropout):
        """Dynamically build mlp layer"""
        dimension = input_dim  # graph latent space
        # loop mlp layers
        self.mlp = []
        for _ in range(mlp_layers - 1):
            reduced_dimension = round(dimension / 2)
            self.mlp.append(mlx.nn.Sequential(
                mlx.nn.Linear(dimension, reduced_dimension),
                mlx.nn.BatchNorm(reduced_dimension),
                mlx.nn.ReLU(),
                mlx.nn.Dropout(dropout)
            ))
            dimension = reduced_dimension

        # final mlp layer
        if num_classes is None:
            num_classes = 1
        self.final_mlp = mlx.nn.Linear(dimension, num_classes)

        return self.mlp, self.final_mlp


# Graph Convolutional Neural Network
class GCNNet(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, mlp_layers: int = 1,
                 reduce_graph: bool = False, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 num_classes: Union[int, None] = 2, act: str = "relu",
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """
        Generate graph model for graph convolutional (GCNC) network. The default model will generate two graph layers
        and one mlp layer.
        :param node_dim: int
            Set the length of node feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param graph_dropout: float
            Set the dropout for each graph layer.
        :param mlp_dropout: float
            Indicate a dropout layer after each mlp layer. Defaults to 0 (no dropout).
        :param num_classes: Union[int, None]
            Number of classes for the output layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        global graph_hidden_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False
        if num_classes == 1:
            num_classes = None
        self.classes = num_classes
        self.add_feats = add_feats

        # set model type
        if isinstance(self.classes, int) and self.classes > 1:
            self.type = "Classification"
        elif self.classes == 1 or self.classes is None:
            self.type = "Regression"

        # set graph layers
        self.gcn_conv = []
        self.normalization = []

        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            self.gcn_conv.append(GCNConv(in_dim, hidden_dim))
            self.normalization.append(mlx.nn.BatchNorm(hidden_dim))

            # update dimensions for next layer
            graph_hidden_dim = hidden_dim
            if reduce_graph:
                hidden_dim = round(hidden_dim / 2)

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = hidden_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, mlp_dropout)

    def __call__(self, edge_index, x, batch_indices, add_feats=None):
        # graph layers
        act_fn = MLX_ACT[self.act]
        for conv, norm in zip(self.gcn_conv, self.normalization):
            x = conv(edge_index, x)
            x = norm(x)
            x = act_fn(x)
            if self.graph_dropout > 0:
                x = mlx.nn.Dropout(p=self.graph_dropout)(x)

        # readout layer
        x = _global_pooling(x, batch_indices, pool_type=self.pooling)

        # concatenate additional features
        if add_feats is not None:
            # convert to mx.array
            add_feats = mx.array(add_feats, dtype=x.dtype)
            # reshape to mx.array to match latent space shape
            if add_feats.ndim == 1:
                add_feats = add_feats.reshape(x.shape[0], -1)
            if add_feats.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x has {x.shape[0]}, add_feats has {add_feats.shape[0]}")
            x = mx.concatenate([x, add_feats], axis=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
            if self.mlp_dropout > 0:
                x = mlx.nn.Dropout(p=self.mlp_dropout)(x)
        x = self.final_mlp(x)

        return x


class GINENet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: int = 16, hidden_dim: int = 64, graph_layers: int = 2,
                 mlp_layers: int = 1, reduce_graph: bool = False, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 num_classes: Union[int, None] = 2, act: str = "relu",
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "add"):
        """
        Generate graph model for graph isomorphism convolutional (GINEConv) network. The default model will generate two
        graph layers and one mlp layer.
        :param node_dim: int
            Length of node feature vector.
        :param edge_feat_dim: int
            Length of edge feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param graph_dropout: float
            Set the graph dropout during activation.
        :param mlp_dropout: float
            Indicate a dropout layer after each mlp layer. Defaults to 0 (no dropout).
        :param num_classes: Union[int, None]
            Number of classes for the output layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        global reduced_graph_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False
        if num_classes == 1:
            num_classes = None
        self.classes = num_classes
        self.add_feats = add_feats

        # set model type
        if isinstance(self.classes, int) and self.classes > 1:
            self.type = "Classification"
        elif self.classes == 1 or self.classes is None:
            self.type = "Regression"

        # set graph layers
        self.gine_conv = []
        self.normalization = []

        # set graph layers
        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the layer nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            single_mlp = mlx.nn.Sequential(
                mlx.nn.Linear(in_dim, hidden_dim),
                mlx.nn.BatchNorm(hidden_dim),
                mlx.nn.ReLU(),
                mlx.nn.Linear(hidden_dim, hidden_dim),
                mlx.nn.ReLU()
            )

            # add GINE layer
            self.gine_conv.append(GINConv(mlp=single_mlp, node_features_dim=in_dim, edge_features_dim=edge_feat_dim))
            self.normalization.append(mlx.nn.BatchNorm(hidden_dim))

            # update dimensions for next layer
            graph_hidden_dim = hidden_dim
            if reduce_graph:
                hidden_dim = round(hidden_dim / 2)

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = graph_hidden_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, mlp_dropout)

    def __call__(self, edge_index, x, edge_features, batch_indices, add_feats=None):
        # graph layers
        act_fn = MLX_ACT[self.act]
        for conv, norm in zip(self.gine_conv, self.normalization):
            x = conv(edge_index, x)
            x = norm(x)
            x = act_fn(x)
            if self.graph_dropout > 0:
                x = mlx.nn.Dropout(p=self.graph_dropout)(x)

        # readout layer
        x = _global_pooling(x, batch_indices, pool_type=self.pooling)

        # concatenate additional features
        if add_feats is not None:
            # convert to mx.array
            add_feats = mx.array(add_feats, dtype=x.dtype)
            # reshape to mx.array to match latent space shape
            if add_feats.ndim == 1:
                add_feats = add_feats.reshape(x.shape[0], -1)
            if add_feats.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x has {x.shape[0]}, add_feats has {add_feats.shape[0]}")
            x = mx.concatenate([x, add_feats], axis=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
            if self.mlp_dropout > 0:
                x = mlx.nn.Dropout(p=self.mlp_dropout)(x)
        x = self.final_mlp(x)

        return x


class GSAGENet(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, mlp_layers: int = 1,
                 graph_dropout: float = 0.0, mlp_dropout: float = 0.0, num_classes: Union[int, None] = 2,
                 act: str = "relu", reduce_graph=False, add_feats: Optional[Union[np.array, list, int]] = None,
                 pooling: str = "mean"):
        """
        Generate graph SAGE model neural network (GSAGENet). The default model will generate two graph layers and one
        mlp layer.
        :param node_dim: int
            Set the length of node feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in teh graph network.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param graph_dropout: float
            Set the dropout for each graph layer.
        :param mlp_dropout: float
            Set the dropout for each MLP layer.
        :param num_classes: Union[int, None]
            Number of classes for the output layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set the pooling method.
        """
        global graph_hidden_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False
        if num_classes == 1:
            num_classes = None
        self.classes = num_classes
        self.add_feats = add_feats

        # set model type
        if isinstance(self.classes, int) and self.classes > 1:
            self.type = "Classification"
        elif self.classes == 1 or self.classes is None:
            self.type = "Regression"

        # set graph layers
        self.sage_conv = []
        self.normalization = []

        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            self.sage_conv.append(SAGEConv(in_dim, hidden_dim))
            self.normalization.append(mlx.nn.BatchNorm(hidden_dim))

            # update dimensions for next layer
            graph_hidden_dim = hidden_dim
            if reduce_graph:
                hidden_dim = round(hidden_dim / 2)

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = hidden_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, mlp_dropout)

    def __call__(self, edge_index, x, batch_indices, add_feats=None):
        # graph layers
        act_fn = MLX_ACT[self.act]
        for conv, norm in zip(self.sage_conv, self.normalization):
            x = conv(edge_index, x)
            x = norm(x)
            x = act_fn(x)
            if self.graph_dropout > 0:
                x = mlx.nn.Dropout(p=self.graph_dropout)(x)

        # readout layer
        x = _global_pooling(x, batch_indices, pool_type=self.pooling)

        # concatenate additional features
        if add_feats is not None:
            # convert to mx.array
            add_feats = mx.array(add_feats, dtype=x.dtype)
            # reshape to mx.array to match latent space shape
            if add_feats.ndim == 1:
                add_feats = add_feats.reshape(x.shape[0], -1)
            if add_feats.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x has {x.shape[0]}, add_feats has {add_feats.shape[0]}")
            x = mx.concatenate([x, add_feats], axis=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
            if self.mlp_dropout > 0:
                x = mlx.nn.Dropout(p=self.mlp_dropout)(x)
        x = self.final_mlp(x)

        return x


class GATNet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: Optional[int] = None, hidden_dim: int = 128, heads: int = 1,
                 graph_layers: int = 2, mlp_layers: int = 1, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 act: str = "relu", reduce_graph=False, add_feats: Optional[Union[np.array, list, int]] = None,
                 num_classes: Union[int, None] = 2, pooling: str = "mean"):
        """
        Generate a Graph Attention Network neural network (GATNet). The default model will generate two graph layers and
        one mlp layer.
        :param node_dim: int
            Set the length of node feature vector.
        :param edge_feat_dim: Optional[int]
            Optional. Set the dimensions of the edge feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param heads: int
            Set teh number of attention heads.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param graph_dropout: float
            Set the dropout for each graph layer.
        :param mlp_dropout: float
            Set the dropout for each MLP layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param num_classes: Union[int, None]
            Number of classes for the output layer.
        :param pooling: str
            Set pooling method.
        """
        global reduced_graph_dim, out_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False
        if num_classes == 1:
            num_classes = None
        self.classes = num_classes
        self.add_feats = add_feats

        # set model type
        if isinstance(self.classes, int) and self.classes > 1:
            self.type = "Classification"
        elif self.classes == 1 or self.classes is None:
            self.type = "Regression"

        # set graph layers
        self.gat_conv = []
        self.normalization = []

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.gat_conv.append(GATConv(node_features_dim=in_channels, out_features_dim=graph_hidden_dim,
                                             heads=heads, edge_features_dim=edge_feat_dim, dropout=graph_dropout))
                # normalize nodes
                self.normalization.append(mlx.nn.BatchNorm(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.gat_conv.append(GATConv(node_features_dim=in_channels, out_features_dim=graph_hidden_dim,
                                             heads=heads, edge_features_dim=edge_feat_dim, dropout=graph_dropout))
                # normalize nodes
                self.normalization.append(mlx.nn.BatchNorm(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels for loop

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = graph_hidden_dim
        mlp_dim = mlp_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, mlp_dropout)

    def __call__(self, edge_index, x, edge_features=None, batch_indices=None, add_feats=None):
        # graph layers
        act_fn = MLX_ACT[self.act]
        for conv, norm in zip(self.gat_conv, self.normalization):
            # passes based on edge_features dimensions
            if hasattr(conv, 'edge_lin_proj') and edge_features is not None:
                x = conv(edge_index, x, edge_features)
                x = norm(x)
                x = act_fn(x)
                if self.graph_dropout > 0:
                    x = mlx.nn.Dropout(p=self.graph_dropout)(x)
            else:
                x = conv(edge_index, x)
                x = norm(x)
                x = act_fn(x)
                if self.graph_dropout > 0:
                    x = mlx.nn.Dropout(p=self.graph_dropout)(x)

        # readout layer
        x = _global_pooling(x, batch_indices, pool_type=self.pooling)

        # concatenate additional features
        if add_feats is not None:
            # convert to mx.array
            add_feats = mx.array(add_feats, dtype=x.dtype)
            # reshape to mx.array to match latent space shape
            if add_feats.ndim == 1:
                add_feats = add_feats.reshape(x.shape[0], -1)
            if add_feats.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x has {x.shape[0]}, add_feats has {add_feats.shape[0]}")
            x = mx.concatenate([x, add_feats], axis=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
            if self.mlp_dropout > 0:
                x = mlx.nn.Dropout(p=self.mlp_dropout)(x)
        x = self.final_mlp(x)

        return x


class MLP_Baseline(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = False
        self.y_mean = None
        self.y_std = None
        self.type = None
        self.initial_params = None
        # info for plot
        self.metrics = Metrics()

    def fit(self, x_train: Union[list, np.ndarray], y_train: Union[list, np.ndarray], x_test: Union[list, np.ndarray],
            y_test: Union[list, np.ndarray], epochs: int = 10, optimizer: str = "adam",
            criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0, batch_size: int = 1024,
            device: str = "auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar: bool = True,
            verbose: bool = False, device_warn: bool = True, seed: int = 42, **kwargs):
        """
        Fit the model. The x_train and y_train and x_test and y_test given will be internally converted into a torch
        tensor before fitting the model.
        :param x_train: Union[list, np.ndarray]
            The training dataset features.
        :param y_train: Union[list, np.ndarray]
            The training labels.
        :param x_test: Union[list, np.ndarray]
            The testing dataset features.
        :param y_test: Union[list, np.ndarray]
            The testing labels.
        :param epochs: int
             Set the number of training epochs.
        :param optimizer: str
            Set the model optimizer.
        :param criterion: str
            Set the model loss function.
        :param lr: float
            Set hte optimizer learning rate.
        :param wt_decay: float
            Reduce the calculation of weights to the loss function.
        :param batch_size: int
            Set the batch size during model training. Defaults to 1024.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param estop: Optional[Union[EarlyStopping, int]
            Trigger EarlyStopping during model fitting. Can manually call the estop criteria, or call an int to
            designate the patience number. If an int is given, the early stopping will be triggered based on the
            smallest test loss for each epoch.
        :param pbar:
            Whether to show a progress bar during model fit for each epoch
        :param verbose: bool
            Determine whether to output train/test accuracy and loss for each epoch.
        :param device_warn: bool
            Option to print which tensor/model is on 'cpu', 'cuda', or 'mps'.
        :param kwargs: dict
            kwargs for a given criterion.
        :return:
        """
        global y_mean, y_std
        # set model
        model = self

        # check if on the same device
        fit_device = get_device(device)
        mx.set_default_device(fit_device)

        # set optimizer and criterion
        if optimizer in MLX_OPTI.keys():
            if optimizer in NO_WT_DECAY:
                # for optimizers with no learning rate
                optimizer = MLX_OPTI[optimizer](learning_rate=lr)
            else:
                optimizer = MLX_OPTI[optimizer](learning_rate=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! nly {list(MLX_OPTI.keys())} allowed!")

        # set criterion
        self.criterion = _set_criterion(criterion, self.type, self.classes)

        # check dataset input:
        np.random.seed(seed)  # try to enforce reproducibility
        if device_warn:
            if device == "auto" or device == "gpu":
                print(f"Model running on {state_apple_silicon_device()}")
            else:
                print(f"Model running on CPU. May have issues. Run using 'auto' or 'gpu' instead!")

        # check dataset input and convert ot mx array:
        if all(isinstance(x, np.ndarray) for x in (x_train, x_test, y_train, y_test)):
            x_train, x_test, y_train, y_test = (mx.array(x_train), mx.array(x_test), mx.array(y_train),
                                                mx.array(y_test))
        # convert list of ndarray to mlx array
        elif all(isinstance(x, list) for x in (x_train, x_test, y_train, y_test)):
            x_train, x_test, y_train, y_test = (np.asarray(x_train, dtype=np.float32),
                                                np.asarray(x_test, dtype=np.float32),
                                                np.asarray(y_train, dtype=np.float32),
                                                np.asarray(y_test, dtype=np.float32))
            x_train, x_test, y_train, y_test = (mx.array(x_train), mx.array(x_test), mx.array(y_train),
                                                mx.array(y_test))

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        # set loss and gradient function
        loss_and_grad_fn = mlx.nn.value_and_grad(model, self._loss_fn)

        mx.eval(model.parameters())

        # classification
        if isinstance(self.classes, int) and self.classes > 1:
            for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
                train_loss, train_acc, train_prec = train_mlp_c(model, x_train, y_train, optimizer, loss_and_grad_fn,
                                                                batch_size)
                test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test_mlp_c(model, x_test, y_test,
                                                                                      self._loss_fn, batch_size)
                # add metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.test_loss.append(test_loss)
                self.metrics.train_accuracy.append(train_acc)
                self.metrics.test_accuracy.append(test_acc)
                self.metrics.train_precision.append(train_prec)
                self.metrics.test_precision.append(test_prec)
                self.metrics.mcc.append(mcc)
                self.metrics.recall.append(recall)
                self.metrics.f1_score.append(f1)
                self.metrics.roc_auc.append(roc_auc)

                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

                # early stopping
                if estop is not None:
                    # determine monitoring metric
                    if estop.monitor == "val_loss":
                        monitor = test_loss
                    elif estop.monitor == "val_acc":
                        monitor = test_acc
                    else:
                        raise ValueError("Only 'val_loss' or 'val_acc' monitors allowed!")

                    estop(monitor)

                    if estop.early_stop:
                        print(f"Early stopping triggered after {epoch + 1} epochs!")
                        break
        # regression
        elif self.classes == 1 or self.classes is None:
            # for scaling, collect all y labels
            all_y_list = []
            all_y_list.append(y_train)
            all_y = mx.concatenate(all_y_list, axis=0) if all_y_list else None

            y_mean = all_y.mean()
            y_std = all_y.std()
            if y_std == 0: y_std = 1.0

            for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
                train_loss, train_mae, train_rmse, train_r2 = train_mlp_r(model, x_train, y_train, loss_and_grad_fn,
                                                                          optimizer, batch_size)
                test_loss, test_mae, test_rmse, test_r2 = test_mlp_r(model, x_test, y_test, self.criterion, batch_size,
                                                                     y_mean, y_std)
                # add metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.test_loss.append(test_loss)
                self.metrics.train_mae.append(train_mae)
                self.metrics.test_mae.append(test_mae)
                self.metrics.train_rmse.append(train_rmse)
                self.metrics.test_rmse.append(test_rmse)
                self.metrics.train_r2.append(train_r2)
                self.metrics.test_r2.append(test_r2)

                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train R2: {train_r2:.4f} | "
                          f"Test Loss: {test_loss:.4f} | Test R2: {test_r2:.4f}")

                # early stopping
                if estop is not None:
                    # determine monitoring metric
                    if estop.monitor == "val_loss":
                        monitor = test_loss
                    else:
                        raise ValueError("Only 'val_loss' monitors allowed!")

                    estop(monitor)

                    if estop.early_stop:
                        print(f"Early stopping triggered after {epoch + 1} epochs!")
                        break

        # to unstandardize predictions
        if self.classes == 1 or self.classes is None:
            self.y_mean = y_mean
            self.y_std = y_std

        # var for cross validation
        self.is_fitted = True

        # get epoch
        best_train_epoch = np.argmin(self.metrics.train_loss)
        best_test_epoch = np.argmin(self.metrics.test_loss)
        self.metrics.best_epoch = best_test_epoch.item()  # convert from np to int

        if device_warn:
            if isinstance(self.classes, int) and self.classes > 1:  # classification
                print(f"Best Training Loss: {round(self.metrics.train_loss[best_train_epoch], 3)}")
                print(f"Best Test Loss: {round(self.metrics.test_loss[best_test_epoch], 3)}")
                print(f"Best Training Accuracy: {round(self.metrics.train_accuracy[best_train_epoch], 3)}")
                print(f"Best Test Accuracy: {round(self.metrics.test_accuracy[best_test_epoch], 3)}")
            else:
                print(f"Best Training Loss: {round(self.metrics.train_loss[best_train_epoch], 3)}")
                print(f"Best Test Loss: {round(self.metrics.test_loss[best_test_epoch], 3)}")
                print(f"Best Training MAE: {round(self.metrics.train_mae[best_train_epoch], 3)}")
                print(f"Best Test MAE: {round(self.metrics.test_mae[best_test_epoch], 3)}")

    def predict(self, data: Union[np.array, list, mx.array], batch_size: int = None,
                binary: Union[bool, float, str, None] = True, cutoff: float = 0.5, return_type: str = "list",
                add_feats: Optional[Union[np.array, list, int]] = None, pbar: bool = False):
        """
        Handle predictions for an input MLX Dataloader.
        :param data: Union[np.array, list]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param batch_size: int
            Set the batch size for prediction. Only used if the input data is of type list.
        :param binary: Union[bool, float, str, None]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. Input strings can be "multi" or 'proba', where "multi" will output a
            multi-class labels and "proba" will output a binary probability vector. If 'False' is given, the model will
            output the probability for the final classification label.
        :param cutoff:
            Set the cutoff for binary predictions. This param is only available for binary models and when the binary
            param is set to 'True'.
        :param return_type: str
            Set the return type of the prediction. Can be "array" or "list".
        :param pbar: bool
             Show progress bar for predictions.
        :return:
        """
        global pred_out
        model = self
        if not self.is_fitted:
            raise RuntimeError(f"Model is not fitted yet!")

        all_preds = []

        # check dataset input
        if isinstance(data, mx.array):
            data = data
        elif isinstance(data, (list, np.ndarray)):
            if isinstance(data, list):
                data = np.asarray(data, dtype=np.float32)
            data = mx.array(data)
        else:
            raise TypeError(f"Input data type {type(data)} not supported! Need np.ndarray, list, or mx.array!")

        # evaluation mode
        self.eval()

        for x in tqdm(data, disable=not pbar):
            # set data dimensions
            x = mx.expand_dims(x, axis=0)

            output = model(x)

            # regression output
            if self.type == "Regression":
                pred_out = output
            # classification output
            elif self.type == "Classification":
                # binary labels
                if binary == True:
                    proba = mx.softmax(output, axis=1)[:, 1]
                    pred_out = (proba >= cutoff).astype(mx.int32)
                elif binary is False or binary is None:
                    pred_out = mx.softmax(output, axis=1)[:, 1]
                # multi-class classification
                elif binary == "multi":
                    pred_out = mx.argmax(output, axis=1)
                # output probabilities
                elif binary == "proba":
                    pred_out = mx.softmax(output, axis=1)
                else:
                    raise ValueError("Only 'multi', 'proba', boolean, is supported for binary param!")
            else:
                raise ValueError("Model is not labeled as type 'Regression' or 'Classification'?")

            mx.eval(pred_out)
            all_preds.append(pred_out)

        # combine predictions from batch, concact and extract mx array
        predictions = mx.concatenate(all_preds, axis=0)
        # if regression, get set scale
        if self.type == "Regression":
            predictions = mx.squeeze(predictions)
            if hasattr(self, 'y_mean') and hasattr(self, 'y_std'):
                predictions = predictions * self.y_std + self.y_mean

        # conditional check to output as list or tensor
        if return_type == "list":
            return predictions.tolist()
        elif return_type == "array":
            return predictions
        else:
            raise TypeError(f"Type {return_type} not recognized!")

    def save(self, filename: str):
        """
        Save a fitted MLX model.
        :param filename: str
            The filepath for the saved MLX model. Must be in '.npz' format.
        :return:
        """
        model = self

        # flatten model
        params = tree_flatten(model.parameters())
        mx.savez(filename, **dict(params))

    def load(self, model_path: Optional[str] = None, param_path: Optional[str] = None,
             num_classes: Union[int, None] = 2, criterion: str = "cross_entropy"):
        """
        Load saved model state.
        :param model_path: Optional[str]
            Path to saved torch model.
        :param param_path: Optional[str]
            Set the path for the saved parameters in .json file.
        :param num_classes: Union[int, None]
            The number of classes for the output layer. Must match what was used for the saved mlx model.
        :param criterion: str
            Set the model loss function.
        :return:
        """

        if param_path is not None:
            # load params from json file
            with open(param_path, 'r') as f:
                params = json.load(f)

            # additional params not found in json file
            params['num_classes'] = self.classes

            for_model_params = None
            print("for_model_params needs to be included!")
            # load model with params
            model_params = {param_key: params[param_key] for param_key in for_model_params}
            model = self.__class__(**model_params)
        else:
            model = self

        # load model and set to eval mode
        weights = mx.load(model_path)
        # separate model parameters from Regression scale
        model_params = {k: v for k, v in weights.items() if k not in ['y_mean', 'y_std']}
        model.update(tree_unflatten(list(model_params.items())))

        # set Regression or Classification attributes
        R_CRITERION = ['mse', 'huber', 'l1', 'smooth_l1']
        C_CRITERION = ['cross_entropy', 'nll', 'kl_divergence']
        is_regression = criterion in R_CRITERION and (self.classes is None or self.classes == 1)
        is_classification = criterion in C_CRITERION and (self.classes is not None and self.classes >= 2)

        # check model conditions
        if is_regression:
            # check weight
            if 'y_mean' not in weights or 'y_std' not in weights:
                raise ValueError(f"Mismatch! criterion='{criterion}' is set for 'Regression', "
                                 "but model is configured for 'Classification'?")
            model.type = "Regression"
            model.y_mean = weights['y_mean']
            model.y_std = weights['y_std']

        elif is_classification:
            # check weight
            if 'y_mean' in weights or 'y_std' in weights:
                raise ValueError(f"Mismatch! criterion='{criterion}' is set for 'Classification', "
                                 "but model is configured for 'Regression'?")
            model.type = "Classification"
        else:
            raise ValueError(
                f"Mismatch! criterion='{criterion}', num_classes={self.classes}. "
                "Check model params!"
            )

        # add tags
        model.is_fitted = True
        print("Model loaded!")

        return model

    def store_initial_params(self):
        """store initial parameters later reset"""
        if self.initial_params is None:
            self.initial_params = copy.deepcopy(dict(self.parameters()))

    def _loss_fn(self, model, X, y, parameters=None):
        y_output = model(X)
        loss = mx.mean(self.criterion(y_output, y))
        return loss

    def _eval_fn(self, model, X, y):
        """Compute accuracy"""
        return mx.mean(mx.argmax(model(X), axis=1) == y)


class MLP(MLP_Baseline):
    def __init__(self, input_size: int = 1024, hidden_size: int = 64, layers: int = 1, dropout: float = 0.0,
                 num_classes: Union[int, None] = 2, reduce_size: bool = True, act: str = "relu"):
        """
        Generate an MLP model.
        :param input_size: int
            The input size of the data.
        :param hidden_size: int
            The size of the hidden layers.
        :param layers: int
            Set the number of layers for the MLP model.
        :param dropout: float
            Set the dropout rate for the MLP model.
        :param num_classes: Union[int, None]
            Set the number of classes for the output layer.
        :param reduce_size: bool
            If set to true, each subsequent hidden layers will be halved.
        :param act: str
            Set teh non-linear activation function. Defaults to "relu".
        :param pooling:
        """
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.is_fitted = False
        self.criterion = None
        if num_classes == 1:
            num_classes = None
        self.classes = num_classes

        # set model type
        if isinstance(self.classes, int) and self.classes > 1:
            self.type = "Classification"
        elif self.classes == 1 or self.classes is None:
            self.type = "Regression"

        # set dimension of mlp layers
        self.model = self._build_mlp(input_size, hidden_size, layers, num_classes, dropout, reduce_size)

        # info for plot
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_precision = []
        self.test_precision = []
        self.mcc = []
        self.recall = []
        self.f1_score = []
        self.roc_auc = []

    def __call__(self, x, add_feats=None):
        # concatenate additional features
        if add_feats is not None:
            # convert to mx.array
            add_feats = mx.array(add_feats, dtype=x.dtype)
            # reshape to mx.array to match latent space shape
            if add_feats.ndim == 1:
                add_feats = add_feats.reshape(x.shape[0], -1)
            if add_feats.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x has {x.shape[0]}, add_feats has {add_feats.shape[0]}")
            x = mx.concatenate([x, add_feats], axis=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
            if self.dropout > 0:
                x = mlx.nn.Dropout(p=self.dropout)(x)
        x = self.final_mlp(x)

        return x

    def _build_mlp(self, input_dim, hidden_size, layers, num_classes, dropout, reduce_size):
        """Dynamically build mlp layer. Method shared between graph neural networks."""
        layer_num = 0
        dimension = input_dim
        # loop mlp layers
        self.mlp = []

        for _ in range(layers - 1):
            # optional, halve hidden size if reduce_size is True
            if reduce_size and layer_num > 0:
                hidden_size = round(hidden_size / 2)
            self.mlp.append(mlx.nn.Sequential(
                mlx.nn.Linear(dimension, hidden_size),
                mlx.nn.BatchNorm(hidden_size),
                mlx.nn.ReLU(),
                mlx.nn.Dropout(dropout)
            ))
            dimension = hidden_size
            layer_num += 1
        # final mlpl layer
        if num_classes is None:
            num_classes = 1
        self.final_mlp = mlx.nn.Linear(dimension, num_classes)

        # wrap into a sequential module
        return self.mlp, self.final_mlp


# shuffle and generate minibatches, for MLP model
def _batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.shape[0]))
    for s in range(0, y.shape[0], batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]


def train_mlp_c(model, x_train, y_train, optimizer, loss_and_grad_fn, batch_size):
    """Support function for training the MLX model"""
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    for X, y in _batch_iterate(batch_size, x_train, y_train):
        loss, grads = loss_and_grad_fn(model, X, y)

        # update params
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # get predictions from logits
        y_output = model(X)
        pred = mx.argmax(y_output, axis=1)
        correct += (pred == y).sum().item()
        total_samples += y.shape[0]

        # store predictions and labels
        all_preds.append(pred)
        all_labels.append(y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # true positives and false positives
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()

    # average loss per batch
    train_loss = total_loss / batch_count
    train_acc = correct / total_samples
    train_precision = tp / (tp + fp + 1e-8)

    return train_loss, train_acc, train_precision


def test_mlp_c(model, x_test, y_test, loss_fn, batch_size):
    """Support function for testing MLX model"""
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    batch_count = 0
    all_predictions = []
    all_proba = []
    all_labels = []

    for X, y in _batch_iterate(batch_size, x_test, y_test):
        # forward pass and loss calculation
        y_output = model(X)
        loss = loss_fn(model, X, y)
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # predictions
        predictions = mx.argmax(y_output, axis=1)
        proba = mx.softmax(y_output, axis=1)
        correct += (predictions == y).sum().item()
        total_samples += y.shape[0]

        # store for metrics calculation
        all_predictions.append(predictions)
        all_proba.append(proba)
        all_labels.append(y)

    # calculate metrics
    all_predictions = mx.concatenate(all_predictions, axis=0)
    all_proba = mx.concatenate(all_proba, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # true positives and false positives
    tp = ((all_predictions == 1) & (all_labels == 1)).sum().item()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum().item()

    # average loss per batch
    test_loss = total_loss / batch_count
    test_acc = correct / total_samples
    test_precision = tp / (tp + fp + 1e-8)

    # convert to numpy for sklearn metrics
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    all_proba_np = np.array(all_proba)
    # probability for positive class
    if all_proba_np.ndim > 1:
        all_proba_np = all_proba_np[:, 1]

    # calculate additional metrics
    mcc = matthews_corrcoef(all_labels_np, all_predictions_np)
    recall = recall_score(all_labels_np, all_predictions_np)
    f1 = f1_score(all_labels_np, all_predictions_np)
    roc_auc = roc_auc_score(all_labels_np, all_proba_np)

    return test_loss, test_acc, test_precision, mcc, recall, f1, roc_auc


def train_mlp_r(model, x_train, y_train, loss_and_grad_fn, optimizer, batch_size):
    """Support function for mlx regression model"""
    model.train()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    # for scaling, collect all y labels
    all_y_list = []
    for X, y in _batch_iterate(batch_size, x_train, y_train):
        all_y_list.append(y)

    # for data in x_train:
    #     all_y_list.append(data.y)
    all_y = mx.concatenate(all_y_list, axis=0) if all_y_list else None

    y_mean = all_y.mean()
    y_std = all_y.std()
    if y_std == 0: y_std = 1.0

    for X, y in _batch_iterate(batch_size, x_train, y_train):
        # forward pass
        # unit scale
        y_true_reshaped = mx.expand_dims(y, axis=-1)
        y_standardized = (y_true_reshaped - y_mean) / y_std

        loss, grads = loss_and_grad_fn(model, X, y_standardized)

        # Update params
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # get prediction
        y_output = model(X)
        pred = y_output

        all_preds.append(pred)
        all_labels.append(y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # unscale predictions
    all_preds = all_preds * y_std + y_mean
    # all_labels = all_labels * y_std + y_mean

    # average loss
    train_loss = total_loss / batch_count

    # mean absolute error
    train_mae = mean_absolute_error(all_labels, all_preds)

    # root mean squared error
    train_rmse = mean_squared_error(all_labels, all_preds)

    # r-squared
    train_r2 = r2_score(all_labels, all_preds)

    return train_loss, train_mae, train_rmse, train_r2


def test_mlp_r(model, x_test, y_test, criterion, batch_size, y_mean, y_std):
    """Support function for mlx regression model"""
    model.eval()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    # calculate loss on scaled units
    y_target_scaled = (y_test - y_mean) / y_std
    if y_target_scaled.ndim == 1:
        y_target_scaled = y_target_scaled.reshape(-1, 1)

    for X, y in _batch_iterate(batch_size, x_test, y_test):
        # forward pass
        y_output = model(X)

        loss = mx.mean(criterion(y_output, y_target_scaled))
        total_loss += loss.item()
        batch_count += 1

        # get predictions and accuracy calculation
        all_preds.append(y_output)
        all_labels.append(y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # unscale predictions for MAE/RMSE calculation
    if y_mean is not None and y_std is not None:
        all_preds = all_preds * y_std + y_mean

    # average loss
    test_loss = total_loss / batch_count

    # mean absolute error
    test_mae = mean_absolute_error(all_labels, all_preds)

    # root mean squared error
    test_rmse = mean_squared_error(all_labels, all_preds)

    # r-squared
    test_r2 = r2_score(all_labels, all_preds)

    return test_loss, test_mae, test_rmse, test_r2


def train_c(model, train_loader, loss_and_grad_fn, optimizer):
    """Support function for training mlx classification model"""
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    for data in train_loader:
        # forward pass
        (loss, y_output), grads = loss_and_grad_fn(model=model, graph=data, labels=data.y)

        # update params
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # get predictions from logits
        pred = y_output.argmax(axis=1)
        correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs

        # store predictions and labels
        all_preds.append(pred)
        all_labels.append(data.y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # true positives and false positives
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()

    # average loss per batch
    train_loss = total_loss / batch_count
    train_acc = correct / total_samples
    train_precision = tp / (tp + fp + 1e-8)

    return train_loss, train_acc, train_precision


def test_c(model, test_loader, criterion):
    """Support function for testing mlx classification model"""
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    batch_count = 0
    all_predictions = []
    all_proba = []
    all_labels = []

    for data in test_loader:
        # forward pass
        # check if edge_features used by model, checks using hte model.__call__
        sig = inspect.signature(model.__call__)
        if "edge_features" in sig.parameters and hasattr(data, "edge_features"):
            try:
                y_output = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices,
                                 data.add_feats)
            except AttributeError:
                y_output = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices)
        elif hasattr(data, "add_feats"):
            y_output = model(data.edge_index, data.node_features, data.batch_indices, data.add_feats)
        else:
            y_output = model(data.edge_index, data.node_features, data.batch_indices)

        # calculate loss
        loss = mx.mean(criterion(y_output, data.y))
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # get discrete predictions for accuracy calculation
        predictions = y_output.argmax(axis=1)
        proba = mx.softmax(y_output, axis=1)[:, 1]
        correct += (predictions == data.y).sum().item()
        total_samples += data.num_graphs

        # store for metrics calculation
        all_predictions.append(predictions)
        all_proba.append(proba)
        all_labels.append(data.y)

    # calculate metrics
    all_predictions = mx.concatenate(all_predictions, axis=0)
    all_proba = mx.concatenate(all_proba, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # true positives and false positives
    tp = ((all_predictions == 1) & (all_labels == 1)).sum().item()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum().item()

    # average loss per batch
    test_loss = total_loss / batch_count
    test_acc = correct / total_samples
    test_precision = tp / (tp + fp + 1e-8)

    # convert to numpy for sklearn metrics
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    all_proba_np = np.array(all_proba).squeeze()

    # calculate additional metrics
    mcc = matthews_corrcoef(all_labels_np, all_predictions_np)
    recall = recall_score(all_labels_np, all_predictions_np)
    f1 = f1_score(all_labels_np, all_predictions_np)
    roc_auc = roc_auc_score(all_labels_np, all_proba_np)

    return test_loss, test_acc, test_precision, mcc, recall, f1, roc_auc


def train_r(model, train_loader, loss_and_grad_fn, optimizer, y_mean, y_std):
    """Support function for mlx regression model"""
    model.train()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    for data in train_loader:
        # forward pass
        # Reshape targets from (N,) to (N, 1) to match the model output (N, 1)
        y_true_reshaped = mx.expand_dims(data.y, axis=-1)
        y_standardized = (y_true_reshaped - y_mean) / y_std

        # Pass the reshaped target to the loss function wrapper
        (loss, y_output), grads = loss_and_grad_fn(model=model, graph=data, labels=y_standardized)

        # Update params
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        batch_count += 1  # to count batches

        # get prediction
        pred = y_output

        all_preds.append(pred)
        all_labels.append(data.y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # unscale predictions
    all_preds = all_preds * y_std + y_mean
    # all_labels = all_labels * y_std + y_mean

    # average loss
    train_loss = total_loss / batch_count

    # mean absolute error
    train_mae = mean_absolute_error(all_labels, all_preds)

    # root mean squared error
    train_rmse = mean_squared_error(all_labels, all_preds)

    # r-squared
    train_r2 = r2_score(all_labels, all_preds)

    return train_loss, train_mae, train_rmse, train_r2


def test_r(model, test_loader, criterion, y_mean, y_std):
    """Support function for mlx regression model"""
    model.eval()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    for data in test_loader:
        # forward pass
        sig = inspect.signature(model.__call__)
        if "edge_features" in sig.parameters and hasattr(data, "edge_features"):
            try:
                y_output = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices,
                                 data.add_feats)
            except AttributeError:
                y_output = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices)
        elif hasattr(data, "add_feats"):
            y_output = model(data.edge_index, data.node_features, data.batch_indices, data.add_feats)
        else:
            y_output = model(data.edge_index, data.node_features, data.batch_indices)

        # calculate loss on scaled units
        data_y_reshaped = mx.expand_dims(data.y, axis=-1)
        y_target_scaled = (data_y_reshaped - y_mean) / y_std
        loss = mx.mean(criterion(y_output, y_target_scaled))
        total_loss += loss.item()
        batch_count += 1

        # get predictions and accuracy calculation
        all_preds.append(y_output)
        all_labels.append(data.y)

    # calculate metrics
    all_preds = mx.concatenate(all_preds, axis=0)
    all_labels = mx.concatenate(all_labels, axis=0)

    # unscale predictions for MAE/RMSE calculation
    if y_mean is not None and y_std is not None:
        all_preds = all_preds * y_std + y_mean
        # all_labels = all_labels * y_std + y_mean

    # average loss
    test_loss = total_loss / batch_count

    # mean absolute error
    test_mae = mean_absolute_error(all_labels, all_preds)

    # root mean squared error
    test_rmse = mean_squared_error(all_labels, all_preds)

    # r-squared
    test_r2 = r2_score(all_labels, all_preds)

    return test_loss, test_mae, test_rmse, test_r2


def _global_pooling(x, batch_indices, pool_type='add'):
    if pool_type == 'add':
        return global_add_pool(x, batch_indices)
    elif pool_type == 'mean':
        return global_mean_pool(x, batch_indices)
    elif pool_type == 'max':
        return global_max_pool(x, batch_indices)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


# support function to move torch like tensor into mlx tensor format
def _torch_to_mlx(graph):
    add_feats = 0  # if add_feats are used
    mlx_dict = {}  # to hold arrays

    # loop through tensor and convert to mx array
    for attr in graph.keys():
        attr_value = getattr(graph, attr)
        if attr_value is not None:
            # Convert to float32 by default, except edge_index
            if attr == 'edge_index':
                mlx_dict[attr] = mx.array(attr_value, dtype=mx.int32)
            else:
                mlx_dict[attr] = mx.array(attr_value, dtype=mx.float32)
            if attr == 'add_feats':
                mlx_dict[attr] = mx.array(attr_value, dtype=mx.float32)
                add_feats = 1
            else:
                pass

    graph_args = {
        "edge_index": mlx_dict["edge_index"],
        "node_features": mlx_dict["x"],
        "edge_features": mlx_dict["edge_attr"],
    }

    if "y" in mlx_dict:
        graph_args["y"] = mlx_dict["y"]

    if add_feats == 1 and "add_feats" in mlx_dict:
        graph_args["add_feats"] = mlx_dict["add_feats"]

    graph_mlx = GraphData(**graph_args)

    return graph_mlx


# support function to set criterion
def _set_criterion(criterion, task_type, num_classes):
    if criterion is None:
        raise ValueError("No criterion defined!")

    # define by model type
    if task_type == "Classification":
        pulled_criterion = MLX_CLASSIFICATION
    elif task_type == "Regression":
        pulled_criterion = MLX_REGRESSION
    else:
        raise ValueError("Task type must be 'Classification' or 'Regression'.")

    # get loss function
    if isinstance(criterion, str):
        if criterion not in pulled_criterion:
            raise ValueError(
                f"Criterion '{criterion}' not supported for task type '{task_type}'. Check 'num_classes' or 'criterion' params!")
        criterion = pulled_criterion[criterion]

    return criterion


if __name__ == "__main__":
    import doctest

    doctest.testmod()
