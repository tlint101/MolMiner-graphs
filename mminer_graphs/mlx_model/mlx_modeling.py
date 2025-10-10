import inspect
import copy
import json
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mlx_graphs.data import GraphData
from mlx_graphs.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from mlx.utils import tree_flatten, tree_unflatten
from typing import Optional, Union
from tqdm import tqdm
from mminer_graphs.model.model_utils import mol_to_graph
from mminer_graphs.mlx_models.mlx_utils import get_device, MLX_OPTI, MLX_ACT, MLX_CRITERIONS, NO_WT_DECAY
import numpy as np
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, roc_auc_score


class MolToGraph:
    def __init__(self, smi: str, label: Optional[Union[int, float]] = None, graph_type: str = "AtomGraph",
                 add_smi: bool = False, add_feats=None, with_atom_index: bool = False):
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


class Base(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = None
        self.initial_params = None
        # info for plot
        self.metrics = Metrics()

    def fit(self, train_loader: Union[Dataloader, list], test_loader: Union[Dataloader, list], epochs: int = 10,
            optimizer: str = "adam", criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0,
            batch_size: int = 1024, device: str = "auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar=True,
            verbose: bool = False, device_warn: bool = True,
            **kwargs):
        """
        Fit the model.
        :param model: mlx.nn.Module
            The MLX model to fit.
        :param train_loader: Union[mlx_graphs.loaders.Dataloader, list]
            The data loader for the training data.
        :param test_loader: Union[mlx_graphs.loaders.Dataloader, list]
            The data loader for the test data.
        :param epochs: int
            Set the number of training epochs.
        :param optimizer: str
            Determine the model optimizer.
        :param criterion: str
            Set the model loss function.
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
        :param kwargs: dict
            kwargs for a given criterion.
        :return:
        """
        # set model to current instance
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
        if criterion in MLX_CRITERIONS.keys():
            criterion = MLX_CRITERIONS[criterion]
        elif criterion is None:
            raise ValueError("No Criterion defined!")
        else:
            raise ValueError(f"Criterion not recognized! Only {list(MLX_CRITERIONS.keys())} allowed!")
        self.criterion = criterion

        # check dataset input:
        np.random.seed(42)  # try to enforce reproducibility
        if isinstance(train_loader, list):
            train_loader = Dataloader(train_loader, batch_size=batch_size,
                                      shuffle=True)  # avoid overfitting to sample order
        if isinstance(test_loader, list):
            test_loader = Dataloader(test_loader, batch_size=batch_size)

        if device_warn:
            print(f"Model running on {fit_device}")

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        # set loss and gradient function
        loss_and_grad_fn = mlx.nn.value_and_grad(model, self._forward_fn)

        mx.eval(model.parameters())

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            train_loss, train_acc, train_prec = train(model, train_loader, loss_and_grad_fn, optimizer)
            # train_acc, train_prec = train_metrics(model, train_loader, loss_and_grad_fn, optimizer, criterion)
            test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test(model, test_loader, criterion)

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

        # var for cross validation
        self.is_fitted = True

        if device_warn:
            print(f"Average Training Loss: {round(np.mean(self.metrics.train_loss), 3)}")
            print(f"Average Test Loss: {round(np.mean(self.metrics.test_loss), 3)}")
            print(f"Average Training Accuracy: {round(np.mean(self.metrics.train_accuracy), 3)}")
            print(f"Average Test Accuracy: {round(np.mean(self.metrics.test_accuracy), 3)}")

        # # Optional, this will print the model and its layers
        # # return instance
        # return self

    def predict(self, data: Union[np.array, list, Dataloader], batch_size:int=None, binary: Union[bool, float, str] = True,
                return_type: str = "list"):
        """
        Handle predictions for an input MLX Dataloader.
        :param data: Union[np.array, list, Dataloader]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param batch_size: int
            Set the batch size for prediction. Only used if the input data is of type list.
        :param binary: Union[bool, float, str]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. If a float is given, then a binary predictions will be given with the
            input float used as the cutoff for positive (1) labels. Input strings can be "multi" or "proba", where
            "multi" will output a multi-class labels and "proba" will output a binary probability vector.
        :param return_type: str
            Set the return type of the prediction. Can be "array" or "list".
        :return:
        """
        # # check if on the same device
        # fit_device = get_device(device)
        # print(f"Prediction running on {fit_device}")
        # set model to current instance
        model = self

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

        for x in data:
            # forward pass
            # check if edge_features used by model, checks using hte model.__call__
            sig = inspect.signature(model.__call__)
            kwargs = {"edge_index": x.edge_index, "x": x.node_features, "batch_indices": x.batch_indices}
            if "edge_features" in sig.parameters and hasattr(x, "edge_features"):
                kwargs["edge_features"] = x.edge_features
            if "add_feats" in sig.parameters and hasattr(x, "add_feats"):
                kwargs["add_feats"] = x.add_feats

            output = model(**kwargs)

            # set for binary, multi-class, or probability output
            if binary == True:
                proba = mx.softmax(output, axis=1)[:, 1]
                proba = (proba >= 0.5).astype(mx.int32)
            # multi-class classification
            elif binary == False or binary == "multi":
                proba = mx.softmax(output, axis=1)
                proba = mx.argmax(proba, axis=1)
            # output probabilities
            elif binary == "proba":
                proba = mx.softmax(output, axis=1)
            elif isinstance(binary, float):
                proba = mx.softmax(output, axis=1)[:, 1]
            else:
                raise ValueError("Only 'multi', 'proba', boolean, or float is supported!!")
            all_preds.append(proba)

        # extract mx array from all_pred list
        predictions = all_preds[0]

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

    # todo add documentation
    def load(self, model_path: str = None, param_path: Optional[str] = None, add_feats: Optional[np.ndarray] = None,
             node_dim: int = 142, edge_feat_dim: int = 16, num_classes: int = 2, criterion: str = "cross_entropy", ):
        """
        Load a saved MLX model.
        :param filename: str
            The filepath for the saved MLX model. Must be in '.npz' format.
        :return:
        """
        model = self

        if param_path:
            # load params from json file
            with open(param_path, 'r') as f:
                params = json.load(f)

            # additional params not found in json file
            params['node_dim'] = node_dim
            params['edge_feat_dim'] = edge_feat_dim
            params['num_classes'] = num_classes
            params['criterion'] = criterion

            # load params by model
            if model.__class__ == GCNNet:
                for_model_params = ["node_dim", "hidden_dim", "graph_layers", "mlp_layers", "graph_dropout",
                                    "mlp_dropout", "add_feats"]
            elif model.__class__ == GINENet:
                for_model_params = ["node_dim", "edge_feat_dim", "hidden_dim", "graph_layers", "mlp_layers",
                                    "mlp_dropout", "add_feats"]
            else:
                raise ValueError("Cannot identify model class! Only 'GCNNet' or 'GINENet' supported!")
            model_params = {param_key: params[param_key] for param_key in for_model_params}

            # # to check loading of model
            # print(model_params)

            # build model, an instance of GNN
            self.is_fitted = True
            params = mx.load(model_path)
            self.update(tree_unflatten(list(params.items())))

            # load saved model
            print("Model loaded!")
        else:

            self.is_fitted = True
            params = mx.load(model_path)
            self.update(tree_unflatten(list(params.items())))

            # load saved model
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

    def _loss_fn(self, y_hat, y, parameters=None):
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
        loss = self._loss_fn(y_output, labels, model.parameters())
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
        self.final_mlp = mlx.nn.Linear(dimension, num_classes)

        return self.mlp, self.final_mlp


# Graph Convolutional Neural Network
class GCNNet(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, mlp_layers: int = 1,
                 reduce_graph: bool = False, graph_dropout: float = 0.0, mlp_dropout: float = 0.0, num_classes: int = 2,
                 act: str = "relu", add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """
        Generate graph model for graph convolutional (GCNC) network. The default model will generate two graph layers
        and one mlp layer.
        :param node_dim: int
            Length of feature vector per node.
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
        :param num_classes: int
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

        # set graph layers
        self.gcn_conv = []

        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            self.gcn_conv.append(GCNConv(in_dim, hidden_dim))

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
        for conv in self.gcn_conv:
            x = act_fn(conv(edge_index, x))
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


# todo confirm parity to torch models
class GINENet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: int = 16, hidden_dim: int = 64, graph_layers: int = 2,
                 mlp_layers: int = 1, reduce_graph: bool = False, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 num_classes: int = 2, act: str = "relu", add_feats: Optional[Union[np.array, list, int]] = None,
                 pooling: str = "add"):
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
        :param num_classes: int
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

        # set graph layers
        self.gine_conv = []

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
        for conv in self.gine_conv:
            x = act_fn(conv(edge_index, x, edge_features))
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
                 graph_dropout: float = 0.0, mlp_dropout: float = 0.0, num_classes: int = 2,
                 act: str = "relu", reduce_graph=False, add_feats: Optional[Union[np.array, list, int]] = None,
                 pooling: str = "mean"):
        global graph_hidden_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False

        # set graph layers
        self.sage_conv = []

        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            self.sage_conv.append(SAGEConv(in_dim, hidden_dim))

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
        for conv in self.sage_conv:
            x = act_fn(conv(edge_index, x))
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
                 num_classes: int = 2, act: str = "relu", reduce_graph=False,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        global reduced_graph_dim, out_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.mlp_dropout = mlp_dropout
        self.is_fitted = False

        # set graph layers
        self.gat_conv = []

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.gat_conv.append(GATConv(node_features_dim=in_channels, out_features_dim=graph_hidden_dim,
                                             heads=heads, edge_features_dim=edge_feat_dim, dropout=graph_dropout))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.gat_conv.append(GATConv(node_features_dim=in_channels, out_features_dim=graph_hidden_dim,
                                             heads=heads, edge_features_dim=edge_feat_dim, dropout=graph_dropout))
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
        for conv in self.gat_conv:
            # passes based on edge_features dimensions
            if hasattr(conv, 'edge_lin_proj') and edge_features is not None:
                x = act_fn(conv(edge_index, x, edge_features))
            else:
                x = act_fn(conv(edge_index, x))

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
        self.model = None
        self.is_fitted = False
        self.initial_params = None
        # info for plot
        self.metrics = Metrics()

    def fit(self, x_train: Union[list, np.ndarray], y_train: Union[list, np.ndarray], x_test: Union[list, np.ndarray],
            y_test: Union[list, np.ndarray], epochs: int = 10, optimizer: str = "adam",
            criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0, batch_size: int = 1024,
            device: str = "auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar: bool = True,
            verbose: bool = False, device_warn: bool = True, **kwargs):
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
        # set model
        model = self.model

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
        if criterion in MLX_CRITERIONS.keys():
            criterion = MLX_CRITERIONS[criterion]
        elif criterion is None:
            raise ValueError("No Criterion defined!")
        else:
            raise ValueError(f"Criterion not recognized! Only {list(MLX_CRITERIONS.keys())} allowed!")
        self.criterion = criterion

        # check dataset input:
        np.random.seed(42)  # try to enforce reproducibility
        if device_warn:
            print(f"Model running on {fit_device}")

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

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            # pass MLP class instance (self)
            train_loss, train_acc, train_prec = train_mlp(model, x_train, y_train, optimizer, loss_and_grad_fn,
                                                          batch_size)
            test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test_mlp(model, x_test, y_test, self._loss_fn,
                                                                                batch_size)

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

        # var for cross validation
        self.is_fitted = True

        if device_warn:
            print(f"Average Training Loss: {round(np.mean(self.metrics.train_loss), 3)}")
            print(f"Average Test Loss: {round(np.mean(self.metrics.test_loss), 3)}")
            print(f"Average Training Accuracy: {round(np.mean(self.metrics.train_accuracy), 3)}")
            print(f"Average Test Accuracy: {round(np.mean(self.metrics.test_accuracy), 3)}")

    def predict(self, data: Union[np.array, list, mx.array], binary: Union[bool, float, str] = True,
                return_type: str = "list"):
        """
        Handle predictions for an input MLX Dataloader.
        :param data: Union[np.array, list]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param binary: Union[bool, float, str]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. If a float is given, then a binary predictions will be given with the
            input float used as the cutoff for positive (1) labels. Input strings can be "multi" or "proba", where
            "multi" will output a multi-class labels and "proba" will output a binary probability vector.
        :param return_type: str
            Set the return type of the prediction. Can be "array" or "list".
        :return:
        """
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

        # # make sure data is 2D
        # if data.ndim == 1:
        #     data = mx.expand_dims(data, axis=0)

        # evaluation mode
        self.model.eval()

        output = self.model(data)

        # set for binary, multi-class, or probability output
        if binary == True:
            proba = mx.softmax(output, axis=1)[:, 1]
            proba = (proba >= 0.5).astype(mx.int32)
        # multi-class classification
        elif binary == False or binary == "multi":
            proba = mx.softmax(output, axis=1)
            proba = mx.argmax(proba, axis=1)
        # output probabilities
        elif binary == "proba":
            proba = mx.softmax(output, axis=1)
        elif isinstance(binary, float):
            proba = mx.softmax(output, axis=1)[:, 1]
        else:
            raise ValueError("Only 'multi', 'proba', boolean, or float is supported!!")
        all_preds.append(proba)

        # extract mx array from all_pred list
        predictions = all_preds[0]

        # conditional check to output as list or tensor
        if return_type == "list":
            return predictions.tolist()
        elif return_type == "array":
            return predictions
        else:
            raise TypeError(f"Type {return_type} not recognized!")

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
                 num_classes: int = 2, reduce_size: bool = True, act: str = "relu"):
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
        :param num_classes: int
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

    def __call__(self, x):
        for layer in self.model:
            x = layer(x)

        return x

    @staticmethod
    def _build_mlp(input_dim, hidden_size, layers, num_classes, dropout, reduce_size):
        """Dynamically build mlp layer. Method shared between graph neural networks."""
        layer_num = 0
        dimension = input_dim  # graph latent space
        # loop mlp layers
        model_layers = []

        for _ in range(layers - 1):
            # optional, halve hidden size if reduce_size is True
            if reduce_size and layer_num > 0:
                hidden_size = round(hidden_size / 2)
            model_layers.append(mlx.nn.Sequential(
                mlx.nn.Linear(dimension, hidden_size),
                mlx.nn.BatchNorm(hidden_size),
                mlx.nn.ReLU(),
                mlx.nn.Dropout(dropout)
            ))
            dimension = hidden_size
            layer_num += 1
        # final mlpl layer
        model_layers.append(mlx.nn.Linear(dimension, num_classes))

        # wrap into a sequential module
        return mlx.nn.Sequential(*model_layers)


# shuffle and generate minibatches, for MLP model
def _batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.shape[0]))
    for s in range(0, y.shape[0], batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]


def train_mlp(model, x_train, y_train, optimizer, loss_and_grad_fn, batch_size):
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


def test_mlp(model, x_test, y_test, loss_fn, batch_size):
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


def train(model, train_loader, loss_and_grad_fn, optimizer):
    """Support function for ValueError: y should be a 1d array, got an array of shape (137, 2) instead.training the mlx model"""
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


def test(model, test_loader, criterion):
    """Support function for testing the mlx model"""
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

    if add_feats == 1:
        graph_mlx = GraphData(edge_index=mlx_dict['edge_index'], node_features=mlx_dict['x'],
                              edge_features=mlx_dict['edge_attr'], y=mlx_dict['y'],
                              add_feats=mlx_dict['add_feats'])
    else:
        graph_mlx = GraphData(edge_index=mlx_dict['edge_index'], node_features=mlx_dict['x'],
                              edge_features=mlx_dict['edge_attr'], y=mlx_dict['y'])
    return graph_mlx


if __name__ == "__main__":
    import doctest

    doctest.testmod()
