import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.nn import GCNConv, GINEConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import summary as sum
from torch_geometric.loader import DataLoader
from torchinfo import summary as torchinfo_summary
from torch_geometric.data import Data
from mminer_graphs.model.model_utils import set_seed, mol_to_graph, get_device, OPTIMIZERS, CRITERIONS, F_ACT
from tqdm import tqdm
from typing import Optional, Union, Type
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, roc_auc_score


def summary(model: nn.Module, data: Optional[Union[DataLoader, list]] = None, batch_size: Optional[int] = None,
            input_size: Optional[tuple] = None, **kwargs):
    """
    Summarize the input nn.Module in a tabular format. If using graph models, input_size is not needed. If using MLP,
    then input_size is needed, but data is not needed.
    :param model: nn.Module
        The input torch model.
    :param data: Union[DataLoader, list]
        The data for training the model. Information must be loaded in class 'DataLoader'. If a 'list' is given, it will
        be internally loaded to the DataLoader.
    :param batch_size: int
        Only needed if the input data is of class 'list'.Needed to set the batch_size of the class 'DataLoader'.
    :param input_size: tuple
        Shape of input of tabular data.
    :param kwargs: dict
        Additional arguments passed to torchinfo.
    :return:
    """

    # for MLP model
    if input_size is not None:
        return torchinfo_summary(model, input_size=input_size, batch_size=batch_size, **kwargs)
    else:
        # for graph models
        # check dataset input:
        if isinstance(data, list):
            if batch_size is None:
                raise ValueError("batch_size needed if data is type 'list'!!")
            data = DataLoader(data, batch_size=batch_size, shuffle=True, worker_init_fn=set_seed(42))
        if not isinstance(data, DataLoader):
            raise TypeError("Issue with input data. Must be a list or a DataLoader instance!")

        # get data elements from iterable
        data_input = next(iter(data))

        # return tabular string
        return sum(model=model, data=data_input)


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
            Whether to include the smiles string with the torch Data object.
        :param add_feats: Optional[Union[np.array, list]]
            Include additional information to include with the molecular graph tensor. Must be given as a np.array or a
            list.
        :param with_atom_index: bool
            To include the smi string with atom index. Defaults to False.
        """
        self.graph_data = mol_to_graph(smi=smi, label=label, graph_type=graph_type, add_smi=add_smi,
                                       add_feats=add_feats, with_atom_index=with_atom_index)


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


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = None

    def fit(self, train_loader: Union[DataLoader, list], test_loader: Union[DataLoader, list], epochs: int = 10,
            optimizer: str = "adam", criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0,
            batch_size: int = 1024, device: str = "auto", estop: Optional[Union[EarlyStopping, int]] = None,
            pbar: bool = True, verbose: bool = False, device_warn: bool = True, **kwargs):
        """
        Fit the model.
        :param train_loader: Union[torch_geometric.loader.DataLoader, list]
            The data loader for the training data.
        :param test_loader: torch_geometric.loader.DataLoader
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
            is a list instead of a DataLoader.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param estop: Optional[Union[EarlyStopping, int]]
            Trigger EarlyStopping during model fitting. Can manually call the estop criteria, or call an int to
            designate the patience number. If an int is given, the early stopping will be triggered based on the
            smallest test loss for each epoch.
        :param pbar: bool
            Whether to show a progress bar during model fit for each epoch.
        :param verbose: bool
            Determine whether to output train/test accuracy and loss for each epoch.
        :param device_warn: bool
            Option to print which tensor/model is on 'cpu', 'cuda', or 'mps'.
        :param kwargs: dict
            kwargs for a given criterion.
        :return:
        """
        # set model to current instance
        model = self

        # check if on the same device
        fit_device = get_device(device)
        self.to(fit_device)

        # set optimizer and criterion
        if optimizer in OPTIMIZERS.keys():
            optimizer = OPTIMIZERS[optimizer](model.parameters(), lr=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! Only {list(OPTIMIZERS.keys())} allowed!")

        if criterion in CRITERIONS.keys():
            criterion = CRITERIONS[criterion](**kwargs)
        elif criterion is None:
            raise ValueError("No Criterion defined!")
        else:
            raise ValueError(f"Criterion not recognized! Only {list(CRITERIONS.keys())} allowed!")

        # check dataset input:
        if isinstance(train_loader, list):
            train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=set_seed(42))  # avoid overfitting to sample order
        if isinstance(test_loader, list):
            test_loader = DataLoader(test_loader, batch_size=batch_size)

        if device_warn:
            print(f"Model and Tensor running on {fit_device}")

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            train_loss, train_acc, train_prec = train(model, train_loader, optimizer, criterion, fit_device)
            test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test(model, test_loader, criterion, fit_device)

            # add metrics
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.train_accuracy.append(train_acc)
            self.test_accuracy.append(test_acc)
            self.train_precision.append(train_prec)
            self.test_precision.append(test_prec)
            self.mcc.append(mcc)
            self.recall.append(recall)
            self.f1_score.append(f1)
            self.roc_auc.append(roc_auc)

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
            print(f"Average Training Loss: {round(np.mean(self.train_loss), 3)}")
            print(f"Average Test Loss: {round(np.mean(self.test_loss), 3)}")
            print(f"Average Training Accuracy: {round(np.mean(self.train_accuracy), 3)}")
            print(f"Average Test Accuracy: {round(np.mean(self.test_accuracy), 3)}")

        # # Optional, this will print the model and its layers
        # # return instance
        # return self

    def predict(self, data: Union[np.array, list, DataLoader], device: str = "auto",
                binary: Union[bool, float, str] = True, return_type: str = "list"):
        """
        Handle predictions for an input torch Dataloader.
        :param model: torch.nn.Module
            A trained torch model.
        :param data: Union[np.array, list, DataLoader]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param binary: Union[bool, float, str]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. If a float is given, then a binary predictions will be given with the
            input float used as the cutoff for positive (1) labels. Input strings can be "multi" or "proba", where
            "multi" will output a multi-class labels and "proba" will output a binary probability vector.
        :param return_type: str
            Set the return type of the prediction. Can be "tensor" or "list".
        :return:
        """
        # set model to current instance
        model = self

        # check if on the same device
        fit_device = get_device(device)
        print(f"Prediction running on {fit_device}")

        # evaluation mode
        model.eval()
        all_preds = []
        with torch.no_grad():  # disable gradient calculation
            for x in data:
                datapoint = x.to(fit_device)
                output = model(datapoint)
                # set for binary, multi-class, or probability output
                if binary == True or isinstance(binary, float):
                    proba = F.softmax(output, dim=1)[:, 1]
                    if isinstance(binary, float):
                        proba = (proba > binary).long()
                    else:
                        proba = (proba >= 0.5).long()
                # multi-class classification
                elif binary == False or binary == "multi":
                    proba = F.softmax(output, dim=1)
                    proba = torch.argmax(proba, dim=1)
                # output probabilities
                elif binary == "proba":
                    proba = F.softmax(output, dim=1)
                else:
                    raise ValueError("Only 'multi', 'proba', boolean, or float is supported!!")
                all_preds.append(proba)

            # conditional check to output as list or tensor
            if return_type == "list":
                prediction = torch.cat(all_preds)
                return prediction.tolist()
            elif return_type == "tensor":
                return torch.cat(all_preds)
            else:
                raise TypeError(f"Type {return_type} not recognized!")

    def reset_params(self):
        """
        reset the layers in GCNConvNet
        :return:
        """

        # graph layers
        conv_layer_names = ["gcn_conv", "gine_conv", "sage_conv", "gat_conv"]
        conv_layers = None
        for name in conv_layer_names:
            if hasattr(self, name):
                conv_layers = getattr(self, name)
                break

        if conv_layers is None:
            raise ValueError("No defined conv layers found!")

        # mlp layers
        for x in self.mlp:
            for layer in x:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # final mlp layer
        if hasattr(self.final_mlp, 'reset_parameters'):
            self.final_mlp.reset_parameters()

    def save(self, filepath: str):
        """
        Save a fitted torch model.
        :param filepath: str
            The filepath for the saved torch model. Must be in .pth format.
        :return:
        """
        torch.save(self.state_dict(), filepath)
        return "Model saved!"

    def load(self, model_path: str = None, param_path: Optional[str] = None, add_feats: Optional[np.ndarray] = None,
             node_dim: int = 142, edge_feat_dim: int = 16, num_classes: int = 2, criterion: str = "cross_entropy",
             device: str = "auto"):
        """
        Load a torch model from a .pth file.
        :param param_path: str
            Path to model parameters in json format.
        :param model_path: str
            The filepath for the saved torch model. Must be in .pth format.
        :param add_feats: np.ndarray
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
            params['device'] = device

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
            build_model = model.__class__(**model_params)

            # load saved model
            build_model.load_state_dict(torch.load(model_path))
            print("Model loaded!")

            return build_model
        else:
            model = self
            # load model and set to eval mode
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            return model

    def _build_mlp(self, input_dim, mlp_layers, num_classes, dropout):
        """Dynamically build mlp layer. Method shared between graph neural networks."""
        dimension = input_dim  # graph latent space
        # loop mlp layers
        self.mlp = nn.ModuleList()
        for _ in range(mlp_layers - 1):
            reduced_dimension = round(dimension / 2)
            self.mlp.append(nn.Sequential(
                nn.Linear(dimension, reduced_dimension),
                nn.BatchNorm1d(reduced_dimension),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            dimension = reduced_dimension
        # final mlpl layer
        self.final_mlp = nn.Linear(dimension, num_classes)

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
        self.is_fitted = False

        # set graph layers
        self.gcn_conv = nn.ModuleList()

        # set graph layers
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # graph layers
        act_fn = F_ACT[self.act]
        for conv in self.gcn_conv:
            x = act_fn(conv(x, edge_index))
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            add_feats = data.add_feats
            if not isinstance(add_feats, torch.Tensor):
                add_feats = torch.tensor(add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = add_feats.to(device=x.device, dtype=x.dtype)

            # ensure add_feats is 2D
            if add_feats.dim() == 1:
                add_feats = add_feats.unsqueeze(0)  # [1, features]

            # match batch size if needed
            if add_feats.size(0) == 1 and x.size(0) > 1:
                add_feats = add_feats.expand(x.size(0), -1)
            elif add_feats.size(0) != x.size(0):
                raise ValueError(f"add_feats batch size {add_feats.size(0)} does not match x batch size {x.size(0)}")

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# Graph Isomorphism Network with Edge Features
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
        self.is_fitted = False

        # set graph layers
        self.gine_conv = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            nnet = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # normalize node embeddings
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            self.gine_conv.append(GINEConv(nn=nnet, edge_dim=edge_feat_dim))

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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # graph layers
        act_fn = F_ACT[self.act]
        for conv in self.gine_conv:
            x = act_fn(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            add_feats = data.add_feats
            if not isinstance(add_feats, torch.Tensor):
                add_feats = torch.tensor(add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = add_feats.to(device=x.device, dtype=x.dtype)

            # ensure add_feats is 2D
            if add_feats.dim() == 1:
                add_feats = add_feats.unsqueeze(0)  # [1, features]

            # match batch size if needed
            if add_feats.size(0) == 1 and x.size(0) > 1:
                add_feats = add_feats.expand(x.size(0), -1)
            elif add_feats.size(0) != x.size(0):
                raise ValueError(f"add_feats batch size {add_feats.size(0)} does not match x batch size {x.size(0)}")

            x = torch.cat([x, add_feats], dim=1)

        # build mlp layer
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# GraphSAGE
class GSAGENet(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, mlp_layers: int = 1,
                 graph_dropout: float = 0.0, mlp_dropout: float = 0.0, num_classes: int = 2,
                 act: str = "relu", reduce_graph=False, add_feats: Optional[Union[np.array, list, int]] = None,
                 pooling: str = "mean"):
        """
        Generate a graph SAGE model neural network (GSAGENet). The default model will generate two graph layers and one
        mlp layer.
        :param node_dim: int
            Length of node feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param graph_dropout: float
            Set the graph dropout during activation.
        :param mlp_dropout: float
            Set the dropout for each MLP layer.
        :param num_classes: int
            Set the number of output classes
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
        self.is_fitted = False

        # set graph layers
        self.sage_conv = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.sage_conv.append(
                    SAGEConv(in_channels=in_channels, out_channels=graph_hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.sage_conv.append(
                    SAGEConv(in_channels=in_channels, out_channels=graph_hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels for loop

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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # check activation function
        if self.act not in F_ACT:
            raise ValueError(f"Unsupported activation function {self.act}"
                             f"Only {list(F_ACT.keys())} are supported!")

        # graph layers
        act_fn = F_ACT[self.act]
        for conv in self.sage_conv:
            x = act_fn(conv(x, edge_index))
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            add_feats = data.add_feats
            if not isinstance(add_feats, torch.Tensor):
                add_feats = torch.tensor(add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = add_feats.to(device=x.device, dtype=x.dtype)

            # ensure add_feats is 2D
            if add_feats.dim() == 1:
                add_feats = add_feats.unsqueeze(0)  # [1, features]

            # match batch size if needed
            if add_feats.size(0) == 1 and x.size(0) > 1:
                add_feats = add_feats.expand(x.size(0), -1)
            elif add_feats.size(0) != x.size(0):
                raise ValueError(f"add_feats batch size {add_feats.size(0)} does not match x batch size {x.size(0)}")

        # build mlp layer
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# Graph Attention Network
class GATNet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: Optional[int] = None, hidden_dim: int = 128, heads: int = 1,
                 graph_layers: int = 2, mlp_layers: int = 1, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 num_classes: int = 2, act: str = "relu", reduce_graph=False,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """

        :param node_dim: int
            Length of node feature vector.
        :param edge_feat_dim: Optional[int]
            Length of edge feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param heads: int
            Set the number of attention mechanisms to be used in parallel.
        :param graph_layers: int
            Set the number of graph layers
        :param mlp_layers: int
            Set the number of MLP layers.
        :param graph_dropout: float
            Set the dropout for the attention coefficients and during activation.
        :param mlp_dropout: float
            Set the dropout layer for each MLP layer.
        :param num_classes: int
            Set the number of output classes.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        super().__init__()
        self.act = act
        self.pooling = pooling
        self.edge_feat_dim = edge_feat_dim
        self.graph_dropout = graph_dropout
        self.is_fitted = False

        # set graph layers
        self.gat_conv = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.gat_conv.append(GATConv(in_channels=in_channels, out_channels=graph_hidden_dim, heads=heads,
                                             edge_dim=edge_feat_dim, dropout=graph_dropout))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.gat_conv.append(GATConv(in_channels=in_channels, out_channels=graph_hidden_dim, heads=heads,
                                             edge_dim=edge_feat_dim, dropout=graph_dropout))
                in_channels = graph_hidden_dim  # reset in_channels for loop

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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_feat_dim is not None:
            edge_attr = data.edge_attr
        else:
            edge_attr = None

        # check activation function
        if self.act not in F_ACT:
            raise ValueError(f"Unsupported activation function {self.act}"
                             f"Only {list(F_ACT.keys())} are supported!")

        # graph layers
        act_fn = F_ACT[self.act]
        for conv in self.gat_conv:
            x = act_fn(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            add_feats = data.add_feats
            if not isinstance(add_feats, torch.Tensor):
                add_feats = torch.tensor(add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = add_feats.to(device=x.device, dtype=x.dtype)

            # ensure add_feats is 2D
            if add_feats.dim() == 1:
                add_feats = add_feats.unsqueeze(0)  # [1, features]

            # match batch size if needed
            if add_feats.size(0) == 1 and x.size(0) > 1:
                add_feats = add_feats.expand(x.size(0), -1)
            elif add_feats.size(0) != x.size(0):
                raise ValueError(f"add_feats batch size {add_feats.size(0)} does not match x batch size {x.size(0)}")

        # build mlp layer
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# Graph Isomorphism Network
class GINNet(nn.Module):
    pass


class SplineConvNet(nn.Module):
    pass


class MLP_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.is_fitted = None

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
        self.to(fit_device)

        # set optimizer and criterion
        if optimizer in OPTIMIZERS.keys():
            optimizer = OPTIMIZERS[optimizer](model.parameters(), lr=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! Only {list(OPTIMIZERS.keys())} allowed!")

        if criterion in CRITERIONS.keys():
            criterion = CRITERIONS[criterion](**kwargs)
        elif criterion is None:
            raise ValueError("No Criterion defined!")
        else:
            raise ValueError(f"Criterion not recognized! Only {list(CRITERIONS.keys())} allowed!")

        # check dataset input:
        if all(isinstance(x, np.ndarray) for x in (x_train, x_test, y_train, y_test)):
            pass  # can convert to tensor
        elif all(isinstance(x, list) for x in (x_train, x_test, y_train, y_test)):  # convert list to array
            x_train, x_test, y_train, y_test = (np.array(x_train), np.array(x_test), np.array(y_train),
                                                np.array(y_test))

        # convert to tensor
        x_train = torch.tensor(x_train, dtype=torch.float32, device=fit_device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=fit_device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=fit_device)
        y_test = torch.tensor(y_test, dtype=torch.long, device=fit_device)

        # wrap tensors into a Dataset
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        # load data
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       worker_init_fn=set_seed(42))  # avoid overfitting to sample order
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size)

        if device_warn:
            print(f"Model and Tensor running on {fit_device}")

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            # pass MLP class instance (self)
            train_loss, train_acc, train_prec = train_mlp(self, train_loader, optimizer, criterion, fit_device)
            test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test_mlp(self, test_loader, criterion,
                                                                                fit_device)

            # add metrics
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.train_accuracy.append(train_acc)
            self.test_accuracy.append(test_acc)
            self.train_precision.append(train_prec)
            self.test_precision.append(test_prec)
            self.mcc.append(mcc)
            self.recall.append(recall)
            self.f1_score.append(f1)
            self.roc_auc.append(roc_auc)

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
            print(f"Average Training Loss: {round(np.mean(self.train_loss), 3)}")
            print(f"Average Test Loss: {round(np.mean(self.test_loss), 3)}")
            print(f"Average Training Accuracy: {round(np.mean(self.train_accuracy), 3)}")
            print(f"Average Test Accuracy: {round(np.mean(self.test_accuracy), 3)}")

    def predict(self, data: Union[np.ndarray, list], device: str = "auto", binary: Union[bool, float, str] = True,
                return_type: str = "list"):
        """
        Handle predictions for an input torch Dataloader.
        :param data: Union[np.array, list]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param binary: Union[bool, float, str]
            Set the output for classification models as binary or multi-class. A boolean value will set predictions to
            be binary or multi-class predictions. If a float is given, then a binary predictions will be given with the
            input float used as the cutoff for positive (1) labels. Input strings can be "multi" or "proba", where
            "multi" will output a multi-class labels and "proba" will output a binary probability vector.
        :param return_type: str
            Set the return type of the prediction. Can be "tensor" or "list".
        :return:
        """
        # set model
        model = self

        # check if on the same device
        fit_device = get_device(device)
        print(f"Prediction running on {fit_device}")

        # check dataset input:
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, list):
            data = np.array(data)

        if not isinstance(data, torch.Tensor):
            # convert to tensor
            data = torch.tensor(data, dtype=torch.float32, device=fit_device)
        # make sure data is 2D
        if data.ndim == 1:
            data = data.unsqueeze(0)  # shape [1, features]

        # evaluation mode
        model.eval()
        all_preds = []
        with torch.no_grad():  # disable gradient calculation
            output = model(data)
            # set for binary, multi-class, or probability output
            if binary == True or isinstance(binary, float):
                proba = F.softmax(output, dim=1)[:, 1]
                if isinstance(binary, float):
                    proba = (proba > binary).long()
                else:
                    proba = (proba >= 0.5).long()
            # multi-class classification
            elif binary == False or binary == "multi":
                proba = F.softmax(output, dim=1)
                proba = torch.argmax(proba, dim=1)
            # output probabilities
            elif binary == "proba":
                proba = F.softmax(output, dim=1)
            else:
                raise ValueError("Only 'multi', 'proba', boolean, or float is supported!!")
            all_preds.append(proba)

            # conditional check to output as list or tensor
        if return_type == "list":
            prediction = torch.cat(all_preds)
            return prediction.tolist()
        elif return_type == "tensor":
            return torch.cat(all_preds)
        else:
            raise TypeError(f"Type {return_type} not recognized!")

    def save(self, filepath: str):
        """
        Save a fitted torch model.
        :param filepath: str
            The filepath for the saved torch model. Must be in .pth format.
        :return:
        """
        model = self
        torch.save(model.state_dict(), filepath)
        return "Model saved!"

    def load(self, model_path: Optional[str] = None, param_path: Optional[str] = None, device: str = "auto"):
        """
        Load saved model state.
        :param model_path: Optional[str]
            Path to saved torch model.
        :param param_path: Optional[str]
            Set the path for the saved parameters in .json file.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :return:
        """
        model = self
        # load model and set to eval mode
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print("Model loaded!")


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

    def forward(self, data, **kwargs):
        # kwargs needed to 'ignore' inputs when calling summary
        for layer in self.model:
            data = layer(data)
        return data

    @staticmethod
    def _build_mlp(input_dim, hidden_size, layers, num_classes, dropout, reduce_size):
        """Dynamically build mlp layer. Method shared between graph neural networks."""
        layer_num = 0
        dimension = input_dim  # graph latent space
        # loop mlp layers
        model_layers = nn.ModuleList()

        for _ in range(layers - 1):
            # optional, halve hidden size if reduce_size is True
            if reduce_size and layer_num > 0:
                hidden_size = round(hidden_size / 2)
            model_layers.append(nn.Sequential(
                nn.Linear(dimension, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            dimension = hidden_size
            layer_num += 1
        # final mlpl layer
        model_layers.append(nn.Linear(dimension, num_classes))

        return model_layers


# train/test for MLP model
def train_mlp(model, train_loader, optimizer, criterion, device):
    """Support function for training the torch model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for data, y in train_loader:
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(data)

        # # todo ass if/else to classify regression or classification
        # # create a dictionary to sort through?
        # if criterion is a regression:
        #     targets = targets.view(-1, 1).float()
        #     loss = criterion(outputs, targets)
        # elif tast is classification
        #     loss = criterion(outputs, targets.long())

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        # all prediction labels
        all_preds.append(pred)
        all_labels.append(y)

    # precision values
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    # true positives
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    # false positives
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()

    model_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_pcc = tp / (tp + fp + 1e-8)  # to avoid division by 0

    return model_loss, train_acc, train_pcc


@torch.no_grad()
def test_mlp(model, test_loader, criterion, device):
    """Support function for testing the torch model"""
    global mcc, recall, f1, roc_auc
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    all_labels = []

    for data, y in test_loader:
        data = data.to(device)
        y = y.to(device)

        # forward pass
        out = model(data)

        loss = criterion(out, y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        # all prediction labels
        predictions.append(pred)
        all_labels.append(y)

    # precision values
    predictions = torch.cat(predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()

    # true positives
    tp = ((predictions == 1) & (all_labels == 1)).sum().item()
    # false positives
    fp = ((predictions == 1) & (all_labels == 0)).sum().item()

    test_loss = total_loss / len(test_loader)
    test_acc = correct / total
    test_prec = tp / (tp + fp + 1e-8)  # to avoid division by 0

    # mcc
    mcc = matthews_corrcoef(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    roc_auc = roc_auc_score(all_labels, predictions)

    return test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc


# train/test for graph models
def train(model, train_loader, optimizer, criterion, device):
    """Support function for training the torch model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # check device type
    # device_type = get_device(device)

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(data)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        # all prediction labels
        all_preds.append(pred)
        all_labels.append(data.y)

    # precision values
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    # true positives
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    # false positives
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()

    model_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_pcc = tp / (tp + fp + 1e-8)  # to avoid division by 0

    return model_loss, train_acc, train_pcc


@torch.no_grad()
def test(model, test_loader, criterion, device):
    """Support function for testing the torch model"""
    global mcc, recall, f1, roc_auc
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    all_labels = []

    # check device type
    # device_type = get_device(device)

    for data in test_loader:
        data = data.to(device)

        # forward pass
        out = model(data)

        loss = criterion(out, data.y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        # all prediction labels
        predictions.append(pred)
        all_labels.append(data.y)

    # precision values
    predictions = torch.cat(predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()

    # true positives
    tp = ((predictions == 1) & (all_labels == 1)).sum().item()
    # false positives
    fp = ((predictions == 1) & (all_labels == 0)).sum().item()

    test_loss = total_loss / len(test_loader)
    test_acc = correct / total
    test_prec = tp / (tp + fp + 1e-8)  # to avoid division by 0

    # mcc
    mcc = matthews_corrcoef(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    roc_auc = roc_auc_score(all_labels, predictions)

    return test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc


def _global_pooling(x, batch, pool_type='add'):
    if pool_type == 'add':
        return global_add_pool(x, batch)
    elif pool_type == 'mean':
        return global_mean_pool(x, batch)
    elif pool_type == 'max':
        return global_max_pool(x, batch)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
