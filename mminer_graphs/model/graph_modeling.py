import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, GraphSAGE, GATConv, global_mean_pool, global_max_pool, \
    global_add_pool, SAGEConv
from torch_geometric.loader import DataLoader
from mminer_graphs.model.model_utils import set_seed, mol_to_graph, get_device, OPTIMIZERS, CRITERIONS, F_ACT
from tqdm import tqdm
from typing import Optional, Union
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

    def fit(self, model, train_loader: Union[DataLoader, list], test_loader: Union[DataLoader, list], epochs: int = 10,
            optimizer: str = "adam", criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0,
            batch_size: int = 1024, device="auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar=True,
            verbose: bool = False, device_warn: bool = True,
            **kwargs):
        """
        Fit the model.
        :param model: torch.nn.Module
            The torch model to fit.
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
        :param pbar:
            Whether to show a progress bar during model fit for each epoch.
        :param verbose: bool
            Determine whether to output train/test accuracy and loss for each epoch.
        :param device_warn:
            Option to print which tensor/model is on 'cpu', 'cuda' or 'mps'.
        :param kwargs: dict
            kwargs for a given criterion.
        :return:
        """

        # check if on the same device
        fit_device = get_device(device)
        self.to(fit_device)

        # set optimizer and criterion
        if optimizer in OPTIMIZERS.keys():
            optimizer = OPTIMIZERS[optimizer](model.parameters(), lr=lr, weight_decay=wt_decay)
        elif criterion is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError("Optimizer not recognized!")
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

        if not isinstance(train_loader, DataLoader) or not isinstance(test_loader, DataLoader):
            raise TypeError("Issue with the train_loader and hte test_loader. Must be a list or a DataLoader instance!")

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

        # return instance
        return self

    @staticmethod
    def predict(model: nn.Module, test_loader: Union[np.array, list], device: str = "auto", return_type: str = "list"):
        """
        Handle predictions for an input torch Dataloader.
        :param model: torch.nn.Module
            A trained torch model.
        :param test_loader: Union[np.array, list]
            Data for prediction. Must be converted into a graph and organized in an array or list.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param return_type: str
            Set the return type of the prediction. Can be "tensor" or "list".
        :return:
        """
        # check if on the same device
        fit_device = get_device(device)
        print(f"Prediction running on {fit_device}")

        # evaluation mode
        model.eval()
        all_preds = []
        with torch.no_grad():  # disable gradient calculation
            for data in test_loader:
                datapoint = data.to(fit_device)
                output = model(datapoint)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds)

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
        if hasattr(self, "gcn_conv"):
            conv_layers = getattr(self, "gcn_conv")
        elif hasattr(self, "gine_conv"):
            conv_layers = getattr(self, "gine_conv")
        else:
            raise ValueError("No conv layers defined found!")
        for layer in conv_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # mlp layers
        for x in self.mlp:
            for layer in x:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # final mlp layer
        if hasattr(self.final_mlp, 'reset_parameters'):
            self.final_mlp.reset_parameters()

    @staticmethod
    def save(model: nn.Module, filepath: str):
        """
        Save a fitted torch model.
        :param model: nn.Module
            A trained torch model.
        :param filepath: str
            The filepath for the saved torch model. Must be in .pth format.
        :return:
        """
        torch.save(model.state_dict(), filepath)
        return "Model saved!"

    @staticmethod
    def load(model: nn.Module, filepath: str):
        """
        Load a torch model from a .pth file.
        :param model: nn.Module
            Instantiate the model class to load the saved parameter tensors into.
        :param filepath: str
            The filepath for the saved torch model. Must be in .pth format.
        :return:
        """
        model.load_state_dict(torch.load(filepath))
        return "Model loaded!"

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
    def __init__(self, node_dim: int = 142, hidden_dim: int = 64, graph_layers: int = 2, mlp_layers: int = 1,
                 reduce_graph: bool = False, dropout: float = 0.0, num_classes: int = 2,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """
        Generate graph model for graph convolutional (GCNC) network. The default model will generate two graph layers
        and one mlp layer.
        :param node_dim: int
            Length of feature vector per node.
        :param hidden_dim: int
            Set the size of the hidden layers.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param dropout: float
            Indicate a dropout layer after each mlp layer. Defaults to 0 (no dropout).
        :param num_classes: int
            Number of classes for the output layer.
         :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        global graph_hidden_dim
        super().__init__()
        self.pooling = pooling
        self.is_fitted = False

        # set graph layers
        self.gcn_conv = nn.ModuleList()
        # set graph layers
        # first layer -> input to hidden
        self.gcn_conv.append(GCNConv(node_dim, hidden_dim))
        # loop GCNConv layer
        graph_hidden_dim = hidden_dim
        for x in range(graph_layers - 1):
            graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim
            self.gcn_conv.append(GCNConv(graph_hidden_dim, graph_hidden_dim))
            graph_hidden_dim = graph_hidden_dim

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = hidden_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, dropout)

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
        for conv in self.gcn_conv:
            x = F.relu(conv(x, edge_index))

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            if not isinstance(data.add_feats, torch.Tensor):
                add_feats = torch.tensor(data.add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = data.add_feats.to(device=x.device, dtype=x.dtype)

            x = torch.cat([x, add_feats], dim=1)

        # build mlp layers
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# Graph Isomorphism Network with Edge Features
class GINENet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: int = 16, hidden_dim: int = 64, graph_layers: int = 2,
                 mlp_layers: int = 1, reduce_graph: bool = False, dropout: float = 0.0, num_classes: int = 2,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "add"):
        """
        Generate graph model for graph isomorphism convolutional (GINEConv) network. The default model will generate two
        graph layers and one mlp layer.
        :param node_dim: int
            Length of node feature vector.
        :param edge_feat_dim: int
            Length of edge feature vector.
        :param hidden_dim: ing
            Set the size of the hidden layers.
        :param graph_layers: int
            Number of graph layers.
        :param mlp_layers: int
            Set the number of MLP layers.
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param dropout: float
            Indicate a dropout layer after each mlp layer. Defaults to 0 (no dropout).
        :param num_classes: int
            Number of classes for the output layer.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of features to be added. If an int is given, it must correspond to the length of the additional
            features. If a np.array/list of the additional features are given, the length of the data will be calculated
            and used to append the MLP input dimensions accordingly.
        :param pooling: str
            Set pooling method.
        """

        global reduced_graph_dim
        super().__init__()
        self.pooling = pooling
        self.is_fitted = False

        # set graph layers
        self.gine_conv = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            reduced_graph_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            # for updating the nodes
            nnet = nn.Sequential(
                nn.Linear(hidden_dim if x > 0 else node_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # normalize node embeddings
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            self.gine_conv.append(GINEConv(nn=nnet, edge_dim=edge_feat_dim))

        # if a list of additional features are given, will be added to the graph latent array
        if isinstance(add_feats, list) or isinstance(add_feats, np.ndarray):
            add_feat = len(add_feats[0])
        elif isinstance(add_feats, int):
            add_feat = add_feats
        else:
            add_feat = 0

        # get dimension of mlp layers
        mlp_dim = graph_hidden_dim + add_feat
        self.mlp, self.final_mlp = self._build_mlp(mlp_dim, mlp_layers, num_classes, dropout)

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
        for conv in self.gine_conv:
            x = F.relu(conv(x, edge_index, edge_attr))

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            if not isinstance(data.add_feats, torch.Tensor):
                add_feats = torch.tensor(data.add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = data.add_feats.to(device=x.device, dtype=x.dtype)

            x = torch.cat([x, add_feats], dim=1)

        # build mlp layer
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# GraphSAGE
class GSAGENet(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 64, graph_layers: int = 2,
                 mlp_layers: int = 1, graph_dropout: float = 0.0, mlp_dropout: float = 0.0, num_classes: int = 2,
                 act: str = "relu", reduce_graph=False,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """
        Generate a graph SAGE model neural network (GSAGENet). The default model will generate two graph layers and one
        mlp layer.
        :param node_dim: int
            Length of node feature vector.
        :param hidden_dim: int
            Set the size of the hidden layers.
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
            if not isinstance(data.add_feats, torch.Tensor):
                add_feats = torch.tensor(data.add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = data.add_feats.to(device=x.device, dtype=x.dtype)

            x = torch.cat([x, add_feats], dim=1)

        # build mlp layer
        for layer in self.mlp:
            x = layer(x)
        x = self.final_mlp(x)

        return x


# Graph Attention Network
class GATNet(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: Optional[int] = None, hidden_dim: int = 64, heads: int = 1,
                 graph_layers: int = 2, mlp_layers: int = 1, graph_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 num_classes: int = 2, act: str = "relu", reduce_graph=False,
                 add_feats: Optional[Union[np.array, list, int]] = None, pooling: str = "mean"):
        """

        :param node_dim: int
            Length of node feature vector.
        :param hidden_dim: int
            Set the size of the hidden layers.
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

        # check activation function
        if self.act not in F_ACT:
            raise ValueError(f"Unsupported activation function {self.act}"
                             f"Only {list(F_ACT.keys())} are supported!")

        # graph layers
        act_fn = F_ACT[self.act]
        for conv in self.gat_conv:
            x = act_fn(conv(x, edge_index))
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        # concatenate additional features
        if hasattr(data, "add_feats"):
            if not isinstance(data.add_feats, torch.Tensor):
                add_feats = torch.tensor(data.add_feats, device=x.device, dtype=x.dtype)
            else:
                add_feats = data.add_feats.to(device=x.device, dtype=x.dtype)

            x = torch.cat([x, add_feats], dim=1)

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


# set train() to GCNConvNet_train
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
