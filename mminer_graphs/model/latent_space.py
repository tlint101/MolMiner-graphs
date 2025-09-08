import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from mminer_graphs.model.graph_modeling import EarlyStopping
from mminer_graphs.model.model_utils import set_seed, get_device, OPTIMIZERS, CRITERIONS, F_ACT
from tqdm import tqdm
from typing import Optional, Union, Type
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, roc_auc_score


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = None

    def fit(self, train_loader: Union[DataLoader, list], test_loader: Union[DataLoader, list], epochs: int = 10,
            optimizer: str = "adam", criterion: str = "cross_entropy", lr: float = 0.01, wt_decay: float = 0.0,
            batch_size: int = 1024, device="auto", estop: Optional[Union[EarlyStopping, int]] = None, pbar=True,
            verbose: bool = False, device_warn: bool = True,
            **kwargs):
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
            optimizer = OPTIMIZERS[optimizer](self.train().parameters(), lr=lr, weight_decay=wt_decay)
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

        if device_warn:
            print(f"Model and Tensor running on {fit_device}")

        # reset early_stopping if used
        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor='val_loss', patience=estop, mode='min')

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            self.train()
            train_loss, train_acc, train_prec = train(self.train(), train_loader, optimizer, criterion, fit_device)
            test_loss, test_acc, test_prec, mcc, recall, f1, roc_auc = test(self.train(), test_loader, criterion,
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

    def train_graph_space(self, train_loader: Union[DataLoader, list], test_loader: Union[DataLoader, list],
                          batch_size: int = 1024, device: str = "cpu"):
        """
        Extract graph latent space vectors from the train and test dataset. Will output train_x, train_y, test_x and
        test_y.
        :param train_loader: Union[DataLoader, list]
            The data loader for the training data.
        :param test_loader: Union[DataLoader, list]
            The data loader for the test data.
        :param batch_size: int
            Set the  batch size during model training. This parameter is only used if the train_loader/test_loader input
            is a list instead of a DataLoader.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :return:
        """
        # check if on the same device
        fit_device = get_device(device)

        # check dataset input:
        if isinstance(train_loader, list):
            train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=set_seed(42))  # avoid overfitting to sample order
        if isinstance(test_loader, list):
            test_loader = DataLoader(test_loader, batch_size=batch_size)

        train_space = []
        train_labels = []
        test_space = []
        test_labels = []

        # evaluation mode
        self.eval()
        with torch.no_grad():
            for data in tqdm(train_loader, desc="Converting training data"):
                data = data.to(fit_device)
                latent_space = self(data)
                train_space.append(latent_space)
                train_labels.append(data.y)
            for data in tqdm(test_loader, desc="Converting test data"):
                data = data.to(fit_device)
                latent_space = self(data)
                test_space.append(latent_space)
                test_labels.append(data.y)

        train_x = torch.cat(train_space, dim=0).cpu().numpy()
        train_y = torch.cat(train_labels, dim=0).cpu().numpy()
        test_x = torch.cat(test_space, dim=0).cpu().numpy()
        test_y = torch.cat(test_labels, dim=0).cpu().numpy()
        return train_x, train_y, test_x, test_y

    def predict_graph_space(self, data_loader: Union[DataLoader, list], batch_size: int = 1024, device: str = "cpu"):
        """
        :param data_loader: Union[DataLoader, list]
            The data loader for prediction dataset.
        :param batch_size: int
            Set the  batch size during model training. This parameter is only used if the train_loader/test_loader input
            is a list instead of a DataLoader.
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :return:
        """
        # check if on the same device
        fit_device = get_device(device)

        # evaluation mode
        self.eval()
        # check dataset input:
        if isinstance(data_loader, list):
            data_loader = DataLoader(data_loader, batch_size=batch_size, worker_init_fn=set_seed(42))

        pred_space = []

        with torch.no_grad():
            for data in tqdm(data_loader, desc="Converting data"):
                data = data.to(fit_device)
                latent_space = self(data)
                pred_space.append(latent_space)

        latent_space = torch.cat(pred_space, dim=0).cpu().numpy()
        return latent_space


# Graph Convolutional Neural Network
class GCNN(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, reduce_graph: bool = False,
                 graph_dropout: float = 0.0, act: str = "relu", pooling: str = "mean"):
        """
        Generate graph model for graph convolutional (GCNC) network. The default model will generate two graph layers
        and one mlp layer.
        :param node_dim: int
            Length of feature vector per node.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param graph_layers: int
            Number of graph layers.
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param graph_dropout: float
            Set the graph dropout during activation.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param pooling: str
            Set pooling method.
        """

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

        # graph layer training info
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

        return x


class GINE(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: int = 16, hidden_dim: int = 64, graph_layers: int = 2,
                 reduce_graph: bool = False, graph_dropout: float = 0.0, act: str = "relu", pooling: str = "add"):
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
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param graph_dropout: float
            Set the graph dropout during activation.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param pooling: str
            Set pooling method.
        """

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

        # graph layer training info
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

        return x


# GraphSAGE
class GSAGE(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, graph_dropout: float = 0.0,
                 act: str = "relu", reduce_graph=False, pooling: str = "mean"):
        """
        Generate a graph SAGE model neural network (GSAGENet). The default model will generate two graph layers and one
        mlp layer.
        :param node_dim: int
            Length of node feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in the graph network.
        :param graph_layers: int
            Number of graph layers.
        :param graph_dropout: float
            Set the graph dropout during activation.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
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

        # graph layer training info
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

        return x


# Graph Attention Network
class GAT(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: Optional[int] = None, hidden_dim: int = 128, heads: int = 1,
                 graph_layers: int = 2, graph_dropout: float = 0.0, act: str = "relu", reduce_graph=False,
                 pooling: str = "mean"):
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
        :param graph_dropout: float
            Set the dropout for the attention coefficients and during activation.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
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

        # graph layer training info
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

        return x


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
