import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from mminer_graphs.model.model_utils import set_seed, get_device, F_ACT
from tqdm import tqdm
from typing import Optional, Union, Type


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def split_graph_space(self, train_loader: Union[DataLoader, list], test_loader: Union[DataLoader, list],
                          batch_size: int = 1, device: str = "cpu"):
        """
        Extract graph latent space vectors from the train and test dataset. This will output train_x, train_y, test_x and
        test_y datasets. Note, if the training dataset is given as a list, it will be shuffled
        :param train_loader: Union[DataLoader, list]
            The data loader for the training data.
        :param test_loader: Union[DataLoader, list]
            The data loader for the test data.
        :param batch_size: int
            Set the  batch size during model training. This parameter is only used if the train_loader/test_loader input
            is a list instead of type DataLoader.
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

        # convert to numpy on cpu
        train_x = torch.cat(train_space, dim=0).cpu().numpy()
        train_y = torch.cat(train_labels, dim=0).cpu().numpy()
        test_x = torch.cat(test_space, dim=0).cpu().numpy()
        test_y = torch.cat(test_labels, dim=0).cpu().numpy()
        return train_x, train_y, test_x, test_y

    def extract_graph_space(self, data_loader: Union[DataLoader, list], batch_size: int = 1024, device: str = "cpu"):
        """
        Convert input molecular graphs into a numpy array. Results can the nbe fed into a sklearn model. This is
        primarily used for the prediction step.
        :param data_loader: Union[DataLoader, list]
            The data loader for prediction dataset.
        :param batch_size: int
            Set the  batch size during model training. This parameter is only used if the train_loader/test_loader input
            is a list instead of type DataLoader.
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

        space = []

        with torch.no_grad():
            for data in tqdm(data_loader, desc="Converting data"):
                data = data.to(fit_device)
                latent_space = self(data)
                space.append(latent_space)

        latent_space = torch.cat(space, dim=0).cpu().numpy()
        return latent_space


# Graph Convolutional Neural Network
class GCN(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, reduce_graph: bool = False,
                 graph_dropout: float = 0.0, act: str = "relu", pooling: str = "mean"):
        """
        Generate a latent graph space vectors from graph convolutional layer. The default model will generate two graph
        layers. Results will be outputted as an array.
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

        # set graph layers
        self.gcn_conv = nn.ModuleList()
        self.normalization = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        for x in range(graph_layers):
            # for updating the nodes
            in_dim = node_dim if x == 0 else graph_hidden_dim
            self.gcn_conv.append(GCNConv(in_dim, hidden_dim))
            self.normalization.append(nn.BatchNorm1d(hidden_dim))

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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # graph layers
        act_fn = F_ACT[self.act]
        for i, (conv, norm) in enumerate(zip(self.gcn_conv, self.normalization)):
            x = conv(x, edge_index)
            x = norm(x)
            x = act_fn(x)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        return x


class GINE(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: int = 16, hidden_dim: int = 64, graph_layers: int = 2,
                 reduce_graph: bool = False, graph_dropout: float = 0.0, act: str = "relu", pooling: str = "add"):
        """
        Generate a latent graph space vectors from Graph Isomorphism Network convolution layer. The default model
        will generate two graph layers. Results will be outputted as an array.
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

        # set graph layers
        self.gine_conv = nn.ModuleList()
        self.normalization = nn.ModuleList()

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
            self.normalization.append(nn.BatchNorm1d(hidden_dim))

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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # graph layers
        act_fn = F_ACT[self.act]
        for i, (conv, norm) in enumerate(zip(self.gine_conv, self.normalization)):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = act_fn(x)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        return x


# GraphSAGE
class GSAGE(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, graph_dropout: float = 0.0,
                 act: str = "relu", reduce_graph=False, pooling: str = "mean"):
        """
        Generate a latent graph space vectors from GraphSAGE convolution layer. The default model will generate two
        graph layers. Results will be outputted as an array.
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

        # set graph layers
        self.sage_conv = nn.ModuleList()
        self.normalization = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.sage_conv.append(
                    SAGEConv(in_channels=in_channels, out_channels=graph_hidden_dim))
                # normalize nodes
                self.normalization.append(nn.BatchNorm1d(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.sage_conv.append(
                    SAGEConv(in_channels=in_channels, out_channels=graph_hidden_dim))
                # normalize nodes
                self.normalization.append(nn.BatchNorm1d(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels for loop

        # graph layer training info
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_precision = []
        self.test_precision = []

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # check activation function
        if self.act not in F_ACT:
            raise ValueError(f"Unsupported activation function {self.act}"
                             f"Only {list(F_ACT.keys())} are supported!")

        # graph layers
        act_fn = F_ACT[self.act]
        for i, (conv, norm) in enumerate(zip(self.sage_conv, self.normalization)):
            x = conv(x, edge_index)
            x = norm(x)
            x = act_fn(x)
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
        Generate a latent graph space vectors from Graph Attention Network convolution layer. The default model will
        generate two graph layers. Results will be outputted as an array.
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

        # set graph layers
        self.gat_conv = nn.ModuleList()
        self.normalization = nn.ModuleList()

        # set graph layers
        graph_hidden_dim = hidden_dim
        in_channels = node_dim
        for x in range(graph_layers):
            if x > 0:
                graph_hidden_dim = round(graph_hidden_dim / 2) if reduce_graph is True else graph_hidden_dim

            if x == 0:
                self.gat_conv.append(GATConv(in_channels=in_channels, out_channels=graph_hidden_dim, heads=heads,
                                             edge_dim=edge_feat_dim, dropout=graph_dropout))
                # normalize nodes
                self.normalization.append(nn.BatchNorm1d(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels
            else:
                self.gat_conv.append(GATConv(in_channels=in_channels, out_channels=graph_hidden_dim, heads=heads,
                                             edge_dim=edge_feat_dim, dropout=graph_dropout))
                # normalize nodes
                self.normalization.append(nn.BatchNorm1d(hidden_dim))
                in_channels = graph_hidden_dim  # reset in_channels for loop

        # graph layer training info
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_precision = []
        self.test_precision = []

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
        for i, (conv, norm) in enumerate(zip(self.gat_conv, self.normalization)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = act_fn(x)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)

        # readout layer
        x = _global_pooling(x, batch, pool_type=self.pooling)

        return x


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
