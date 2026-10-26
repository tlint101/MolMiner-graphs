import inspect
import numpy as np
from tqdm import tqdm
from typing import Optional, Union
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from mminer_graphs.mlx_models.mlx_utils import get_device, MLX_ACT


class Base(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fitted = None
        self.initial_params = None

    def split_graph_space(self, train_loader: Union[Dataloader, list], test_loader: Union[Dataloader, list],
                          batch_size: int = 1, device: str = "auto"):
        """
        Extract graph latent space vectors from the train and test dataset. This will output train_x, train_y, test_x,
        and test_y datasets. Note, if the training dataset is given as a list, it will be shuffled.
        :param train_loader: Union[Dataloader, list]
            The data loader for the training data.
        :param test_loader: Union[Dataloader, list]
            The data loader for the test data.
        :param batch_size: int
            Set the batch size during model training. This parameter is only used if hte train_loader/test_loader input
            is a list instead of type Dataloader.
        :param device: str
            Set the location of the arrays. Defaults to Apple Silicon GPU.
        :return:
        """
        # check device
        fit_device = get_device(device)
        model = self

        # check dataset input:
        if isinstance(train_loader, list):
            train_loader = Dataloader(train_loader, batch_size=batch_size, shuffle=True)
        if isinstance(test_loader, list):
            test_loader = Dataloader(test_loader, batch_size=batch_size)

        train_space = []
        train_labels = []
        test_space = []
        test_labels = []

        # evaluate
        mx.eval()
        sig = inspect.signature(model.__call__)
        for data in tqdm(train_loader, desc="Converting training graph data"):
            if "edge_features" in sig.parameters and hasattr(data, "edge_features"):  # for edge features
                latent_space = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices)
            else:
                latent_space = model(data.edge_index, data.node_features, data.batch_indices)
            train_space.append(latent_space)
            train_labels.append(data.y)
        for data in tqdm(test_loader, desc="Converting test graph data"):
            if "edge_features" in sig.parameters and hasattr(data, "edge_features"):  # for edge features
                latent_space = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices)
            else:
                latent_space = model(data.edge_index, data.node_features, data.batch_indices)
            test_space.append(latent_space)
            test_labels.append(data.y)

        # combine mlx arrays
        train_x = mx.concatenate(train_space, axis=0)
        train_y = mx.concatenate(train_labels, axis=0)
        test_x = mx.concatenate(test_space, axis=0)
        test_y = mx.concatenate(test_labels, axis=0)

        # convert mlx arrays to numpy
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return train_x, train_y, test_x, test_y

    def extract_graph_space(self, data_loader: Union[Dataloader, list], batch_size: int = 1, device: str = "auto"):
        """
        Convert input molecular graphs into a numpy array. Results can the nbe fed into a sklearn model. This is
        primarily used for the prediction step.
        :param data_loader: Union[Dataloader, list]
            The data loader for prediction dataset.
        :param batch_size: int
            Set the batch size during model training. This parameter is only used if hte train_loader/test_loader input
            is a list instead of type Dataloader.
        :param device: str
            Set the location of the arrays. Defaults to Apple Silicon GPU.
        :return:
        """
        # check device
        fit_device = get_device(device)
        model = self

        # check dataset input:
        if isinstance(data_loader, list):
            data_loader = Dataloader(data_loader, batch_size=batch_size)

        space = []

        # evaluate
        mx.eval()
        sig = inspect.signature(model.__call__)
        for data in tqdm(data_loader, desc="Converting graph data"):
            if "edge_features" in sig.parameters and hasattr(data, "edge_features"):  # for edge features
                latent_space = model(data.edge_index, data.node_features, data.edge_features, data.batch_indices)
            else:
                latent_space = model(data.edge_index, data.node_features, data.batch_indices)
            space.append(latent_space)

        # combine mlx arrays and convert to numpy
        latent_space = mx.concat(space, axis=0)
        latent_space = np.array(latent_space)
        return latent_space

    def _loss_fn(self, y_hat, y, parameters=None):
        loss = mx.mean(self.criterion(y_hat, y))
        return loss

    def _forward_fn(self, model, graph, labels):
        """forward function"""
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

        global graph_hidden_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout
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

    def __call__(self, edge_index, x, batch_indices):
        # graph layers
        act_fn = MLX_ACT[self.act]
        for conv in self.gcn_conv:
            x = act_fn(conv(edge_index, x))
            if self.graph_dropout > 0:
                x = mlx.nn.Dropout(p=self.graph_dropout)(x)

        # readout layer
        x = _global_pooling(x, batch_indices, pool_type=self.pooling)

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

        global reduced_graph_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout

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

        return x


class GSAGE(Base):
    def __init__(self, node_dim: int = 142, hidden_dim: int = 128, graph_layers: int = 2, graph_dropout: float = 0.0,
                 act: str = "relu", reduce_graph=False, pooling: str = "mean"):
        """
        Generate a latent graph space vectors from GraphSAGE convolution layer. The default model will generate two
        graph layers. Results will be outputted as an array.
        :param node_dim: int
            Set the length of node feature vector.
        :param hidden_dim: int
            Set the dimensions of the hidden layers in teh graph network.
        :param graph_layers: int
            Number of graph layers.
        :param graph_dropout: float
            Set the dropout for each graph layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param pooling: str
            Set the pooling method.
        """
        global graph_hidden_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout

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

        return x


class GAT(Base):
    def __init__(self, node_dim: int = 142, edge_feat_dim: Optional[int] = None, hidden_dim: int = 128, heads: int = 1,
                 graph_layers: int = 2, graph_dropout: float = 0.0, act: str = "relu", reduce_graph=False,
                 pooling: str = "mean"):
        """
        Generate a latent graph space vectors from Graph Attention Network convolution layer. The default model will
        generate two graph layers. Results will be outputted as an array.
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
        :param graph_dropout: float
            Set the dropout for each graph layer.
        :param act: str
            Set the non-linear activation function. Default to "relu".
        :param reduce_graph: bool
            Determine if the hidden_dim in the graph layer should be reduced by 2 for each subsequent layer.
        :param pooling: str
            Set pooling method.
        """
        global reduced_graph_dim, out_dim
        super().__init__()
        self.act = act
        self.pooling = pooling
        self.graph_dropout = graph_dropout

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

        return x


def _global_pooling(x, batch_indices, pool_type='add'):
    if pool_type == 'add':
        return global_add_pool(x, batch_indices)
    elif pool_type == 'mean':
        return global_mean_pool(x, batch_indices)
    elif pool_type == 'max':
        return global_max_pool(x, batch_indices)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
