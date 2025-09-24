import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import Optional, Union
from torchviz import make_dot
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import DataLoader
from mminer_graphs.model.model_utils import set_seed, get_device


class Plots:
    def __init__(self, model: Optional[torch.nn] = None, data_loader: Optional[DataLoader] = None,
                 x_test: Optional[Union[list, np.ndarray]] = None, y_test: Optional[Union[list, np.ndarray]] = None,
                 batch_size: Optional[int] = None, device: str = "auto"):
        """
        Instantiate Plots class. For MLP models only, x_test and y_test are required. Otherwise, data_loader is only
        needed for graph models.
        :param model: Optional[torch.nn]
        :param data_loader: Optional[DataLoader]
        :param x_test: Optional[Union[list, np.ndarray]]
        :param y_test: Optional[Union[list, np.ndarray]]
        :param batch_size: Optional[int]
        :param device: str
            Set the device for the test_loader tensors. Can be "auto", "cuda", "mps", or "cpu".
        """
        self.model = model
        self.device = device

        # if list or nd.array given for x_test and y_test
        if x_test and y_test is not None:
            # check dataset input:
            if all(isinstance(x, np.ndarray) for x in (x_test, y_test)):
                pass  # can convert to tensor
            elif all(isinstance(x, list) for x in (x_test, y_test)):  # convert list to array
                x_test, y_test = (np.array(x_test), np.array(y_test))

            # convert to tensor
            x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
            y_test = torch.tensor(y_test, dtype=torch.long, device=device)

            # wrap tensors into a Dataset
            test_dataset = TensorDataset(x_test, y_test)

            # load data
            test_loader = TorchDataLoader(test_dataset, batch_size=batch_size)
        else:
            test_loader = None

        if data_loader:
            self.data = data_loader
        elif test_loader:
            self.data = test_loader

    def curves(self, model: Optional[torch.nn] = None, metrics: list = ["accuracy", "loss"],
               line_color: tuple = ("#1f77b4", "#ff7f0e"), legend_loc: str = "best", figsize: tuple = (14, 5),
               title: str = None, savepath: str = None):
        """
        Plot the model history. This will generate a single plot containing 2 subplots. It can be any combination of
        'accuracy', 'loss', or 'precision'.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param metrics: list
            A list of metrics to plot. Can only take a combination of 'accuracy', 'loss', or 'precision'.
        :param line_color: tuple
            A tuple containing the line colors.
        :param legend_loc: str
            Set the location of the figure legends.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None, the default title will be used.
        :param savepath: str
            Set the savepath for the image.
        :return:
        """
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("Torch model is not fitted!!!")

        # get data metrics from model
        metric_list = list(model.__dict__)

        # if list, only accept accuracy, loss, or precision
        if isinstance(metrics, list):
            for metric in metrics:
                if metric not in {'accuracy', 'loss', 'precision'}:
                    raise ValueError(f"Only 'loss', 'accuracy', or 'precision' are supported!!")
                else:
                    input_metrics = metrics

        # match input_metrics to metric_list
        metric_input_match = [metric for metric in metric_list if any(key in metric for key in input_metrics)]
        metrics_len = len(metric_input_match)
        if metrics_len == 0:
            raise ValueError(f"Input metrics '{metrics}' not found!")

        if title is None:
            title = "Model Training History"

        # plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)
        axes[0].plot(getattr(model, metric_input_match[0]), color=line_color[0])
        axes[0].plot(getattr(model, metric_input_match[1]), color=line_color[1])
        axes[0].legend(["Train", "Test"], loc=legend_loc)
        axes[0].set_title(f"Model {metric_input_match[0].removeprefix('train_').removeprefix('test_').capitalize()}")
        axes[0].set_ylabel(metric_input_match[0].removeprefix('train_').removeprefix('test_').capitalize())
        axes[0].set_xlabel("Epoch")
        axes[0].grid(False)

        axes[1].plot(getattr(model, metric_input_match[2]), color=line_color[0])
        axes[1].plot(getattr(model, metric_input_match[3]), color=line_color[1])
        axes[1].legend(["Train", "Test"], loc=legend_loc)
        axes[1].set_title(f"Model {metric_input_match[2].removeprefix('train_').removeprefix('test_').capitalize()}")
        axes[1].set_ylabel(metric_input_match[2].removeprefix('train_').removeprefix('test_').capitalize())
        axes[1].set_xlabel("Epoch")
        axes[1].grid(False)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def confusion_matrix(self, model: Optional[torch.nn] = None, data: Optional[DataLoader] = None,
                         class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                         title: str = None, binary: Union[bool, float, str] = True, savepath: str = None):
        """
        Generate a confusion matrix for binary classification torch models.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param data: DataLoader
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to binary.
        :param palette: str
            Set the color palette for the figure.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None given, the default title will be used.
        :param binary: Union[bool, float, str]
            Set the prediction output. Can only accept 'True', 'multi', or a float as inputs.
        :param savepath: str
            Set the figure savepath.
        :return:
        """

        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("Torch model is not fitted!!!")
        if data is None:
            data = self.data

        # get labels from test_loader
        all_labels = []
        # get labels
        # for graph data
        try:
            for datapoint in data:
                labels = datapoint.y
                all_labels.extend(labels.to('cpu'))  # move tensors to cpu
        except AttributeError:
            # for MLP tabular data
            for x_batch, y_batch in data:
                labels = y_batch
                data = x_batch
                all_labels.extend(labels.to('cpu'))

        # check binary input
        if binary == False or binary == "proba":
            raise ValueError("Confusion Matrix can only accept 'True', 'multi', or a float!")

        # get prediction labels
        if self.device == "cuda" or self.device == "gpu":
            return_type = "tensor"
        else:
            return_type = "list"

        predictions = model.predict(data=data, device=self.device, binary=binary, return_type=return_type)

        if self.device == "cuda" or self.device == "gpu":
            # move tensors to cpu
            y_true = np.array(all_labels)
            y_pred = predictions.detach().cpu().numpy()
        else:
            # convert to np.array
            y_true = np.array(all_labels)
            y_pred = np.array(predictions)

        # calculate matrix
        cm_matrix = confusion_matrix(y_true, y_pred)

        # get matrix labels
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = [value for value in cm_matrix.flatten()]
        group_percentages = [f"{round(value * 100, 3)}%" for value in cm_matrix.flatten() / np.sum(cm_matrix)]
        labels = [f"{v1}\n\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        if title is None:
            title = "Confusion Matrix on Test Dataset"
        if class_names is None:
            class_names = ['0', '1']

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_matrix, annot=labels, fmt='', cmap=palette,
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def classification_report(self, model: Optional[torch.nn] = None, data: Optional[DataLoader] = None,
                              class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                              title: str = None, binary: Union[bool, float, str] = True, savepath: str = None):
        """
        Generate a classification report for binary classification torch models.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param data: DataLoader
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to binary.
        :param palette: str
            Set the color palette for the figure.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None given, the default title will be used.
        :param binary: Union[bool, float, str]
            Set the prediction output. Can only accept 'True', 'multi', or a float as inputs.
        :param savepath: str
            Set the figure savepath.
        :return:
        """
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("Torch model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Classification Report"
        if class_names is None:
            class_names = ['0', '1']

        # get labels from test_loader
        all_labels = []
        # get labels
        # for graph data
        try:
            for datapoint in data:
                labels = datapoint.y
                all_labels.extend(labels.to('cpu'))  # move tensors to cpu
        except AttributeError:
            # for MLP tabular data
            for x_batch, y_batch in data:
                labels = y_batch
                data = x_batch
                all_labels.extend(labels.to('cpu'))

        # check binary input
        if binary == False or binary == "proba":
            raise ValueError("Confusion Matrix can only accept 'True', 'multi', or a float!")

        # get prediction labels
        if self.device == "cuda" or self.device == "gpu":
            return_type = "tensor"
        else:
            return_type = "list"

        predictions = model.predict(data=data, device=self.device, binary=binary, return_type=return_type)

        if self.device == "cuda" or self.device == "gpu":
            # move tensors to cpu
            y_true = np.array(all_labels)
            y_pred = predictions.detach().cpu().numpy()
        else:
            # convert to np.array
            y_true = np.array(all_labels)
            y_pred = np.array(predictions)

        # calculate classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        # remove junk data
        keys_to_remove = {'accuracy', 'macro avg', 'weighted avg'}
        filtered_dict = {k: v for k, v in report_dict.items() if k not in keys_to_remove}
        # add dict to dataframe
        df = pd.DataFrame(filtered_dict).T[['precision', 'recall', 'f1-score']]

        if title is None:
            title = "Classification Report"

        # plot
        fig = plt.figure(figsize=figsize)
        sns.heatmap(df, annot=True, cmap=palette, fmt=".3f", cbar=True)
        plt.title(title)
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.show()

        plt.show()

    def plot_model(self, model: nn.Module = None, train_loader: DataLoader = None, savepath: str = None,
                   figtype: str = "png"):
        """
        Generate a plot of the torch model architecture.
        :param model: nn.Module
            The torch model to render as image.
        :param train_loader: DataLoader
            The dataset used to train the model. Must be of type torch_geometric.data.DataLoader.
        :param savepath: str
            Savepath of the figure. Do not include the file extension.
        :param figtype: str
            Type of figure to generate. Defaults to "png". The figure type will be appended to the savepath as the file
            extension.
        :return:
        """

        if model is None:
            model = self.model
        if train_loader is None:
            train_loader = self.data

        batch = next(iter(train_loader))

        output = model(batch)
        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

        if savepath:
            dot.render(savepath, format=figtype)

        return dot
