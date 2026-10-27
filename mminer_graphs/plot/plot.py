import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Any
from numpy import dtype, ndarray
from torchviz import make_dot
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


class Plots:
    def __init__(self, model: Optional[torch.nn] = None, data_loader: Optional[Union[DataLoader, list]] = None,
                 add_feats: Optional[Union[np.ndarray, list, int]] = None,
                 x_data: Optional[Union[list, np.ndarray]] = None, y_data: Optional[Union[list, np.ndarray]] = None,
                 batch_size: Optional[int] = None, device: str = "auto", font: str = "Arial", font_size: int = 12):
        """
        Instantiate Plots class. For MLP models only, x_test and y_test are required. Otherwise, graph data will be
        input using the data argument.
        :param model: Optional[torch.nn]
            Trained torch model
        :param data_loader: Optional[DataLoader]
            The test dataset used during training. Can be a DataLoader or a list. If a list is given, it will be
            converted to type DataLoader internally.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param x_data: Optional[Union[list, np.ndarray]]
            Tabular data features used for torch MLP models.
        :param y_data: Optional[Union[list, np.ndarray]]
            Tabular data labels used for torch MLP models.
        :param batch_size: Optional[int]
            Set the batch size for the Dataloader. Only used if the data given is a list.
        :param device: str
            Set the device for the test_loader tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param font: str
            Set the font type of the image.
        :param font_size: int
            Set the figure font size
        """
        self.model = model
        self.device = device
        self.thresholds = None
        self.best_threshold = None
        self.add_feats = add_feats

        # set font
        plt.rcParams['font.family'] = font
        plt.rcParams['font.size'] = font_size

        if data_loader is not None:
            self.data = data_loader

        # if tabular MLP data
        if x_data and y_data is not None:
            # check dataset input:
            if all(isinstance(x, np.ndarray) for x in (x_data, y_data)):
                pass  # can convert to tensor
            elif all(isinstance(x, list) for x in (x_data, y_data)):  # convert list to array
                x_data, y_data = (np.array(x_data), np.array(y_data))

            # convert to tensor
            x_data_tensor = torch.tensor(x_data, dtype=torch.float32, device=device)
            y_data_tensor = torch.tensor(y_data, dtype=torch.long, device=device)

            # wrap tensors into a Dataset
            test_dataset = TensorDataset(x_data_tensor, y_data_tensor)

            # load data
            test_loader = TorchDataLoader(test_dataset, batch_size=batch_size)
            self.data = test_loader
            self.x_data = x_data
            self.y_data = y_data
            self.tabular = True
        else:
            self.x_data = None
            self.y_data = None
            self.tabular = False

    def curves(self, model: Optional[torch.nn] = None, metrics: list = ["loss", "accuracy"],
               line_color: tuple = ("#1f77b4", "#ff7f0e"), nrows=1, ncols: int = 2, legend_loc: str = "best",
               clip: float = None, figsize: tuple = (14, 5), title: str = None, savepath: str = None):
        """
        Plot the model history. This will generate a single plot containing 2 subplots. It can be any combination of
        'accuracy', 'loss', or 'precision'.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param metrics: list
            A list of metrics to plot. Can only take a combination of 'accuracy', 'loss', or 'precision'.
        :param line_color: tuple
            A tuple containing the line colors.
         :param nrows: int
            Set the number of rows for the plot.
        :param ncols: int
            Set the number of columns for the plot.
        :param legend_loc: str
            Set the location of the figure legends.
        :param clip: float
            Remove extreme outliers to produce smoother curve.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None, the default title will be used.
        :param savepath: str
            Set the savepath for the image.
        :return:
        """
        global input_metrics
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("Torch model is not fitted!!!")

        # get data metrics from model
        metric_list = list(model.__dict__)

        # if list, only accept accuracy, loss, or precision, mae, rmse, or r2
        if isinstance(metrics, list):
            for metric in metrics:
                if metric not in {'loss', 'accuracy', 'precision', 'mae', 'rmse', 'r2'}:
                    raise ValueError(f"Only 'loss', 'accuracy', 'precision','mae', 'rmse', or 'r2' are supported!!")
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
        if len(input_metrics) > 1:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            flat_axes = axes.flatten()  # for dynamic plotting
            fig.suptitle(title)
            for x in range(len(input_metrics)):
                # check if num of plots is too many
                if len(flat_axes) < len(input_metrics):
                    raise ValueError("Number of metrics does not match ncols!")

                # get metrics from model
                train_metric = getattr(model.metrics, f"train_{input_metrics[x]}")
                test_metric = getattr(model.metrics, f"test_{input_metrics[x]}")

                # optionally clip array to create smoother lines
                if clip:
                    train_metric = np.clip(train_metric, 0, clip)
                    test_metric = np.clip(test_metric, 0, clip)

                # set plot as axis
                current_ax = flat_axes[x]
                current_ax.plot(train_metric, color=line_color[0])
                current_ax.plot(test_metric, color=line_color[1])
                current_ax.legend(["Train", "Test"], loc=legend_loc)
                current_ax.set_title(f"Model {input_metrics[x].capitalize()}")
                current_ax.set_ylabel(input_metrics[x].capitalize())
                current_ax.set_xlabel("Epoch")
                current_ax.grid(False)

            # hide potential empty plots
            total_plots = nrows * ncols
            for x in range(len(input_metrics), total_plots):
                flat_axes[x].axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title)
            # get metrics from model
            train_metric = getattr(model.metrics, f"train_{input_metrics[0]}")
            test_metric = getattr(model.metrics, f"test_{input_metrics[0]}")

            # optionally clip array to create smoother lines
            if clip:
                train_metric = np.clip(train_metric, 0, clip)
                test_metric = np.clip(test_metric, 0, clip)

            # set plot as axis
            ax.plot(train_metric, color=line_color[0])
            ax.plot(test_metric, color=line_color[1])
            ax.legend(["Train", "Test"], loc=legend_loc)
            ax.set_title(f"Model {input_metrics[0].capitalize()}")
            ax.set_ylabel(input_metrics[0].capitalize())
            ax.set_xlabel("Epoch")
            ax.grid(False)

        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def regression(self, model: Optional[torch.nn] = None, data: Optional[Union[DataLoader, list]] = None,
                   add_feats: Optional[Union[np.array, list, int]] = None, batch_size: int = 1024,
                   dot_color: str = "dodgerblue", dot_alpha: float = 0.6, line_color: str = "red", unity: bool = True,
                   metric_output: Optional[Union[list, str, None]] = ['R2', 'MAE', 'RMSE'], legend_loc: str = "best",
                   title: str = None, xlabel: str = None, ylabel: str = None, figsize: tuple = (8, 5),
                   savepath: str = None):
        """
        Generate a linear regression plot using the model and the input data_loader.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param data: Optional[DataLoader, list]
            A list of metrics to plot. Can only take a combination of 'accuracy', 'loss', or 'precision'.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param batch_size: int
            Set the batch size for the Dataloader. Only used if the data given is a list. Only used if data_loader is
            not given at initialization of PlotsMLX() class.
        :param metric_output: Optional[Union[list, str, None]]
            Determine whether to draw the regression metrics with the figure legend.
        :param dot_color: str
            Set the color of the scatter plot points. Defaults to 'dodgerblue'.
        :param dot_alpha: float
            Set the transparency of the scatter plot points. Defaults to 0.6.
        :param line_color: str
            Set the color of the regression line or line of best fit. Defaults to 'red'.
        :param unity: bool
            Optional. Draw the line of unity. Line represents (predicted value = actual value). If points fall above, model may
            over predict. If points fall below, model may under predict. Defaults to True.
        :param legend_loc: str
            Set the location of the figure legend. Defaults to 'best'.
        :param title: str
            Set the title of the figure. Defaults to None.
        :param xlabel: str
            Set the xlabel of the figure. Defaults to None.
        :param ylabel: str
            Set the ylabel of the figure. Defaults to None.
        :param figsize: tuple
            Set the figure size. Defaults to (8, 5).
        :param savepath: str
            Set the path to save the figure. Defaults to None.
        :return:
        """

        if model is None:
            model = self.model
        if not self.tabular:
            data = self.data
        elif self.tabular:
            data = None
        if add_feats is None:
            add_feats = self.add_feats

        # make predictions
        if data is not None:
            preds = model.predict(data, device=self.device, binary=False, add_feats=add_feats, return_type="tensor")
        elif self.x_data is not None and self.y_data is not None:
            preds = model.predict(self.x_data, device=self.device, binary=False, return_type="tensor")

        # convert tensor to array and flatten
        preds = preds.cpu().detach().numpy()
        preds = preds.flatten()

        # extract y value from DataLoader
        labels = []
        if data is not None:
            for batch in data:
                try:  # to catch if tabular data is given
                    y = batch.y
                except Exception as e:
                    raise AttributeError(f"{e}\nDid you input 'tabular' data for MLP?")
                y_list = y.flatten().tolist()
                labels.extend(y_list)
        elif self.x_data is not None and self.y_data is not None:
            labels = self.y_data

        # add metrics to legend
        if metric_output is not None:
            # initialize vars
            r2 = mae = rmse = None

            # calculate metrics
            best_epoch = model.metrics.best_epoch
            if 'R2' in metric_output:
                r2 = round(model.metrics.test_r2[best_epoch], 3)
            if 'MAE' in metric_output:
                mae = round(model.metrics.test_mae[best_epoch], 3)
            if 'RMSE' in metric_output:
                rmse = round(model.metrics.test_rmse[best_epoch], 3)

            # build string
            text = ""
            if r2 is not None:
                text += f"R²: {r2}\n"
            if mae is not None:
                text += f"MAE: {mae}\n"
            if rmse is not None:
                text += f"RMSE: {rmse}"
        else:
            text = None

        # plot
        fig = plt.figure(figsize=figsize)
        sns.scatterplot(
            x=labels,
            y=preds,
            alpha=dot_alpha,
            color=dot_color,
            # label='Model Predictions'  # Label the scatter points explicitly
        )
        # scatter_kws={'alpha': dot_alpha, 'color': dot_color},
        sns.regplot(x=labels, y=preds, scatter=False,
                    line_kws={'color': line_color}, label='Regression')

        # line of unity
        # y = x, so both are set to actual labels
        unity_y = np.linspace(min(labels), max(labels), 20)  # generate evenly spaced plot points
        unity_x = unity_y
        if unity:
            plt.plot(unity_x,  # x axis
                     unity_y,  # y-axis
                     linestyle='--',
                     linewidth=2,
                     color='0.7',
                     label='Line of Unity')

        # empty plot for label legend
        if text is None:
            plt.plot([], [], linestyle='None', marker='None')
        else:
            plt.plot([], [], linestyle='None', marker='None', label=text)

        if xlabel is None:
            xlabel = "Actual"
        if ylabel is None:
            ylabel = "Prediction"
        if title is None:
            title = "Prediction vs. Actual"

        # plot features
        plt.grid(False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=legend_loc, frameon=False)
        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def confusion_matrix(self, model: Optional[torch.nn] = None, data: Optional[DataLoader] = None,
                         add_feats: Optional[Union[np.array, list, int]] = None,
                         class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                         title: str = None, cutoff: float = 0.5, savepath: str = None):
        """
        Generate a confusion matrix for binary classification torch models.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param data: DataLoader
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to cutoff.
        :param palette: str
            Set the color palette for the figure.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None given, the default title will be used.
        :param cutoff: float
            Set the cutoff for binary predictions.
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
        if add_feats is None:
            add_feats = self.add_feats

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

        # output format
        if self.device == "cuda" or self.device == "gpu":
            return_type = "tensor"
        else:
            return_type = "list"

        # get prediction labels
        y_true, y_pred = self._get_prediction_labels(add_feats, all_labels, cutoff, data, model, return_type)

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
                              add_feats: Optional[Union[np.array, list, int]] = None,
                              class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                              title: str = None, cutoff: float = 0.5, savepath: str = None):
        """
        Generate a classification report for binary classification torch models.
        :param model: Optional[torch.nn]
            A trained torch model.
        :param data: DataLoader
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to cutoff.
        :param palette: str
            Set the color palette for the figure.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None given, the default title will be used.
        :param cutoff: float
            Set the cutoff for binary predictions.
        :param savepath: str
            Set the figure savepath.
        :return:
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
        if add_feats is None:
            add_feats = self.add_feats

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

        # output format
        if self.device == "cuda" or self.device == "gpu":
            return_type = "tensor"
        else:
            return_type = "list"

        # get prediction labels
        y_true, y_pred = self._get_prediction_labels(add_feats, all_labels, cutoff, data, model, return_type)

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

    def _get_prediction_labels(self, add_feats: list | int | None | Any, all_labels: list[Any], cutoff: float,
                               data: Optional[Union[DataLoader, Any]], model: Any | None,
                               return_type: str) -> tuple[ndarray[Any, dtype[Any]], Any]:
        """
        Support function for confusion matrix and classification report.
        """
        if data is not None:
            try:  # todo check if there is a more elegant way to do this
                if isinstance(cutoff, float):
                    predictions = model.predict(data=data, device=self.device, binary=False, add_feats=add_feats,
                                                cutoff=cutoff, return_type=return_type)
                    # apply cutoff
                    predictions = [1 if p >= cutoff else 0 for p in predictions]
                else:
                    predictions = model.predict(data=data, device=self.device, binary=False, add_feats=add_feats,
                                                cutoff=cutoff, return_type=return_type)
            except:
                if isinstance(cutoff, float):
                    predictions = model.predict(data=data, device=self.device, binary=False, cutoff=cutoff,
                                                return_type=return_type)
                    # apply cutoff
                    predictions = [1 if p >= cutoff else 0 for p in predictions]
                else:
                    predictions = model.predict(data=data, device=self.device, binary=False, cutoff=cutoff,
                                                return_type=return_type)

            if self.device == "cuda" or self.device == "gpu":
                # move tensors to cpu
                y_true = np.array(all_labels)
                y_pred = predictions.detach().cpu().numpy()
            else:
                # convert to np.array
                y_true = np.array(all_labels)
                y_pred = np.array(predictions)
        else:
            raise ValueError("No data provided???")
        return y_true, y_pred

    def roc_curve(self, model: Optional[Union[torch.nn, list[torch.nn]]] = None, data: Optional[DataLoader] = None,
                  add_feats: Optional[Union[np.array, list, int]] = None,
                  model_names: Optional[list] = None, color: Union[str, list] = "orange", figsize: tuple = (8, 6),
                  title: str = None, binary: Union[bool, float, str] = False, savepath: str = None):
        """
        Generate a ROC curve for torch classification models. If a list of torch models are given, all curves will be
        plotted on a single plot.
        :param model: Optional[Union[torch.nn, list[torch.nn]]]
            Optional. A trained torch model.
        :param data: Optional[DataLoader]
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param model_names: Optional[list]
            A list of model names. Only used if model is set to a list of torch models.
        :param color: Union[str, list]
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
        if not isinstance(model, list):
            if not model.is_fitted:
                raise ValueError("Torch model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Receiver Operating Characteristic (ROC)"
        if add_feats is None:
            add_feats = self.add_feats

        # get labels from test_loader
        all_labels = []
        # get labels
        for datapoint in data:
            if hasattr(datapoint, "y"):
                label_tensor = datapoint.y
            elif hasattr(datapoint, "label"):
                label_tensor = datapoint.label
            elif hasattr(datapoint, "graph_labels"):
                label_tensor = datapoint.graph_labels
            # if input is tabular
            elif self.tabular:
                label_tensor = datapoint[1]
                data = self.x_data  # ensure data is tabular
            else:
                raise ValueError("Cannot find labels! Only looking for 'y', 'label', or 'graph_labels'")

            # move to cpu
            if isinstance(label_tensor, torch.Tensor):
                label_np = label_tensor.detach().cpu().numpy().flatten()
                all_labels.extend(label_np)
            else:
                all_labels.append(label_tensor)

        all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, device=self.device, add_feats=add_feats,
                                        return_type="list")
                # get false/true positive rate and thresholds
                fpr, tpr, thresholds = roc_curve(all_labels, predictions)
                roc_auc = roc_auc_score(all_labels, predictions)
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} {roc_auc:.3f}")
        else:
            try:
                predictions = model.predict(data=data, binary=binary, device=self.device, add_feats=add_feats,
                                            return_type="list")
            except TypeError as e:
                if "add_feats" in str(e):
                    predictions = model.predict(data=data, binary=binary, device=self.device, return_type="list")
                else:
                    raise e
            # get false/true positive rate and thresholds
            fpr, tpr, thresholds = roc_curve(all_labels, predictions)
            roc_auc = roc_auc_score(all_labels, predictions)

            # identify the best threshold using Youden's J statistic
            j = tpr - fpr
            x = np.argmax(j)
            best_threshold = thresholds[x]
            b_threshold = f"Best threshold: {best_threshold:.3f}"

            # only save for single models
            self.thresholds = best_threshold
            self.best_threshold = b_threshold

            # plot
            plt.plot(fpr, tpr, color=color, lw=2, label=f"ROC curve {roc_auc:.3f}")

        plt.plot([0, 1.05], [0, 1.05], color="gray", linestyle=(0, (5, 10)), lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.show()

        plt.show()

    def pr_curve(self, model: Optional[Union[torch.nn, list[torch.nn]]] = None, data: Optional[DataLoader] = None,
                 add_feats: Optional[Union[np.array, list, int]] = None, model_names: Optional[list] = None,
                 chance_level: bool = False, color: Union[str, list] = "orange", figsize: tuple = (8, 6),
                 title: str = None, binary: Union[bool, float, str] = False, savepath: str = None):
        """
        Generate a Precision=Recall curve for torch classification models. If a list of torch models are given, all
        curves will be plotted on a single plot.
        :param model: Optional[Union[torch.nn, list[torch.nn]]]
            Optional. A trained torch model.
        :param data: Optional[DataLoader]
            A test/validation set for the model. Must be converted into the same format DataLoader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param model_names: Optional[list]
            A list of model names. Only used if model is set to a list of torch models.
        :param chance_level: bool
            Whether to plot a chance level curve. This is the positive label predictions over total samples. The chance
            level indicates precision of randomly guessed positives labels.
        :param color: Union[str, list]
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
        if not isinstance(model, list):
            if not model.is_fitted:
                raise ValueError("Torch model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Precision-Recall Curve"
        if add_feats is None:
            add_feats = self.add_feats

        # get labels from test_loader
        all_labels = []
        # get labels
        for datapoint in data:
            if hasattr(datapoint, "y"):
                label_tensor = datapoint.y
            elif hasattr(datapoint, "label"):
                label_tensor = datapoint.label
            elif hasattr(datapoint, "graph_labels"):
                label_tensor = datapoint.graph_labels
            # if input is tabular
            elif self.tabular:
                label_tensor = datapoint[1]
                data = self.x_data  # ensure data is tabular
            else:
                raise ValueError("Cannot find labels! Only looking for 'y', 'label', or 'graph_labels'")

            # move to cpu
            if isinstance(label_tensor, torch.Tensor):
                label_np = label_tensor.detach().cpu().numpy().flatten()
                all_labels.extend(label_np)
            else:
                all_labels.append(label_tensor)

        all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, device=self.device, add_feats=add_feats,
                                        return_type="list")
                # get false/true positive rate and thresholds
                fpr, tpr, thresholds = roc_curve(all_labels, predictions)
                roc_auc = roc_auc_score(all_labels, predictions)
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} {roc_auc:.3f}")
        else:
            try:
                predictions = model.predict(data=data, binary=binary, device=self.device, add_feats=add_feats,
                                            return_type="list")
            except TypeError as e:
                if "add_feats" in str(e):
                    predictions = model.predict(data=data, binary=binary, device=self.device, return_type="list")
                else:
                    raise e
            # get prec/recall/pr_threshold
            prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
            avg_prec = average_precision_score(all_labels, predictions)

            # plot
            plt.plot(recall, prec, color=color, lw=2, label=f"Precision-Recall Avg: {avg_prec:.3f}")

        if chance_level:
            chance = all_labels.mean()
            plt.axhline(y=chance, color="gray", linestyle=(0, (5, 10)), lw=1, label=f"Chance Level: {chance:.3f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="best")
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
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
