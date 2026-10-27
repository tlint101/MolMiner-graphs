import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union
from mlx_graphs.loaders import Dataloader
import mlx
import mlx.core as mx
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


class PlotsMLX:
    def __init__(self, model: Optional[mlx.nn] = None, data_loader: Optional[Union[Dataloader, list]] = None,
                 add_feats: Optional[Union[np.array, list, int]] = None,
                 x_data: Optional[Union[list, np.ndarray, mx.array]] = None,
                 y_data: Optional[Union[list, np.ndarray, mx.array]] = None, batch_size: int = 1024,
                 device: str = "auto", font: str = "Arial", font_size: int = 12):
        """
        Instantiate Plots class. For MLX MLP models only, x_data and Y-data are required. Otherwise, graph data will
        be input using the data argument.
        :param model: Optional[mlx.nn]
            Trained MLX model
        :param data_loader: Optional[Union[DataLoader, list]]
            The test dataset used during training. Can be a Dataloader or a list. If a list is given, it will be
            converted to type Dataloader internally.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
         :param x_data: Optional[Union[list, np.ndarray, mx.array]]
            Tabular data features used for MLX MLP models.
        :param y_data: Optional[Union[list, np.ndarray, mx.array]]
            Tabular data labels used for MLX MLP models.
        :param batch_size: int
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

        # for graph models, check and convert into Dataloader
        if data_loader is not None:
            if isinstance(data_loader, list):
                self.data = Dataloader(data_loader, batch_size=batch_size, shuffle=False)
            elif isinstance(data_loader, Dataloader):
                self.data = data_loader
        else:
            self.data = None

        # if tabular MLP data
        if x_data is not None and y_data is not None:
            # check dataset input:
            if all(isinstance(x, np.ndarray) for x in (x_data, y_data)):
                pass  # can convert to tensor
            elif all(isinstance(x, list) for x in (x_data, y_data)):  # convert list to array
                x_data, y_data = (np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32))
                x_data, y_data = (mx.array(x_data), mx.array(y_data))
            elif all(isinstance(x, mx.array) for x in (x_data, y_data)):
                pass
            self.x_data = x_data
            self.y_data = y_data
        else:
            self.x_data = None
            self.y_data = None

    def curves(self, model: Optional[mlx.nn] = None, metrics: list = ["loss", "accuracy"],
               line_color: tuple = ("#1f77b4", "#ff7f0e"), nrows=1, ncols: int = 2, legend_loc: str = "best",
               clip: float = None, figsize: tuple = (14, 5), title: str = None, savepath: str = None):
        """
        Plot the model history. This will generate a single plot containing 2 subplots. It can be any combination of
        'accuracy', 'loss', or 'precision'.
        :param model: Optional[mlx.nn]
            A trained MLX model.
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
            raise ValueError("MLX model is not fitted!!!")

        # if list, only accept accuracy, loss, or precision
        if isinstance(metrics, list):
            for metric in metrics:
                if metric not in {'loss', 'accuracy', 'precision', 'mae', 'rmse', 'r2'}:
                    raise ValueError(f"Only 'loss', 'accuracy', or 'precision' are supported!!")
                else:
                    input_metrics = metrics

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

    def regression(self, model: Optional[mlx.nn] = None, data: Optional[Dataloader] = None,
                   add_feats: Optional[Union[np.array, list, int]] = None, batch_size: int = 1024,
                   dot_color: str = "dodgerblue", dot_alpha: float = 0.6, line_color: str = "red", unity: bool = True,
                   metric_output: Optional[Union[list, str, None]] = ['R2', 'MAE', 'RMSE'], legend_loc: str = "best",
                   title: str = None, xlabel: str = None, ylabel: str = None, figsize: tuple = (8, 5),
                   savepath: str = None):
        """
        Generate a linear regression plot using the model and the input data_loader.
        :param model: Optional[mlx.nn]
         A trained MLX model.
        :param data: Optional[Dataloader]
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
        if data is None:
            data = self.data
        if add_feats is None:
            add_feats = self.add_feats

        # make predictions
        if data is not None:
            preds = model.predict(data, batch_size=batch_size, binary=None, add_feats=add_feats, return_type="list")
        elif self.x_data is not None and self.y_data is not None:
            preds = model.predict(self.x_data, batch_size=batch_size, binary=None, return_type="list")

        labels = []
        # extract y value from Dataloader
        if data is not None:
            for batch in data:
                y = batch.y
                y_list = y.flatten().tolist()
                labels.extend(y_list)
        elif self.x_data is not None and self.y_data is not None:
            y_list = self.y_data.flatten().tolist()
            labels.extend(y_list)

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

    def confusion_matrix(self, model: Optional[mlx.nn] = None, data: Optional[Dataloader] = None,
                         batch_size: int = None, add_feats: Optional[Union[np.array, list, int]] = None,
                         class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                         title: str = None, cutoff: float = 0.5, savepath: str = None):
        """
        Generate a confusion matrix for binary classification MLX models.
        :param model: Optional[mlx.nn]
            A trained MLX model.
        :param data: Dataloader
            A test/validation set for the model. Must be converted into the same format Dataloader as used during MLX
            model training.
        :param batch_size: int
            Set the batch size for prediction. Only used if the input data is of type list.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to binary.
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

        global predictions
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Confusion Matrix on Test Dataset"
        if class_names is None:
            class_names = ['0', '1']
        if add_feats is None:
            add_feats = self.add_feats

        # get prediction labels
        y_true, y_pred = self._get_prediction_labels(model, data, cutoff, batch_size, add_feats)

        # calculate matrix
        cm_matrix = confusion_matrix(y_true, y_pred)

        # get matrix labels
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = [value for value in cm_matrix.flatten()]
        group_percentages = [f"{round(value * 100, 3)}%" for value in cm_matrix.flatten() / np.sum(cm_matrix)]
        labels = [f"{v1}\n\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

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

    def classification_report(self, model: Optional[mlx.nn] = None, data: Optional[Dataloader] = None,
                              batch_size: int = None, add_feats: Optional[Union[np.array, list, int]] = None,
                              class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                              title: str = None, cutoff: float = 0.5, savepath: str = None):
        """
        Generate a classification report for binary classification MLX models.
        :param model: Optional[mlx.nn]
            A trained MLX model.
        :param data: Dataloader
            A test/validation set for the model. Must be converted into the same format Dataloader as used during MLX
            model training.
        :param batch_size: int
            Set the batch size for prediction. Only used if the input data is of type list.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param class_names: Optional[list]
            A list of names for the classes. Defaults to binary.
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
            raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Classification Report"
        if class_names is None:
            class_names = ['0', '1']
        if add_feats is None:
            add_feats = self.add_feats

        # get prediction labels
        y_true, y_pred = self._get_prediction_labels(model, data, cutoff, batch_size, add_feats)

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

    def _get_prediction_labels(self, model, data, cutoff, batch_size, add_feats):
        """
        Support function for confusion matrix and classification report.
        """
        # get labels from test_loader
        global predictions
        all_labels = []
        # get labels
        if isinstance(data, Dataloader):
            for datapoint in data:
                if hasattr(datapoint, "y"):
                    all_labels.append(datapoint.y)
                elif hasattr(datapoint, "label"):
                    all_labels.append(datapoint.label)
                elif hasattr(datapoint, "graph_labels"):
                    all_labels.append(datapoint.graph_labels)
                else:
                    raise ValueError("Cannot find labels! Only looking for 'y', 'label', or 'graph_labels'")
            # combine mx.array into single array
            all_labels = mx.concatenate(all_labels, axis=0)

        # get prediction labels
        if data is not None:
            if isinstance(cutoff, float):
                predictions = model.predict(data=data, batch_size=batch_size, add_feats=add_feats, binary=True,
                                            cutoff=cutoff, return_type="list")
                # apply cutoff
                predictions = [1 if p >= cutoff else 0 for p in predictions]
            else:
                predictions = model.predict(data=data, batch_size=batch_size, add_feats=add_feats, binary=True,
                                            cutoff=cutoff, return_type="list")
        elif self.x_data is not None and self.y_data is not None:
            # add y labels
            all_labels.append(self.y_data)
            # get predictions from x_data
            if isinstance(cutoff, float):
                predictions = model.predict(data=self.x_data, batch_size=batch_size, binary=True, cutoff=cutoff,
                                            return_type="list")
                # apply cutoff
                predictions = [1 if p >= cutoff else 0 for p in predictions]
            else:
                predictions = model.predict(data=self.x_data, batch_size=batch_size, binary=True, cutoff=cutoff,
                                            return_type="list")

        # convert to np.array
        y_true = np.array(all_labels).squeeze()  # convert 2D array to 1D
        y_pred = np.array(predictions)

        return y_true, y_pred

    def roc_curve(self, model: Optional[Union[mlx.nn, list[mlx.nn]]] = None, data: Optional[Dataloader] = None,
                  add_feats: Optional[Union[np.array, list, int]] = None, model_names: Optional[list] = None,
                  color: Union[str, list] = "orange", figsize: tuple = (8, 6), title: str = None,
                  binary: Union[bool, float, str] = False, savepath: str = None):
        """
        Generate a ROC curve for MLX classification models. If a list of MLX models are given, all curves will be
        plotted on a single plot.
        :param model: Optional[Union[mlx.nn, list[mlx.nn]]]
            Optional. A trained MLX model.
        :param data: Optional[Dataloader]
            A test/validation set for the model. Must be converted into the same format Dataloader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param model_names: Optional[list]
            A list of model names. Only used if model is set to a list of MLX models.
        :param color: Union[str, list]
            Set the color palette for the figure.
        :param figsize: tuple
            Set the figure size.
        :param title: str
            Set the title of the figure. If None given, the default title will be used.
        param binary: Union[bool, float, str]
            Set the prediction output. Can only accept 'True', 'multi', or a float as inputs.
        :param savepath: str
            Set the figure savepath.
        :return:
        """
        if model is None:
            model = self.model
        if not isinstance(model, list):
            if not model.is_fitted:
                raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Receiver Operating Characteristic (ROC)"
        if add_feats is None:
            add_feats = self.add_feats

        # get labels from test_loader
        all_labels = []
        # get labels
        if data is not None:
            for datapoint in data:
                if hasattr(datapoint, "y"):
                    all_labels.append(datapoint.y)
                elif hasattr(datapoint, "label"):
                    all_labels.append(datapoint.label)
                elif hasattr(datapoint, "graph_labels"):
                    all_labels.append(datapoint.graph_labels)
                else:
                    raise ValueError("Cannot find labels! Only looking for 'y', 'label', or 'graph_labels'")
            # flatten labels to 1D array
            all_labels = np.concatenate(all_labels).ravel()
        elif self.y_data is not None:
            all_labels.append(self.y_data)
            all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
                # get false/true positive rate and thresholds
                fpr, tpr, thresholds = roc_curve(all_labels, predictions)
                roc_auc = roc_auc_score(all_labels, predictions)
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} {roc_auc:.3f}")
            self.best_threshold = "Model given as a list. No 'best threshold' calculated!"
        elif data is not None:
            predictions = model.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
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
        # for mlp
        else:
            predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
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

    def pr_curve(self, model: Optional[Union[mlx.nn, list[mlx.nn]]] = None, data: Optional[Dataloader] = None,
                 add_feats: Optional[Union[np.array, list, int]] = None, model_names: Optional[list] = None,
                 chance_level: bool = False, color: Union[str, list] = "orange", figsize: tuple = (8, 6),
                 title: str = None, binary: Union[bool, float, str] = False, savepath: str = None):
        """
        Generate a Precision=Recall curve for MLX classification models. If a list of MLX models are given, all curves
        will be plotted on a single plot.
        :param model: Optional[Union[mlx.nn, list[mlx.nn]]]
            Optional. A trained MLX model.
        :param data: Optional[Dataloader]
            A test/validation set for the model. Must be converted into the same format Dataloader as used during torch
            model training.
        :param add_feats: Optional[Union[np.array, list, int]]
            The number of additional features to be added. Must be the same as what was given during training.
        :param model_names: Optional[list]
            A list of model names. Only used if model is set to a list of MLX models.
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
                raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Precision-Recall Curve"
        if add_feats is None:
            add_feats = self.add_feats

        # get labels from test_loader
        all_labels = []
        # get labels
        if data is not None:
            for datapoint in data:
                if hasattr(datapoint, "y"):
                    all_labels.append(datapoint.y)
                elif hasattr(datapoint, "label"):
                    all_labels.append(datapoint.label)
                elif hasattr(datapoint, "graph_labels"):
                    all_labels.append(datapoint.graph_labels)
                else:
                    raise ValueError("Cannot find labels! Only looking for 'y', 'label', or 'graph_labels'")
            # flatten labels to 1D array
            all_labels = np.concatenate(all_labels).ravel()
        elif self.y_data is not None:
            all_labels.append(self.y_data)
            all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
                # get prec/recall/pr_threshold
                prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
                avg_prec = average_precision_score(all_labels, predictions)
                plt.plot(recall, prec, color=c, lw=2, label=f"{name} {avg_prec:.3f}")
        elif data is not None:
            predictions = model.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
            # get prec/recall/pr_threshold
            prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
            avg_prec = average_precision_score(all_labels, predictions)

            # plot
            plt.plot(recall, prec, color=color, lw=2, label=f"Precision-Recall Avg: {avg_prec:.3f}")
        else:
            predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
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
        plt.xlim([0, 1.05])
        # plt.ylim([0, 1])
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.show()

        plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
