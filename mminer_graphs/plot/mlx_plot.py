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
                 x_data: Optional[Union[list, np.ndarray, mx.array]] = None,
                 y_data: Optional[Union[list, np.ndarray, mx.array]] = None, batch_size: int = 1024,
                 device: str = "auto"):
        """
        Instantiate Plots class. For MLX MLP models only, x_data and Y-data are required. Otherwise, graph data will
        be input using the data argument.
        :param model: Optional[mlx.nn]
            Trained MLX model
        :param data_loader: Optional[Union[DataLoader, list]]
            The test dataset used during training. Can be a Dataloader or a list. If a list is given, it will be
            converted to type Dataloader internally.
         :param x_data: Optional[Union[list, np.ndarray, mx.array]]
            Tabular data features used for MLX MLP models.
        :param y_data: Optional[Union[list, np.ndarray, mx.array]]
            Tabular data labels used for MLX MLP models.
        :param batch_size: int
            Set the batch size for the Dataloader. Only used if the data given is a list.
        :param device: str
            Set the device for the test_loader tensors. Can be "auto", "cuda", "mps", or "cpu".
        """
        self.model = model
        self.device = device
        self.thresholds = None
        self.best_threshold = None

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
               line_color: tuple = ("#1f77b4", "#ff7f0e"), legend_loc: str = "best", clip: float = None,
               figsize: tuple = (14, 5), title: str = None, savepath: str = None):
        """
        Plot the model history. This will generate a single plot containing 2 subplots. It can be any combination of
        'accuracy', 'loss', or 'precision'.
        :param model: Optional[[mlx.nn]
            A trained MLX model.
        :param metrics: list
            A list of metrics to plot. Can only take a combination of 'accuracy', 'loss', or 'precision'.
        :param line_color: tuple
            A tuple containing the line colors.
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
        global metric1, metric2, metric3, metric4, input_metrics
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("MLX model is not fitted!!!")

        # if list, only accept accuracy, loss, or precision
        if isinstance(metrics, list):
            for metric in metrics:
                if metric not in {'accuracy', 'loss', 'precision'}:
                    raise ValueError(f"Only 'loss', 'accuracy', or 'precision' are supported!!")
                else:
                    input_metrics = metrics

        # get data metrics from model
        # default metrics for accuracy and loss
        metric1 = getattr(model.metrics, f"train_{input_metrics[0]}")
        metric2 = getattr(model.metrics, f"test_{input_metrics[0]}")
        metric3 = getattr(model.metrics, f"train_{input_metrics[1]}")
        metric4 = getattr(model.metrics, f"test_{input_metrics[1]}")

        metrics_list = [metric1, metric2, metric3, metric4]

        # optionally clip array to create smoother lines
        if clip:
            for x, metric in enumerate(metrics_list):
                metrics_list[x] = np.clip(metric, 0, clip)

        if title is None:
            title = "Model Training History"

        # plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)
        axes[0].plot(metrics_list[0], color=line_color[0])
        axes[0].plot(metrics_list[1], color=line_color[1])
        axes[0].legend(["Train", "Test"], loc=legend_loc)
        axes[0].set_title(f"Model {input_metrics[0].capitalize()}")
        axes[0].set_ylabel(input_metrics[0].capitalize())
        axes[0].set_xlabel("Epoch")
        axes[0].grid(False)

        axes[1].plot(metrics_list[2], color=line_color[0])
        axes[1].plot(metrics_list[3], color=line_color[1])
        axes[1].legend(["Train", "Test"], loc=legend_loc)
        axes[1].set_title(f"Model {input_metrics[1].capitalize()}")
        axes[1].set_ylabel(input_metrics[1].capitalize())
        axes[1].set_xlabel("Epoch")
        axes[1].grid(False)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def confusion_matrix(self, model: Optional[mlx.nn] = None, data: Optional[Dataloader] = None,
                         class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                         title: str = None, binary: Union[bool, float, str] = 0.5, savepath: str = None):
        """
        Generate a confusion matrix for binary classification MLX models.
        :param model: Optional[mlx.nn]
            A trained MLX model.
        :param data: Dataloader
            A test/validation set for the model. Must be converted into the same format Dataloader as used during MLX
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

        # get labels from test_loader
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

        # check binary input
        if binary == False or binary == "proba":
            raise ValueError("Confusion Matrix can only accept 'True', 'multi', or a float!")

        # get prediction labels
        if data is not None:
            if isinstance(binary, float):
                predictions = model.predict(data=data, binary=binary, return_type="list")
                # apply cutoff
                predictions = [1 if p >= binary else 0 for p in predictions]
            else:
                predictions = model.predict(data=data, binary=binary, return_type="list")
        elif self.x_data is not None and self.y_data is not None:
            # add y labels
            all_labels.append(self.y_data)
            # get predictions from x_data
            if isinstance(binary, float):
                predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
                # apply cutoff
                predictions = [1 if p >= binary else 0 for p in predictions]
            else:
                predictions = model.predict(data=self.x_data, binary=binary, return_type="list")

        # convert to np.array
        y_true = np.array(all_labels).squeeze()  # convert 2D array to 1D
        y_pred = np.array(predictions)

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
                              class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                              title: str = None, binary: Union[bool, float, str] = 0.5, savepath: str = None):
        """
        Generate a classification report for binary classification MLX models.
        :param model: Optional[mlx.nn]
            A trained MLX model.
        :param data: Dataloader
            A test/validation set for the model. Must be converted into the same format Dataloader as used during MLX
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
            raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if title is None:
            title = "Classification Report"
        if class_names is None:
            class_names = ['0', '1']

        # get labels from test_loader
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

        # check binary input
        if binary == False or binary == "proba":
            raise ValueError("Confusion Matrix can only accept 'True', 'multi', or a float!")

        # get prediction labels
        if data is not None:
            if isinstance(binary, float):
                predictions = model.predict(data=data, binary=binary, return_type="list")
                # apply cutoff
                predictions = [1 if p >= binary else 0 for p in predictions]
            else:
                predictions = model.predict(data=data, binary=binary, return_type="list")
        elif self.x_data is not None and self.y_data is not None:
            # add y labels
            all_labels.append(self.y_data)
            # get predictions from x_data
            if isinstance(binary, float):
                predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
                # apply cutoff
                predictions = [1 if p >= binary else 0 for p in predictions]
            else:
                predictions = model.predict(data=self.x_data, binary=binary, return_type="list")

        # convert to np.array
        y_true = np.array(all_labels).squeeze()  # convert 2D array to 1D
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

    def roc_curve(self, model: Optional[Union[mlx.nn, list[mlx.nn]]] = None, data: Optional[Dataloader] = None,
                  model_names: Optional[list] = None, color: Union[str, list] = "orange", figsize: tuple = (8, 6),
                  title: str = None, binary: Union[bool, float, str] = 0.5, savepath: str = None):
        """
        Generate a ROC curve for MLX classification models. If a list of MLX models are given, all curves will be
        plotted on a single plot.
        :param model: Optional[Union[mlx.nn, list[mlx.nn]]]
            Optional. A trained MLX model.
        :param data: Optional[Dataloader]
            A test/validation set for the model. Must be converted into the same format Dataloader as used during torch
            model training.
        :param model_names: Optional[list]
            A list of model names. Only used if model is set to a list of MLX models.
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
            title = "Receiver Operating Characteristic (ROC)"

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
            all_labels = np.array(all_labels).squeeze()
        elif self.y_data is not None:
            all_labels.append(self.y_data)
            all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, return_type="list")
                # get false/true positive rate and thresholds
                fpr, tpr, thresholds = roc_curve(all_labels, predictions)
                roc_auc = roc_auc_score(all_labels, predictions)
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} {roc_auc:.3f}")
        elif data is not None:
            predictions = model.predict(data=data, binary=binary, return_type="list")
            # get false/true positive rate and thresholds
            fpr, tpr, thresholds = roc_curve(all_labels, predictions)
            roc_auc = roc_auc_score(all_labels, predictions)

            # identify the best threshold using Youden's J statistic
            j = tpr - fpr
            x = np.argmax(j)
            best_threshold = thresholds[x]
            b_threshold = f"Best threshold: {best_threshold:.3f}"

            # only save for single models
            self.thresholds = thresholds
            self.best_threshold = b_threshold

            # plot
            plt.plot(fpr, tpr, color=color, lw=2, label=f"ROC curve {roc_auc:.3f}")
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
            self.thresholds = thresholds
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
                 model_names: Optional[list] = None, chance_level: bool = False, color: Union[str, list] = "orange",
                 figsize: tuple = (8, 6), title: str = None, binary: Union[bool, float, str] = 0.5,
                 savepath: str = None):
        """
        Generate a Precision=Recall curve for MLX classification models. If a list of MLX models are given, all curves
        will be plotted on a single plot.
        :param model: Optional[Union[mlx.nn, list[mlx.nn]]]
            Optional. A trained MLX model.
        :param data: Optional[Dataloader]
            A test/validation set for the model. Must be converted into the same format Dataloader as used during torch
            model training.
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
            all_labels = np.array(all_labels).squeeze()
        elif self.y_data is not None:
            all_labels.append(self.y_data)
            all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                predictions = x.predict(data=data, binary=binary, return_type="list")
                # get prec/recall/pr_threshold
                prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
                avg_prec = average_precision_score(all_labels, predictions)
                plt.plot(recall, prec, color=c, lw=2, label=f"{name} {avg_prec:.3f}")
        elif data is not None:
            predictions = model.predict(data=data, binary=binary, return_type="list")
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
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.show()

        plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
