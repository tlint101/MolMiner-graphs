import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union
from mlx_graphs.loaders import Dataloader
import mlx
import mlx.core as mx
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, r2_score, mean_absolute_error, root_mean_squared_error


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
            if not isinstance(x_data, (np.ndarray, mx.array)):
                x_data = np.asarray(x_data, dtype=np.float32)
            if not isinstance(y_data, (np.ndarray, mx.array)):
                y_data = np.asarray(y_data)
            self.x_data = x_data
            self.y_data = y_data
        else:
            self.x_data = None
            self.y_data = None

    @staticmethod
    def _resolve_task(model, task: Optional[str], required: bool = True, expected_type: Optional[str] = None) -> Optional[str]:
        """Validate `task` argument for MLX MultiTaskGNN vs single-task models."""
        if getattr(model, "task", None) != "MultiTask":
            if task is not None:
                raise ValueError("The 'task' parameter is only supported for MultiTaskGNN models.")
            return None

        if task is None:
            if required:
                raise ValueError(f"Please specify task=... for MultiTaskGNN (available tasks: {model.task_names}).")
            return None

        if task not in model.task_names:
            raise ValueError(f"Unknown task {task!r}; available tasks: {model.task_names}.")

        if expected_type is not None:
            actual_type = model.task_configs[task]["type"]
            if actual_type != expected_type:
                raise ValueError(
                    f"Task {task!r} is a '{actual_type}' task, expected '{expected_type}' for this plot."
                )
        return task

    @staticmethod
    def _unshuffled_graph_loader(data, batch_size: Optional[int] = None):
        """Return a deterministic non-shuffled MLX Dataloader for graph data."""
        bs = batch_size if batch_size is not None else 1024
        if isinstance(data, Dataloader):
            return Dataloader(data.dataset, batch_size=data.batch_size, shuffle=False)
        if isinstance(data, list):
            return Dataloader(data, batch_size=bs, shuffle=False)
        return data

    @staticmethod
    def _extract_multitask_targets(data, task_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract column `task_idx` of multi-target labels and boolean observed mask from MLX graph batches."""
        ys, masks = [], []
        for batch in data:
            y = np.array(batch.y)
            if y.ndim == 1:
                y = y[:, None]
            ys.append(y)
            if hasattr(batch, "y_mask") and batch.y_mask is not None:
                m = np.array(batch.y_mask).astype(bool)
                if m.ndim == 1:
                    m = m[:, None]
                masks.append(m)
            else:
                masks.append(~np.isnan(y))
        y_all = np.concatenate(ys, axis=0)
        m_all = np.concatenate(masks, axis=0)
        return y_all[:, task_idx], m_all[:, task_idx].astype(bool)

    def curves(self, model: Optional[mlx.nn] = None, metrics: list = ["loss", "accuracy"],
               line_color: tuple = ("#1f77b4", "#ff7f0e"), nrows=1, ncols: int = 2, legend_loc: str = "best",
               clip: float = None, figsize: tuple = (14, 5), title: str = None, task: Optional[str] = None,
               savepath: str = None):
        """
        Plot the model history. For MultiTaskGNN, pass `task="<task_name>"` to plot per-task metrics,
        or omit `task` to plot aggregate `metrics=["loss"]`.
        """
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("MLX model is not fitted!!!")

        resolved_task = self._resolve_task(model, task, required=False)
        metrics_obj = model.metrics.tasks[resolved_task] if resolved_task is not None else model.metrics

        # normalize and validate requested metrics
        valid_metrics = {'loss', 'accuracy', 'precision', 'mae', 'rmse', 'r2'}
        if isinstance(metrics, str):
            input_metrics = [metrics]
        elif isinstance(metrics, (list, tuple)):
            input_metrics = list(metrics)
        else:
            raise TypeError("metrics must be a list of metric names or a metric string!")

        if len(input_metrics) == 0:
            raise ValueError("At least one metric must be specified!")

        for metric in input_metrics:
            if metric not in valid_metrics:
                raise ValueError("Only 'loss', 'accuracy', 'precision', 'mae', 'rmse', or 'r2' are supported!!")
            train_key, test_key = f"train_{metric}", f"test_{metric}"
            if train_key not in metrics_obj.__dict__ or test_key not in metrics_obj.__dict__:
                hint = f" Pass task=... (one of {model.task_names}) for per-task metrics." if getattr(model, "task", None) == "MultiTask" and resolved_task is None else ""
                raise ValueError(f"Input metric '{metric}' not found on model.metrics!{hint}")
            if len(getattr(metrics_obj, train_key)) == 0 or len(getattr(metrics_obj, test_key)) == 0:
                raise ValueError(f"No recorded history for metric '{metric}' on this {model.task} model!")

        if title is None:
            title = f"Model Training History — {resolved_task}" if resolved_task is not None else "Model Training History"

        metric_labels = {
            "loss": "Loss",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
        }

        # plot
        if len(input_metrics) > 1:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            flat_axes = axes.flatten()
            fig.suptitle(title)
            for x in range(len(input_metrics)):
                if len(flat_axes) < len(input_metrics):
                    raise ValueError("Number of metrics does not match ncols!")

                train_metric = getattr(metrics_obj, f"train_{input_metrics[x]}")
                test_metric = getattr(metrics_obj, f"test_{input_metrics[x]}")

                if clip:
                    train_metric = np.clip(train_metric, 0, clip)
                    test_metric = np.clip(test_metric, 0, clip)

                label = metric_labels[input_metrics[x]]
                current_ax = flat_axes[x]
                current_ax.plot(train_metric, color=line_color[0])
                current_ax.plot(test_metric, color=line_color[1])
                current_ax.legend(["Train", "Test"], loc=legend_loc)
                current_ax.set_title(f"Model {label}")
                current_ax.set_ylabel(label)
                current_ax.set_xlabel("Epoch")
                current_ax.grid(False)

            total_plots = nrows * ncols
            for x in range(len(input_metrics), total_plots):
                flat_axes[x].axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title)
            train_metric = getattr(metrics_obj, f"train_{input_metrics[0]}")
            test_metric = getattr(metrics_obj, f"test_{input_metrics[0]}")

            if clip:
                train_metric = np.clip(train_metric, 0, clip)
                test_metric = np.clip(test_metric, 0, clip)

            label = metric_labels[input_metrics[0]]
            ax.plot(train_metric, color=line_color[0])
            ax.plot(test_metric, color=line_color[1])
            ax.legend(["Train", "Test"], loc=legend_loc)
            ax.set_title(f"Model {label}")
            ax.set_ylabel(label)
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
                   savepath: str = None, task: Optional[str] = None):
        """
        Generate a linear regression plot using the MLX model and the input data_loader.
        """

        if model is None:
            model = self.model
        if data is None:
            data = self.data if self.data is not None else (self.x_data, self.y_data)
        if add_feats is None:
            add_feats = self.add_feats

        resolved_task = self._resolve_task(model, task, required=True, expected_type="regression")

        # make predictions and extract results
        labels, preds = self._regression_preds(model, data, add_feats, batch_size, task=resolved_task)

        # add metrics computed directly on the plotted (labels, preds)
        if metric_output is not None:
            parts = []
            if 'R2' in metric_output:
                r2 = round(float(r2_score(labels, preds)), 3)
                parts.append(f"R²: {r2}")
            if 'MAE' in metric_output:
                mae = round(float(mean_absolute_error(labels, preds)), 3)
                parts.append(f"MAE: {mae}")
            if 'RMSE' in metric_output:
                rmse = round(float(root_mean_squared_error(labels, preds)), 3)
                parts.append(f"RMSE: {rmse}")
            text = "\n".join(parts) if parts else None
        else:
            text = None

        # plot
        fig = plt.figure(figsize=figsize)
        sns.scatterplot(
            x=labels,
            y=preds,
            alpha=dot_alpha,
            color=dot_color,
        )
        sns.regplot(x=labels, y=preds, scatter=False,
                    line_kws={'color': line_color}, label='Regression')

        unity_y = np.linspace(min(labels), max(labels), 20)
        unity_x = unity_y
        if unity:
            plt.plot(unity_x,
                     unity_y,
                     linestyle='--',
                     linewidth=2,
                     color='0.7',
                     label='Line of Unity')

        if text is None:
            plt.plot([], [], linestyle='None', marker='None')
        else:
            plt.plot([], [], linestyle='None', marker='None', label=text)

        if xlabel is None:
            xlabel = "Actual"
        if ylabel is None:
            ylabel = "Prediction"
        if title is None:
            title = f"Prediction vs. Actual — {resolved_task}" if resolved_task is not None else "Prediction vs. Actual"

        plt.grid(False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=legend_loc, frameon=False)
        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def residuals(self, model: Optional[mlx.nn] = None, data: Optional[Union[Dataloader, list, tuple]] = None,
                  train_data: Optional[Union[Dataloader, list, tuple]] = None,
                  add_feats: Optional[Union[np.array, list, int]] = None, batch_size: int = 1024,
                  histogram: bool = True, title: str = None, color: tuple = (None, None),
                  legend_loc: str = "best", figsize: tuple[float, float] = (8, 5), fontsize: int = 12,
                  titlesize: int = 15, grid: bool = False, savepath: str = None, task: Optional[str] = None):
        """
        Create a residuals plot (actual - predicted) against the predicted values for a regression model.
        """
        if model is None:
            model = self.model
        if data is None:
            data = self.data if self.data is not None else (self.x_data, self.y_data)
        if add_feats is None:
            add_feats = self.add_feats

        resolved_task = self._resolve_task(model, task, required=True, expected_type="regression")

        datasets = [("Test", data, color[1])]
        if train_data is not None:
            datasets.insert(0, ("Train", train_data, color[0]))

        if histogram:
            fig, (ax, hist_ax) = plt.subplots(1, 2, sharey=True, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]})
        else:
            fig, ax = plt.subplots(figsize=figsize)

        for name, current_data, current_color in datasets:
            y_true, y_pred = self._regression_preds(model, current_data, add_feats, batch_size, task=resolved_task)
            residuals = y_true - y_pred
            r2 = r2_score(y_true, y_pred)

            points = ax.scatter(y_pred, residuals, color=current_color, alpha=0.5, label=f"{name} $R^2$ = {r2:.3f}")
            if histogram:
                hist_ax.hist(residuals, bins=50, orientation="horizontal", color=points.get_facecolor()[0])

        ax.axhline(0, color='gainsboro', linestyle='--')
        if histogram:
            hist_ax.axhline(0, color='gainsboro', linestyle='--')
            hist_ax.set_xlabel("Count", fontsize=fontsize)
            hist_ax.grid(grid)

        if title is None:
            title = f"Residual Plot — {resolved_task}" if resolved_task is not None else "Residual Plot"
        ax.grid(grid)
        ax.set_xlabel("Predicted", fontsize=fontsize)
        ax.set_ylabel("Residual (Actual - Predicted)", fontsize=fontsize)
        ax.set_title(title, fontsize=titlesize)
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        if savepath: plt.savefig(savepath, dpi=300)
        plt.show()

    def _regression_preds(self, model, data, add_feats, batch_size, task: Optional[str] = None):
        """
        Get the flattened values (actual, predicted) for a regression model (or regression task of MultiTaskGNN).
        Data is a Dataloader or list of graphs. For tabular MLP models, data is a tuple of (features, labels).
        """
        if getattr(model, "task", None) == "MultiTask":
            loader = self._unshuffled_graph_loader(data, batch_size=batch_size)
            task_idx = model.task_names.index(task)
            y_col, obs = self._extract_multitask_targets(loader, task_idx)
            preds_dict = model.predict(loader, batch_size=batch_size, add_feats=add_feats, return_type="dict")
            return y_col[obs], np.asarray(preds_dict[task])[obs]

        if isinstance(data, tuple):  # tabular MLP data
            x, y = data
            preds = model.predict(x, batch_size=batch_size, binary=None, return_type="list")
            return np.ravel(y), np.ravel(preds)

        # rebuild the loader without shuffling, otherwise the labels will not line up with the predictions
        if isinstance(data, Dataloader):
            data, batch_size = data.dataset, data.batch_size
        data = Dataloader(data, batch_size=batch_size, shuffle=False)
        preds = model.predict(data, batch_size=batch_size, binary=None, add_feats=add_feats, return_type="list")
        labels = [y for batch in data for y in batch.y.flatten().tolist()]
        return np.asarray(labels), np.ravel(preds)

    def confusion_matrix(self, model: Optional[mlx.nn] = None, data: Optional[Dataloader] = None,
                         batch_size: int = None, add_feats: Optional[Union[np.array, list, int]] = None,
                         class_names: Optional[list[str]] = None, palette: str = "Blues", figsize: tuple = (8, 6),
                         title: str = None, cutoff: float = 0.5, savepath: str = None, task: Optional[str] = None):
        """
        Generate a confusion matrix for classification MLX models (or a classification task of MultiTaskGNN).
        """

        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if add_feats is None:
            add_feats = self.add_feats

        resolved_task = self._resolve_task(model, task, required=True, expected_type="classification")

        if resolved_task is not None:
            loader = self._unshuffled_graph_loader(data, batch_size=batch_size)
            task_idx = model.task_names.index(resolved_task)
            y_col, obs = self._extract_multitask_targets(loader, task_idx)
            num_classes = model.task_configs[resolved_task]["num_classes"]
            bs = batch_size if batch_size is not None else 1024
            if num_classes == 2:
                probs = model.predict(loader, batch_size=bs, add_feats=add_feats, binary=False, cutoff=cutoff,
                                      return_type="dict")[resolved_task]
                y_pred = (np.asarray(probs) >= cutoff).astype(int)[obs]
            else:
                y_pred = model.predict(loader, batch_size=bs, add_feats=add_feats, binary="multi",
                                       return_type="dict")[resolved_task][obs]
            y_true = y_col[obs].astype(int)
        else:
            y_true, y_pred = self._get_prediction_labels(model, data, cutoff, batch_size, add_feats)

        # calculate matrix
        cm_matrix = confusion_matrix(y_true, y_pred)

        if cm_matrix.shape == (2, 2):
            group_names = ['TN', 'FP', 'FN', 'TP']
            group_counts = [value for value in cm_matrix.flatten()]
            group_percentages = [f"{round(value * 100, 3)}%" for value in cm_matrix.flatten() / np.sum(cm_matrix)]
            labels = np.asarray(
                [f"{v1}\n\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            ).reshape(2, 2)
        else:
            total = max(int(np.sum(cm_matrix)), 1)
            labels = np.asarray(
                [f"{v}\n{round(v * 100 / total, 2)}%" for v in cm_matrix.flatten()]
            ).reshape(cm_matrix.shape)

        if title is None:
            title = f"Confusion Matrix — {resolved_task}" if resolved_task is not None else "Confusion Matrix on Test Dataset"
        if class_names is None:
            class_names = [str(i) for i in range(cm_matrix.shape[0])]

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
                              title: str = None, cutoff: float = 0.5, savepath: str = None,
                              task: Optional[str] = None):
        """
        Generate a classification report for classification MLX models (or a classification task of MultiTaskGNN).
        """
        if model is None:
            model = self.model
        if not model.is_fitted:
            raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if add_feats is None:
            add_feats = self.add_feats

        resolved_task = self._resolve_task(model, task, required=True, expected_type="classification")

        if resolved_task is not None:
            loader = self._unshuffled_graph_loader(data, batch_size=batch_size)
            task_idx = model.task_names.index(resolved_task)
            y_col, obs = self._extract_multitask_targets(loader, task_idx)
            num_classes = model.task_configs[resolved_task]["num_classes"]
            bs = batch_size if batch_size is not None else 1024
            if num_classes == 2:
                probs = model.predict(loader, batch_size=bs, add_feats=add_feats, binary=False, cutoff=cutoff,
                                      return_type="dict")[resolved_task]
                y_pred = (np.asarray(probs) >= cutoff).astype(int)[obs]
            else:
                y_pred = model.predict(loader, batch_size=bs, add_feats=add_feats, binary="multi",
                                       return_type="dict")[resolved_task][obs]
            y_true = y_col[obs].astype(int)
            if class_names is None:
                class_names = [str(i) for i in range(num_classes)]
        else:
            if class_names is None:
                class_names = ['0', '1']
            y_true, y_pred = self._get_prediction_labels(model, data, cutoff, batch_size, add_feats)

        # calculate classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        keys_to_remove = {'accuracy', 'macro avg', 'weighted avg'}
        filtered_dict = {k: v for k, v in report_dict.items() if k not in keys_to_remove}
        df = pd.DataFrame(filtered_dict).T[['precision', 'recall', 'f1-score']]

        if title is None:
            title = f"Classification Report — {resolved_task}" if resolved_task is not None else "Classification Report"

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

    def _get_prediction_labels(self, model, data, cutoff, batch_size, add_feats):
        """
        Support function for confusion matrix and classification report.
        """
        all_labels = []
        if isinstance(data, list):
            data = Dataloader(data, batch_size=batch_size if batch_size is not None else 1024, shuffle=False)

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

        # get prediction probabilities once and apply cutoff once
        if data is not None:
            probs = model.predict(data=data, batch_size=batch_size, add_feats=add_feats, binary=False,
                                  cutoff=cutoff, return_type="list")
        elif self.x_data is not None and self.y_data is not None:
            all_labels = self.y_data
            probs = model.predict(data=self.x_data, batch_size=batch_size, binary=False, cutoff=cutoff,
                                  return_type="list")
        else:
            raise ValueError("No data provided???")

        # convert to np.array and threshold once
        y_true = np.asarray(all_labels).squeeze()
        if isinstance(cutoff, (float, int, np.floating, np.integer)):
            y_pred = (np.asarray(probs) >= cutoff).astype(int)
        else:
            y_pred = np.asarray(probs)

        return y_true, y_pred

    def roc_curve(self, model: Optional[Union[mlx.nn, list[mlx.nn]]] = None, data: Optional[Dataloader] = None,
                  add_feats: Optional[Union[np.array, list, int]] = None, model_names: Optional[list] = None,
                  color: Union[str, list] = "orange", figsize: tuple = (8, 6), title: str = None,
                  binary: Union[bool, float, str] = False, savepath: str = None, task: Optional[str] = None):
        """
        Generate a ROC curve for MLX classification models (or a binary classification task of MultiTaskGNN).
        """
        if model is None:
            model = self.model
        if not isinstance(model, list):
            if not model.is_fitted:
                raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if add_feats is None:
            add_feats = self.add_feats

        first_model = model[0] if isinstance(model, list) else model
        resolved_task = self._resolve_task(first_model, task, required=True, expected_type="classification")
        if resolved_task is not None and first_model.task_configs[resolved_task]["num_classes"] != 2:
            raise ValueError(f"ROC curve requires a binary classification task (task {resolved_task!r} has "
                             f"{first_model.task_configs[resolved_task]['num_classes']} classes).")

        if title is None:
            title = (f"Receiver Operating Characteristic (ROC) — {resolved_task}"
                     if resolved_task is not None else "Receiver Operating Characteristic (ROC)")

        obs_mask = None
        if resolved_task is not None:
            data = self._unshuffled_graph_loader(data)
            task_idx = first_model.task_names.index(resolved_task)
            y_col, obs_mask = self._extract_multitask_targets(data, task_idx)
            all_labels = y_col[obs_mask].astype(int)
        else:
            all_labels = []
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
                all_labels = np.concatenate(all_labels).ravel()
            elif self.y_data is not None:
                all_labels.append(self.y_data)
                all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                if resolved_task is not None:
                    predictions = np.asarray(
                        x.predict(data=data, binary=binary, return_type="dict", add_feats=add_feats)[resolved_task]
                    )[obs_mask]
                else:
                    predictions = x.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
                fpr, tpr, thresholds = roc_curve(all_labels, predictions)
                roc_auc = roc_auc_score(all_labels, predictions)
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} {roc_auc:.3f}")
            self.best_threshold = "Model given as a list. No 'best threshold' calculated!"
        elif data is not None:
            if resolved_task is not None:
                predictions = np.asarray(
                    model.predict(data=data, binary=binary, return_type="dict", add_feats=add_feats)[resolved_task]
                )[obs_mask]
            else:
                predictions = model.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
            fpr, tpr, thresholds = roc_curve(all_labels, predictions)
            roc_auc = roc_auc_score(all_labels, predictions)

            j = tpr - fpr
            x = np.argmax(j)
            best_threshold = thresholds[x]
            b_threshold = f"Best threshold: {best_threshold:.3f}"

            self.thresholds = best_threshold
            self.best_threshold = b_threshold

            plt.plot(fpr, tpr, color=color, lw=2, label=f"ROC curve {roc_auc:.3f}")
        else:
            predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
            fpr, tpr, thresholds = roc_curve(all_labels, predictions)
            roc_auc = roc_auc_score(all_labels, predictions)

            j = tpr - fpr
            x = np.argmax(j)
            best_threshold = thresholds[x]
            b_threshold = f"Best threshold: {best_threshold:.3f}"

            self.thresholds = best_threshold
            self.best_threshold = b_threshold

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

    def pr_curve(self, model: Optional[Union[mlx.nn, list[mlx.nn]]] = None, data: Optional[Dataloader] = None,
                 add_feats: Optional[Union[np.array, list, int]] = None, model_names: Optional[list] = None,
                 chance_level: bool = False, color: Union[str, list] = "orange", figsize: tuple = (8, 6),
                 title: str = None, binary: Union[bool, float, str] = False, savepath: str = None,
                 task: Optional[str] = None):
        """
        Generate a Precision-Recall curve for MLX classification models (or a binary classification task of MultiTaskGNN).
        """
        if model is None:
            model = self.model
        if not isinstance(model, list):
            if not model.is_fitted:
                raise ValueError("MLX model is not fitted!!!")
        if data is None:
            data = self.data
        if add_feats is None:
            add_feats = self.add_feats

        first_model = model[0] if isinstance(model, list) else model
        resolved_task = self._resolve_task(first_model, task, required=True, expected_type="classification")
        if resolved_task is not None and first_model.task_configs[resolved_task]["num_classes"] != 2:
            raise ValueError(f"PR curve requires a binary classification task (task {resolved_task!r} has "
                             f"{first_model.task_configs[resolved_task]['num_classes']} classes).")

        if title is None:
            title = f"Precision-Recall Curve — {resolved_task}" if resolved_task is not None else "Precision-Recall Curve"

        obs_mask = None
        if resolved_task is not None:
            data = self._unshuffled_graph_loader(data)
            task_idx = first_model.task_names.index(resolved_task)
            y_col, obs_mask = self._extract_multitask_targets(data, task_idx)
            all_labels = y_col[obs_mask].astype(int)
        else:
            all_labels = []
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
                all_labels = np.concatenate(all_labels).ravel()
            elif self.y_data is not None:
                all_labels.append(self.y_data)
                all_labels = np.array(all_labels).squeeze()

        # plot
        fig = plt.figure(figsize=figsize)

        if isinstance(model, list):
            for x, name, c in zip(model, model_names, color):
                if resolved_task is not None:
                    predictions = np.asarray(
                        x.predict(data=data, binary=binary, return_type="dict", add_feats=add_feats)[resolved_task]
                    )[obs_mask]
                else:
                    predictions = x.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
                prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
                avg_prec = average_precision_score(all_labels, predictions)
                plt.plot(recall, prec, color=c, lw=2, label=f"{name} {avg_prec:.3f}")
        elif data is not None:
            if resolved_task is not None:
                predictions = np.asarray(
                    model.predict(data=data, binary=binary, return_type="dict", add_feats=add_feats)[resolved_task]
                )[obs_mask]
            else:
                predictions = model.predict(data=data, binary=binary, return_type="list", add_feats=add_feats)
            prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
            avg_prec = average_precision_score(all_labels, predictions)

            plt.plot(recall, prec, color=color, lw=2, label=f"Precision-Recall Avg: {avg_prec:.3f}")
        else:
            predictions = model.predict(data=self.x_data, binary=binary, return_type="list")
            prec, recall, pr_threshold = precision_recall_curve(all_labels, predictions)
            avg_prec = average_precision_score(all_labels, predictions)

            plt.plot(recall, prec, color=color, lw=2, label=f"Precision-Recall Avg: {avg_prec:.3f}")

        if chance_level:
            chance = all_labels.mean()
            plt.axhline(y=chance, color="gray", linestyle=(0, (5, 10)), lw=1, label=f"Chance Level: {chance:.3f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="best")
        plt.xlim([0, 1.05])
        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
