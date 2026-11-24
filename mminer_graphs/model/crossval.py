"""Scripts to aid cross validation"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from torch import nn
from torch_geometric.loader import DataLoader
from mminer_graphs.model import MolToGraph, EarlyStopping
from mminer_graphs.model.graph_modeling import Metrics
from typing import Optional, Union
from mminer_graphs.model.model_utils import set_seed, seed_worker, make_generator


class CrossVal:
    def __init__(self, model: nn.Module, dataset: pd.DataFrame, estop: Optional[Union[EarlyStopping, int]] = None):
        self.model = model
        self.dataset = dataset
        self.summary = None
        if estop is None:
            estop = 100
            warnings.warn('estop not specified! Will default to 100 epochs!')
            self.early_stopping = EarlyStopping(monitor='val_loss', patience=estop, mode='min')
        elif isinstance(estop, int):
            self.early_stopping = EarlyStopping(monitor='val_loss', patience=estop, mode='min')
        elif isinstance(estop, EarlyStopping):
            self.early_stopping = estop

    def get_cv_split(self, dataset: Optional[pd.DataFrame] = None, labels: str = "CV", cv: int = 5):
        """
        Obtain index for train/test split. Index is based on input CV label headers from DataFrame. This will output a
        tuple with the indices of the train and test split based on CV label headers.
        :param dataset: pd.DataFrame
            The pd.DataFrame containing the labels and cross-validation information. train_indexes and test_indexes will
            be obtained from the input label pd.DataFrame. Format of pd.DataFrame can be found using mol-miner.
        :param labels: str
            The dataset headers containing the train/test labels.
        :param cv: int
            The number of cross-validation groups.
        :return:
        """
        if dataset is None:
            data = self.dataset
        else:
            data = dataset

        splits = []

        for x in range(cv):
            header = f"{labels}{x}"

            train_indexes = np.flatnonzero(data[header].to_numpy() == "train")
            test_indexes = np.flatnonzero(data[header].to_numpy() == "test")
            splits.append((train_indexes, test_indexes))

        return splits

    def get_cv_scores(self, model: Optional[nn.Module] = None, graph_type: str = "AtomGraph",
                      dataset: Optional[pd.DataFrame] = None, labels: str = "CV", cv: int = 1, smi_col=None,
                      activity_col: Optional[Union[str, list[str], tuple[str, ...]]] = None,
                      add_feats: Optional[np.ndarray] = None, batch_size: int = 32, epochs: int = 10,
                      optimizer: str = "adam", lr: float = 0.01, wt_decay: float = 0.0,
                      criterion: str = "cross_entropy", device: str = "auto", drop_last: bool = False,
                      verbose: bool = False, device_warn: bool = True, pbar: bool = True, seed: int = 42):
        """
        Loop a given nn.Module model to get the average cv scores. Dataset should be given as a pd.DataFrame with a
        format found in mol-miner. Supports both single-task models (`activity_col: str`) and `MultiTaskGNN`
        (`activity_col: list[str]`, defaulting to `model.task_names`).
        :param model: Optional[nn.Module]
            Input torch model.
        :param graph_type: str
            The molecular graph type to generate from smiles string. Can be 'AtomGraph', 'PharmacophoreGraph'
            'FunctionalGraph', 'JunctionTreeGraph', or 'ClusterGraph'.
        :param dataset: Optional[pd.DataFrame]
            The pd.DataFrame containing the labels and cross-validation information. train_indexes and test_indexes will
            be obtained from the input label pd.DataFrame. Format of pd.DataFrame can be found using mol-miner.
        :param labels: str
            Indicate teh prefix of the columns containing the train/test labels.
        :param cv: int
            The number of cross-validation folds.
        :param smi_col: str
            Column containing the smiles string.
        :param activity_col: Optional[Union[str, list[str], tuple[str, ...]]]
            Column (single-task) or list of columns (MultiTaskGNN) containing the target labels. For MultiTaskGNN,
            defaults to `model.task_names` when None.
        :param add_feats: Optional[Union[np.array, list]]
            Input additional information that will be appended to the latent space after the graph layers before being
            fit to the mlp layers. The name of the tensor will be 'add_feats'.
        :param batch_size: int
            The number of samples to load for each batch.
        :param epochs: int
            Set the number of epochs for model training.
        :param optimizer: str
            Set the model optimizer.
        :param lr: float
            Set the optimizer learning rate.
        :param wt_decay: float
            Reduce the calculation of weights to the loss function.
        :param criterion: str
            Set the loss criterion (single-task models; MultiTaskGNN uses per-task criteria).
        :param device: str
            Set the device for the tensors. Can be "auto", "cuda", "mps", or "cpu".
        :param drop_last: bool
            Drop last batch if using a normalizer requiring mini-batches greater than 1. Only used on train_loader.
        :param verbose: bool
            Whether to output the train/test accuracy and loss for each epoch.
        :param device_warn:
            Option to print which tensor/model is on 'cpu', 'cuda' or 'mps'.
        :param pbar: bool
            Whether ot output a progress bar for each cross validation fold.
        :param seed: int
            Set the seed for the model.
        :return: pd.DataFrame
            For single-task models: one row per CV fold with `self.summary` storing mean/std.
            For MultiTaskGNN: a MultiIndex `(fold, Task)` DataFrame with `self.summary` storing per-task mean/std
            across folds and `self.macro_summary` storing macro-averaged metrics across tasks.
        """
        if dataset is None:
            dataset = self.dataset
        if model is None:
            model = self.model

        set_seed(seed)

        cv_index = self.get_cv_split(dataset, labels, cv)
        is_multitask = getattr(model, "task", None) == "MultiTask"

        if is_multitask:
            if activity_col is None:
                activity_cols = list(model.task_names)
            elif isinstance(activity_col, str):
                activity_cols = [activity_col]
            elif isinstance(activity_col, (list, tuple)):
                activity_cols = list(activity_col)
            else:
                raise TypeError("For MultiTaskGNN, activity_col must be a list/tuple of column names or None.")

            if len(activity_cols) != len(model.task_names):
                raise ValueError(
                    f"Length of activity_col ({len(activity_cols)}) must match model.task_names ({len(model.task_names)})."
                )
            y_iter = dataset[activity_cols].to_numpy(dtype=np.float32)
        else:
            if isinstance(activity_col, (list, tuple)):
                raise ValueError("Multiple activity_col columns require a MultiTaskGNN model.")
            data = dataset[[smi_col, activity_col]]
            y_iter = data[activity_col]

        # hold scores
        if model.task == "Classification":
            loss_list = []
            acc_list = []
            prec_list = []
            mcc_list = []
            recall_list = []
            f1_list = []
            roc_auc_list = []
        elif model.task == "Regression":
            loss_list = []
            r2_list = []
            mae_list = []
            rmse_list = []
        elif is_multitask:
            mt_rows = []
        else:
            raise ValueError("Issue with model.task!")

        # convert list to numpy
        if add_feats is not None:
            if isinstance(add_feats, list):
                add_feats = np.array(add_feats, dtype=np.float32)

        # featurize all molecules once before the fold loop
        smi_series = dataset[smi_col]
        if add_feats is not None:
            all_graphs = [
                MolToGraph(smi=smi, label=y, graph_type=graph_type, add_feats=extra).graph_data
                for smi, y, extra in zip(smi_series, y_iter, add_feats)
            ]
        else:
            all_graphs = [
                MolToGraph(smi=smi, label=y, graph_type=graph_type).graph_data
                for smi, y in zip(smi_series, y_iter)
            ]

        # extract by positional index
        for x in tqdm(range(len(cv_index)), desc="CV Round", disable=not pbar):
            train_indices = cv_index[x][0]
            test_indices = cv_index[x][1]

            # estop is reset inside fit
            estop = self.early_stopping

            # slice pre-featurized graphs for this fold
            train_graphs = [all_graphs[i] for i in train_indices]
            test_graphs = [all_graphs[i] for i in test_indices]

            # load data
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=seed_worker, generator=make_generator(seed), drop_last=drop_last)
            test_loader = DataLoader(test_graphs, batch_size=batch_size, drop_last=False)

            # confirm model fit during cv cycles
            if model.is_fitted:
                model.reset_params()

            if is_multitask:
                model.fit(train_loader, test_loader, epochs=epochs, optimizer=optimizer, device=device, estop=estop,
                          lr=lr, wt_decay=wt_decay, pbar=False, verbose=verbose, device_warn=device_warn)
            else:
                # clear metrics so best epoch is picked from this fold only
                model.metrics = Metrics()
                model.fit(train_loader, test_loader, epochs=epochs, optimizer=optimizer, device=device, estop=estop,
                          lr=lr, wt_decay=wt_decay, pbar=False, verbose=verbose, criterion=criterion,
                          device_warn=device_warn)

            # scores
            best_test_epoch = (
                model.metrics.best_epoch
                if model.metrics.best_epoch is not None
                else int(np.argmin(model.metrics.test_loss))
            )
            if model.task == "Classification":
                loss_list.append(round(float(model.metrics.test_loss[best_test_epoch]), 3))
                acc_list.append(round(float(model.metrics.test_accuracy[best_test_epoch]), 3))
                prec_list.append(round(float(model.metrics.test_precision[best_test_epoch]), 3))
                mcc_list.append(round(float(model.metrics.mcc[best_test_epoch]), 3))
                recall_list.append(round(float(model.metrics.recall[best_test_epoch]), 3))
                f1_list.append(round(float(model.metrics.f1_score[best_test_epoch]), 3))
                roc_auc_list.append(round(float(model.metrics.roc_auc[best_test_epoch]), 3))
            elif model.task == "Regression":
                loss_list.append(round(float(model.metrics.test_loss[best_test_epoch]), 3))
                r2_list.append(round(float(model.metrics.test_r2[best_test_epoch]), 3))
                mae_list.append(round(float(model.metrics.test_mae[best_test_epoch]), 3))
                rmse_list.append(round(float(model.metrics.test_rmse[best_test_epoch]), 3))
            elif is_multitask:
                cfgs = model.task_configs
                for tname in model.task_names:
                    tm = model.metrics.tasks[tname]
                    t_loss = tm.test_loss[best_test_epoch]
                    row = {
                        "fold": x,
                        "Task": tname,
                        "TaskType": "Classification" if cfgs[tname]["type"] == "classification" else "Regression",
                        "Loss": round(float(t_loss), 3) if np.isfinite(t_loss) else np.nan,
                    }
                    if cfgs[tname]["type"] == "classification":
                        for col_name, hist in [
                            ("Accuracy", tm.test_accuracy),
                            ("Precision", tm.test_precision),
                            ("Recall", tm.recall),
                            ("F1", tm.f1_score),
                            ("MCC", tm.mcc),
                            ("ROC_AUC", tm.roc_auc),
                        ]:
                            val = hist[best_test_epoch]
                            row[col_name] = round(float(val), 3) if np.isfinite(val) else np.nan
                    else:
                        for col_name, hist in [
                            ("r2", tm.test_r2),
                            ("MAE", tm.test_mae),
                            ("RMSE", tm.test_rmse),
                        ]:
                            val = hist[best_test_epoch]
                            row[col_name] = round(float(val), 3) if np.isfinite(val) else np.nan
                    mt_rows.append(row)

        # convert metrics into a pd.DataFrame
        if model.task == "Classification":
            metrics_df = pd.DataFrame({'Accuracy': acc_list, 'Precision': prec_list, 'Recall': recall_list,
                                       'F1': f1_list, 'MCC': mcc_list, 'ROC_AUC': roc_auc_list, 'Loss': loss_list})
            metrics_df.index.name = 'fold'
            self.summary = metrics_df.agg(['mean', 'std']).round(3)
            self.macro_summary = None
        elif model.task == "Regression":
            metrics_df = pd.DataFrame({'r2': r2_list, 'MAE': mae_list, 'RMSE': rmse_list, 'Loss': loss_list})
            metrics_df.index.name = 'fold'
            self.summary = metrics_df.agg(['mean', 'std']).round(3)
            self.macro_summary = None
        elif is_multitask:
            metrics_df = pd.DataFrame(mt_rows)
            task_types = {cfg["type"] for cfg in model.task_configs.values()}
            if task_types == {"classification"}:
                ordered = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "Loss"]
                metrics_df = metrics_df.set_index(["fold", "Task"])[ordered]
                numeric_cols = ordered
            elif task_types == {"regression"}:
                ordered = ["r2", "MAE", "RMSE", "Loss"]
                metrics_df = metrics_df.set_index(["fold", "Task"])[ordered]
                numeric_cols = ordered
            else:
                ordered = ["TaskType", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "r2", "MAE", "RMSE", "Loss"]
                for col in ordered:
                    if col not in metrics_df.columns:
                        metrics_df[col] = np.nan
                metrics_df = metrics_df.set_index(["fold", "Task"])[ordered]
                numeric_cols = [c for c in ordered if c != "TaskType"]

            # Per-task mean/std across folds
            self.summary = metrics_df.groupby(level="Task", sort=False)[numeric_cols].agg(["mean", "std"]).round(3)
            # Macro-average across tasks per fold, then mean/std across folds
            per_fold_macro = metrics_df.groupby(level="fold")[numeric_cols].mean()
            self.macro_summary = per_fold_macro.agg(["mean", "std"]).round(3)
        else:
            raise ValueError("Issue with model.task!")

        return metrics_df

if __name__ == "__main__":
    import doctest

    doctest.testmod()
