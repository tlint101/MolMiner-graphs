"""Scripts to aid cross validation"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mminer_graphs.mlx_models import MolToGraph, EarlyStopping
from mminer_graphs.mlx_models.mlx_modeling import Metrics
from mminer_graphs.mlx_models.mlx_utils import set_seed
from typing import Optional, Union


class CrossVal:
    def __init__(self, model: mlx.nn.Module, dataset: pd.DataFrame, estop: Optional[Union[EarlyStopping, int]] = None):
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

            train_indexes = data[data[header] == "train"].index.values.astype(int)
            test_indexes = data[data[header] == "test"].index.values.astype(int)
            splits.append((train_indexes, test_indexes))

        return splits

    def get_cv_scores(self, graph_type: str = "AtomGraph", dataset: Optional[pd.DataFrame] = None, labels: str = "CV",
                      cv: int = 1, smi_col=None, activity_col=None, add_feats: Optional[Union[np.ndarray, list]] = None,
                      batch_size: int = 32, epochs: int = 10, optimizer: str = "adam", lr: float = 0.01,
                      wt_decay: float = 0.0, criterion: str = "cross_entropy", device: str = "auto",
                      drop_last: bool = False, verbose: bool = False, device_warn: bool = True, pbar: bool = True,
                      seed: int = 42):
        """
        Loop a given nn.Module model to get the average cv scores. Dataset should be given as a pd.DataFrame with a
        format found in mol-miner. The output will be a dictionary containing mole metrics as a list and an average over
        the cv number.
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
        :param activity_col: str
            Column containing the activity labels.
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
            Set the loss criterion.
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
            A pd.DataFrame with one row per cv fold and one column per metric. The mean and std of each metric are
            stored separately in self.summary.
        """
        global loss_list, acc_list, prec_list, mcc_list, recall_list, f1_list, roc_auc_list, r2_list, mae_list, rmse_list, train_graphs, test_graphs
        if dataset is None:
            dataset = self.dataset
        model = self.model

        set_seed(seed)

        cv_index = self.get_cv_split(dataset, labels, cv)
        data = dataset[[smi_col, activity_col]]

        # hold scores
        if model.type == "Classification":
            loss_list = []
            acc_list = []
            prec_list = []
            mcc_list = []
            recall_list = []
            f1_list = []
            roc_auc_list = []
        elif model.type == "Regression":
            loss_list = []
            r2_list = []
            mae_list = []
            rmse_list = []

        # convert list to numpy
        if add_feats is not None:
            if isinstance(add_feats, list):
                add_feats = np.array(add_feats, dtype=np.float32)

        # extract by index
        for x in tqdm(range(len(cv_index)), desc="CV Round", disable=not pbar):
            train_indices = cv_index[x][0]
            test_indices = cv_index[x][1]
            train_df = data.loc[train_indices]
            test_df = data.loc[test_indices]

            # slice add_feats
            if add_feats is not None:
                train_add_feats = add_feats[train_indices]
                test_add_feats = add_feats[test_indices]
                # # debugging
                # print(f"length of train_df: {len(train_df)}")
                # print(f"length of test_df: {len(test_df)}")
                # print(f"length of additional features: {len(add_feats)}")
                # print(f"length of train_add_feats: {len(train_add_feats)}")
                # print(f"length of test_add_feats: {len(test_add_feats)}")

                # create graphs with add_feats
                train_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type, add_feats=extra).graph_data for
                                smi, y, extra in zip(train_df[smi_col], train_df[activity_col], train_add_feats)]
                test_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type, add_feats=extra).graph_data for
                               smi, y, extra in zip(test_df[smi_col], test_df[activity_col], test_add_feats)]
            else:
                train_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type).graph_data for smi, y, in
                                zip(train_df[smi_col], train_df[activity_col])]
                test_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type).graph_data for smi, y in
                               zip(test_df[smi_col], test_df[activity_col])]

            # todo move this outside of loop and create chunks to be passed?
            # load data
            train_loader = Dataloader(train_graphs, batch_size=batch_size, shuffle=True)
            test_loader = Dataloader(test_graphs, batch_size=batch_size)

            # # for troubleshooting
            # print(train_loader.dataset[0])

            # set model
            estop = self.early_stopping

            # confirm model fit during cv cycles
            if x == 0:
                model.store_initial_params()
                # print(model.parameters())
            elif x > 0:
                model.reset_params()
                # print(model.parameters())

            # # for debugging
            # from mminer_graphs.mlx_models import summary
            # print(summary(model))

            # clear metrics so best epoch is picked from this fold only
            model.metrics = Metrics()

            model.fit(train_loader, test_loader, epochs=epochs, optimizer=optimizer, device=device, estop=estop, lr=lr,
                      wt_decay=wt_decay, pbar=False, verbose=verbose, criterion=criterion, device_warn=device_warn)

            # scores
            best_test_epoch = np.argmin(model.metrics.test_loss)
            if model.type == "Classification":
                loss_list.append(float(model.metrics.test_loss[best_test_epoch]))
                acc_list.append(float(model.metrics.test_accuracy[best_test_epoch]))
                prec_list.append(float(model.metrics.test_precision[best_test_epoch]))
                mcc_list.append(float(model.metrics.mcc[best_test_epoch]))
                recall_list.append(float(model.metrics.recall[best_test_epoch]))
                f1_list.append(float(model.metrics.f1_score[best_test_epoch]))
                roc_auc_list.append(float(model.metrics.roc_auc[best_test_epoch]))
            elif model.type == "Regression":
                loss_list.append(float(model.metrics.test_loss[best_test_epoch]))
                r2_list.append(float(model.metrics.test_r2[best_test_epoch]))
                mae_list.append(float(model.metrics.test_mae[best_test_epoch]))
                rmse_list.append(float(model.metrics.test_rmse[best_test_epoch]))

        # convert metrics into a pd.DataFrame
        if model.type == "Classification":
            metrics_df = pd.DataFrame({'Accuracy': acc_list, 'Precision': prec_list, 'Recall': recall_list,
                                       'F1': f1_list, 'MCC': mcc_list, 'ROC_AUC': roc_auc_list, 'Loss': loss_list})
        elif model.type == "Regression":
            metrics_df = pd.DataFrame({'r2': r2_list, 'MAE': mae_list, 'RMSE': rmse_list, 'Loss': loss_list})
        else:
            raise ValueError("Issue with model.type!")
        metrics_df.index.name = 'fold'

        # summary kept separate so metrics_df stays clean for downstream stats
        self.summary = metrics_df.agg(['mean', 'std']).round(3)

        return metrics_df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
