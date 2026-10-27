"""Scripts to aid cross validation"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from torch import nn
from torch_geometric.loader import DataLoader
from mminer_graphs.model import MolToGraph, EarlyStopping
from typing import Optional, Union
from mminer_graphs.model.model_utils import set_seed


class CrossVal:
    def __init__(self, model: nn.Module, dataset: pd.DataFrame, estop: Optional[Union[EarlyStopping, int]] = None):
        self.model = model
        self.dataset = dataset
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

    def get_cv_scores(self, model: Optional[nn.Module] = None, graph_type: str = "AtomGraph",
                      dataset: Optional[pd.DataFrame] = None, labels: str = "CV", cv: int = 1, smi_col=None,
                      activity_col=None, add_feats: Optional[np.array] = None, batch_size: int = 32, epochs: int = 10,
                      optimizer: str = "adam", lr: float = 0.01, wt_decay: float = 0.0,
                      criterion: str = "cross_entropy", device: str = "auto", drop_last: bool = False,
                      verbose: bool = False, device_warn: bool = True, pbar: bool = True, seed: int = 42):
        """
        Loop a given nn.Module model to get the average cv scores. Dataset should be given as a pd.DataFrame with a
        format found in mol-miner. The output will be a dictionary containing mole metrics as a list and an average over
        the cv number.
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
        :return: dict
            A dictionary containing a list of metrics and average metrics. The dictionary can be fed into a
            pd.DataFrame.
        """
        if dataset is None:
            dataset = self.dataset
        if model is None:
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

        # extract by index
        for x in tqdm(range(len(cv_index)), desc="CV Round", disable=not pbar):
            train_df = data.loc[cv_index[x][0]]
            test_df = data.loc[cv_index[x][1]]

            # set model
            model = self.model
            estop = self.early_stopping
            if x > 0:
                estop.reset()

            # create array of graphs for train/test
            if add_feats is not None:
                train_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type, add_feats=extra).graph_data for
                                smi, y, extra in zip(train_df[smi_col], train_df[activity_col], add_feats)]
                test_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type, add_feats=extra).graph_data for
                               smi, y, extra in zip(test_df[smi_col], test_df[activity_col], add_feats)]
            else:
                train_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type).graph_data for smi, y, in
                                zip(train_df[smi_col], train_df[activity_col])]
                test_graphs = [MolToGraph(smi=smi, label=y, graph_type=graph_type).graph_data for smi, y in
                               zip(test_df[smi_col], test_df[activity_col])]

            # load data
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=set_seed(seed), drop_last=drop_last)
            test_loader = DataLoader(test_graphs, batch_size=batch_size, drop_last=False)

            # # for troubleshooting
            # print(train_loader.dataset[0])

            # confirm model fit during cv cycles
            if model.is_fitted:
                model.reset_params()

            model.fit(train_loader, test_loader, epochs=epochs, optimizer=optimizer, device=device, estop=estop, lr=lr,
                      wt_decay=wt_decay, pbar=False, verbose=verbose, criterion=criterion, device_warn=device_warn)

            # scores
            best_test_epoch = np.argmin(model.metrics.test_loss)
            if model.type == "Classification":
                loss_list.append(round(float(model.metrics.test_loss[best_test_epoch]), 3))
                acc_list.append(round(float(model.metrics.test_accuracy[best_test_epoch]), 3))
                prec_list.append(round(float(model.metrics.test_precision[best_test_epoch]), 3))
                mcc_list.append(round(float(model.metrics.mcc[best_test_epoch]), 3))
                recall_list.append(round(float(model.metrics.recall[best_test_epoch]), 3))
                f1_list.append(round(float(model.metrics.f1_score[best_test_epoch]), 3))
                roc_auc_list.append(round(float(model.metrics.roc_auc[best_test_epoch]), 3))
            elif model.type == "Regression":
                loss_list.append(round(float(model.metrics.test_loss[best_test_epoch]), 3))
                r2_list.append(round(float(model.metrics.test_r2[best_test_epoch]), 3))
                mae_list.append(round(float(model.metrics.test_mae[best_test_epoch]), 3))
                rmse_list.append(round(float(model.metrics.test_rmse[best_test_epoch]), 3))

        # convert metrics into a pd.DataFrame
        if model.type == "Classification":
            metrics = {'Accuracy': acc_list, 'Precision': prec_list, 'Recall': recall_list, 'F1': f1_list,
                       'MCC': mcc_list,
                       'ROC_AUC': roc_auc_list, 'Accuracy_ave': round(np.mean(acc_list), 3),
                       'Precision_ave': round(np.mean(prec_list), 3),
                       'Recall_avg': round(np.mean(recall_list), 3),
                       'F1 Score_avg': round(np.mean(f1_list), 3), 'MCC_avg': round(np.mean(mcc_list), 3),
                       'ROC_AUC_avg': round(np.mean(roc_auc_list), 3), 'Loss': loss_list,
                       'Loss_ave': round(np.mean(loss_list), 3), }
        elif model.type == "Regression":
            metrics = {'r2': r2_list, 'MAE': mae_list, 'RMSE': rmse_list, 'r2_ave': round(np.mean(r2_list), 3),
                       'MAE_ave': round(np.mean(mae_list), 3), 'RMSE': round(np.mean(rmse_list), 3),
                       'Loss': loss_list, 'Loss_ave': round(np.mean(loss_list), 3), }
        else:
            raise ValueError("Issue with model.type!")

        metrics_df = pd.DataFrame([metrics])

        return metrics_df
