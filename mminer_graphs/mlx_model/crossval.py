"""Scripts to aid cross validation"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import mlx.nn
import mlx.core as mx
from mlx_graphs.loaders import Dataloader
from mminer_graphs.mlx_models import MolToGraph, EarlyStopping
from mminer_graphs.mlx_models.mlx_utils import set_seed
from typing import Optional, Union


class CrossVal:
    def __init__(self, model: mlx.nn.Module, dataset: pd.DataFrame, estop: Optional[Union[EarlyStopping, int]] = None):
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

    def get_cv_scores(self, graph_type: str = "AtomGraph", dataset: Optional[pd.DataFrame] = None, labels: str = "CV",
                      cv: int = 1, smi_col=None, activity_col=None, add_feats: Optional[np.array] = None,
                      batch_size: int = 32, epochs: int = 10, optimizer: str = "adam", lr: float = 0.01,
                      wt_decay: float = 0.0, device: str = "auto", drop_last: bool = False, verbose: bool = False,
                      device_warn: bool = True, pbar: bool = True, seed: int = 42):
        """
        Loop a given nn.Module model to get the average cv scores. Dataset should be given as a pd.DataFrame with a
        format found in mol-miner. The output will be a dictionary containing mole metrics as a list and an average over
        the cv number.
        :param model: Optional[mlx.nn.Module]
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
        model = self.model

        set_seed(42)

        cv_index = self.get_cv_split(dataset, labels, cv)
        data = dataset[[smi_col, activity_col]]

        loss_list = []
        acc_list = []
        prec_list = []
        mcc_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []

        # extract by index
        for x in tqdm(range(len(cv_index)), desc="CV Round", disable=not pbar):
            train_df = data.loc[cv_index[x][0]]
            test_df = data.loc[cv_index[x][1]]

            # set model
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
            train_loader = Dataloader(train_graphs, batch_size=batch_size, shuffle=True)
            test_loader = Dataloader(test_graphs, batch_size=batch_size)

            # # for troubleshooting
            # print(train_loader.dataset[0])

            # confirm model fit during cv cycles
            if x == 0:
                model.store_initial_params()
                # print(model.parameters())
            elif x > 0:
                model.reset_params()
                # print(model.parameters())

            model.fit(train_loader, test_loader, epochs=epochs, optimizer=optimizer, device=device,
                      estop=estop, lr=lr, wt_decay=wt_decay, pbar=False, verbose=verbose,
                      device_warn=device_warn)

            # scores
            loss_list.append(round(float(np.mean(model.metrics.test_loss)), 3))
            acc_list.append(round(float(np.mean(model.metrics.test_accuracy)), 3))
            prec_list.append(round(float(np.mean(model.metrics.test_precision)), 3))
            mcc_list.append(round(float(np.mean(model.metrics.mcc)), 3))
            recall_list.append(round(float(np.mean(model.metrics.recall)), 3))
            f1_list.append(round(float(np.mean(model.metrics.f1_score)), 3))
            roc_auc_list.append(round(float(np.mean(model.metrics.roc_auc)), 3))

        # convert metrics into a pd.DataFrame
        metrics = {'Accuracy': acc_list, 'Precision': prec_list, 'Recall': recall_list, 'F1': f1_list, 'MCC': mcc_list,
                   'ROC_AUC': roc_auc_list, 'Accuracy_ave': round(np.mean(acc_list), 3),
                   'Precision_ave': round(np.mean(prec_list), 3), 'Recall_avg': round(np.mean(recall_list), 3),
                   'F1 Score_avg': round(np.mean(f1_list), 3), 'MCC_avg': round(np.mean(mcc_list), 3),
                   'ROC_AUC_avg': round(np.mean(roc_auc_list), 3), 'Loss': loss_list,
                   'Loss_ave': round(np.mean(loss_list), 3), }
        metrics_df = pd.DataFrame([metrics])

        return metrics_df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
