"""
Script super class for hyperparameter optimizations.
"""

import json
import math
import optuna
from optuna.trial import TrialState
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple, Type

"""
The following scripts are for generating plots using optuna optimization
"""


class Optimization:
    def __init__(self, model: Type[nn.Module] = None):
        self.model = model

    """load params from txt"""

    @staticmethod
    def load_params(path: str = None):
        # load a json file containing model parameters
        with open(path, 'r') as f:
            params = json.load(f)

        return params

    """plot methods"""

    @staticmethod
    def plot_trials(space: optuna.study = None, title: str = "Optimization Score per Trial",
                    xlabel: str = 'Number of Trials', ylabel: str = 'Score', color: str = "#1f77b4", linewidth: int = 2,
                    fontsize: int = 14, figsize: Tuple[int, int] = (10, 6), savepath: Optional[str] = None,
                    ):
        """
        Plot a line plot of trials vs optimization score.
        :param space: optuna.study
            The optimization results from opt_model.
        :param title:   str
            Set the title of the figure.
        :param xlabel: str
            Set the title of the x-axis
        :param ylabel: str
            Set the title of the y-axis
        :param color: str
            Color of the line. Can be name of color or a hex code.
        :param linewidth: int
            Set the width of the line.
        :param fontsize: int
            Set the fontsize of the figure.
        :param figsize: Tuple[int, int]
            Set the figure size.
        :param savepath: str
            Set figure save path.
        :return:
        """
        # results from opt_model
        space_df = space.trials_dataframe()

        plt.figure(figsize=figsize)
        sns.lineplot(data=space_df, x='number', y='value', color=color, linewidth=linewidth)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    @staticmethod
    def plot_search_space(study: optuna.study.Study, title: str = None, title_size: int = 20, fontsize: int = 14,
                          height: int = 2, marker_size: int = 25, color: list = None, savepath: Optional[str] = None):
        """
        Plot a graph describing the search space using Optuna results.
        :param study: optuna.study.Study
            The study object containing the optimization results.
        :param title: str, optional
            The title of the plot.
        :param title_size: int, optional
            The font size for the title.
        :param fontsize: int, optional
            The font size for the axis labels and tick labels.
        :param height: int, optional
            The height of the individual plots.
        :param marker_size: int, optional
            The marker size for the scatter plot.
        :param color: list, optional
            Colors for different plot elements (e.g., ['Greens', 'red', 'blue']).
        :param savepath: str, optional
            The path to save the figure.
        """
        # results from opt_model
        trials_df = study.trials_dataframe()

        # process and extract the hyperparameter columns
        df_params = trials_df.filter(regex='params_', axis=1)
        cols = df_params.columns.str.replace('params_', '', regex=True)
        df_params = df_params.rename(columns=dict(zip(df_params.columns, cols)))
        # remove columns n_jobs and random_state associated with sklearn models
        df_params = df_params.drop(columns=[col for col in ['n_jobs', 'random_state'] if col in df_params])

        # convert categorical data to numbers
        for col in df_params.columns:
            if df_params[col].dtype == 'object':
                # mapp category
                df_params[col] = df_params[col].astype('category')

                # print mapping
                mapping = dict(enumerate(df_params[col].cat.categories))
                print(f"Category mapping for {col}")
                for code, label in mapping.items():
                    print(f"  {code} → {label}")
                # Replace column with codes
                df_params[col] = df_params[col].cat.codes

        if title is None:
            title = "Search Space"

        sns.set_style("white")

        if color:
            color_kdeplot = color[0]
            color_dots = color[1]
            color_line = color[2]
        else:
            color_kdeplot = "ch:s=.25,rot=-.25"
            color_dots = 'red'
            color_line = 'red'

        # Generate plots
        img = sns.PairGrid(df_params, diag_sharey=False, height=height)
        img = img.map_upper(sns.scatterplot, s=marker_size, color=color_dots, alpha=0.2)
        img = img.map_lower(
            sns.kdeplot,
            fill=True,
            cmap=sns.color_palette(color_kdeplot, as_cmap=True),
        )
        img = img.map_diag(sns.kdeplot, fill=True, alpha=0.2, color=color_line, linewidth=1.5)

        # adjust labels
        for ax in img.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

            # adjust tick labels for each subplot
            ax.tick_params(axis='both', labelsize=12)

        plt.suptitle(title, y=1.02, fontsize=title_size)

        # Save figure if savepath is provided
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()


class EarlyStopping:
    """Set early stopping criteria. Only successful trials will be counted"""

    def __init__(self, patience: int = 30, n_prune: int = 30) -> None:
        self.patience = patience
        self.prune_threshold = n_prune
        self.n_prune = 0
        self.best_score = None
        self.unimproved_trials = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # handle consecutive pruning:
        if trial.state == TrialState.PRUNED:
            self.n_prune += 1
        else:
            self.n_prune = 0

        # trigger statement
        if self.n_prune >= self.prune_threshold:
            print(f"Early stopping triggered: {self.n_prune} consecutive trials pruned!")
            study.stop()
            return

        # handle no score improvements
        if trial.state != TrialState.COMPLETE or trial.value is None or math.isnan(trial.value):
            return

        # valid score
        if self.best_score is None:
            self.best_score = trial.value
            return

        # check if current trial is better than best value
        if trial.value > self.best_score:
            self.best_score = trial.value
            self.unimproved_trials = 0
        else:
            self.unimproved_trials += 1

        # trigger statement
        if self.unimproved_trials >= self.patience:
            print(f"Early stopping triggered: No improvement after {self.unimproved_trials} trials!")
            study.stop()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
