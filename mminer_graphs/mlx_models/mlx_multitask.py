import copy
import json
from typing import Optional, Union

import mlx.core as mx
import mlx.nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_graphs.loaders import Dataloader
import numpy as np
import pandas as pd
from tqdm import tqdm

from mminer_graphs.mlx_models.mlx_modeling import EarlyStopping, Metrics, set_seed
from mminer_graphs.mlx_models.mlx_utils import (
    MLX_OPTI,
    NO_WT_DECAY,
    get_device,
    state_apple_silicon_device,
)
from mminer_graphs.model.multitask import (
    CHECKPOINT_FORMAT,
    VALID_LOSS_WEIGHTINGS,
    _build_evaluation_df,
    _compute_task_metrics_np,
    _format_predictions_np,
    _parse_tasks,
    _resolve_add_feat_dim,
    _validate_multitask_checkpoint,
)

META_KEY = "__multitask_meta__"


class MultiTaskMetrics:
    """
    Training history container for MLX MultiTaskGNN.
    Stores aggregate train_loss/test_loss/best_epoch alongside per-task Metrics objects on `self.tasks[task_name]`.
    """

    def __init__(self, task_names: list[str]):
        self.best_epoch: Optional[int] = None
        self.train_loss: list[float] = []
        self.test_loss: list[float] = []
        self.tasks: dict[str, Metrics] = {name: Metrics() for name in task_names}


class MultiTaskGNN(mlx.nn.Module):
    """
    Composable multi-task molecular graph neural network (Apple Silicon MLX / MLX-Graphs).

    Mirrors `mminer_graphs.model.MultiTaskGNN` and wraps any MLX latent-space encoder (`GCN`, `GINE`, `GSAGE`, `GAT`)
    with an optional shared MLP trunk and per-task prediction heads supporting mixed binary classification,
    multi-class classification, and regression tasks with `NaN`-masked sparse labels.
    """

    def __init__(
        self,
        encoder: mlx.nn.Module,
        tasks: Union[list[str], tuple[str, ...], dict],
        default_task: str = "classification",
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        shared_mlp_layers: int = 0,
        head_mlp_layers: int = 1,
        mlp_dropout: float = 0.0,
        loss_weighting: str = "equal",
    ):
        """
        :param encoder: mlx.nn.Module
            Graph encoder from `mminer_graphs.mlx_models.mlx_latent_space` exposing `.out_dim` (`GCN`, `GINE`, `GSAGE`, `GAT`).
        :param tasks: Union[list[str], tuple[str, ...], dict]
            Either a list of task names (homogeneous tasks using `default_task`), or a dict mapping each task name
            to a config dict, e.g. `{"Class": {"type": "classification", "num_classes": 2}, "AlogP": {"type": "regression"}}`.
        :param default_task: str
            Default task type ("classification" or "regression") when `tasks` is a list of names or omits `"type"`.
        :param add_feats: Optional[Union[np.ndarray, list, int]]
            Optional extra molecular descriptors concatenated onto the pooled graph embedding. Pass an int (feature count)
            or array/list of descriptors.
        :param shared_mlp_layers: int
            Number of shared MLP layers applied after graph readout (and `add_feats`) before branching into task heads.
        :param head_mlp_layers: int
            Number of MLP layers inside each task-specific prediction head (must be >= 1; 1 = single linear projection).
        :param mlp_dropout: float
            Dropout probability applied inside shared and head MLP hidden blocks.
        :param loss_weighting: str
            Strategy for combining per-task losses: `"equal"`, `"manual"` (uses per-task `"weight"`), or
            `"uncertainty"` (Kendall homoscedastic uncertainty weighting with learnable log-variances).
        """
        super().__init__()
        if not hasattr(encoder, "out_dim") or not isinstance(encoder.out_dim, (int, np.integer)) or encoder.out_dim <= 0:
            raise ValueError(
                "encoder must expose a positive integer `.out_dim` attribute "
                "(use an encoder from mminer_graphs.mlx_models.mlx_latent_space)."
            )
        if shared_mlp_layers < 0:
            raise ValueError(f"shared_mlp_layers must be >= 0 (got {shared_mlp_layers}).")
        if head_mlp_layers < 1:
            raise ValueError(f"head_mlp_layers must be >= 1 (got {head_mlp_layers}).")
        if loss_weighting not in VALID_LOSS_WEIGHTINGS:
            raise ValueError(
                f"loss_weighting must be one of {VALID_LOSS_WEIGHTINGS} (got {loss_weighting!r})."
            )

        self.encoder = encoder
        # Prefix non-parameter dicts/lists with '_' so mlx.nn.Module.parameters() never traverses metadata
        self._task_names, self._task_configs = _parse_tasks(tasks, default_task=default_task)
        self.task = "MultiTask"
        self.loss_weighting = loss_weighting
        self.mlp_dropout = float(mlp_dropout)
        self._add_feats = add_feats
        self._add_feat_dim = _resolve_add_feat_dim(add_feats)
        self._encoder_uses_edges = encoder._has_param("edge_features") if hasattr(encoder, "_has_param") else False
        self.is_fitted = False
        self._y_means: dict[str, float] = {}
        self._y_stds: dict[str, float] = {}
        self._initial_params = None
        self.metrics = MultiTaskMetrics(self._task_names)

        # Optional shared MLP trunk (instantiate Dropout in __init__ so eval() disables it)
        trunk_dim = int(encoder.out_dim) + self._add_feat_dim
        self.shared_mlp = []
        dim = trunk_dim
        for _ in range(shared_mlp_layers):
            next_dim = max(1, round(dim / 2))
            self.shared_mlp.append(
                mlx.nn.Sequential(
                    mlx.nn.Linear(dim, next_dim),
                    mlx.nn.BatchNorm(next_dim),
                    mlx.nn.ReLU(),
                    mlx.nn.Dropout(p=self.mlp_dropout),
                )
            )
            dim = next_dim

        # Per-task prediction heads stored as list (indexed 0..T-1 so task names may contain dots or hyphens)
        self.heads = []
        for name in self._task_names:
            out_dim = self._task_configs[name]["out_dim"]
            layers = []
            h_dim = dim
            for _ in range(head_mlp_layers - 1):
                next_dim = max(1, round(h_dim / 2))
                layers.extend(
                    [
                        mlx.nn.Linear(h_dim, next_dim),
                        mlx.nn.BatchNorm(next_dim),
                        mlx.nn.ReLU(),
                        mlx.nn.Dropout(p=self.mlp_dropout),
                    ]
                )
                h_dim = next_dim
            layers.append(mlx.nn.Linear(h_dim, out_dim))
            self.heads.append(mlx.nn.Sequential(*layers))

        # Learnable homoscedastic log-variances for Kendall uncertainty weighting
        if self.loss_weighting == "uncertainty":
            self.log_vars = mx.zeros((len(self._task_names),), dtype=mx.float32)

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)

    @property
    def task_configs(self) -> dict[str, dict]:
        return copy.deepcopy(self._task_configs)

    @property
    def add_feats(self):
        return self._add_feats

    @property
    def y_means(self) -> dict[str, float]:
        return dict(self._y_means)

    @property
    def y_stds(self) -> dict[str, float]:
        return dict(self._y_stds)

    def get_head(self, task_name: str) -> mlx.nn.Module:
        """Return the prediction head module for a given task name."""
        if task_name not in self._task_configs:
            raise KeyError(f"Unknown task {task_name!r}; available tasks: {self._task_names}.")
        return self.heads[self._task_names.index(task_name)]

    def __call__(self, edge_index, x, batch_indices, edge_features=None, add_feats=None) -> dict[str, mx.array]:
        if self._encoder_uses_edges:
            h = self.encoder(
                edge_index=edge_index,
                x=x,
                edge_features=edge_features,
                batch_indices=batch_indices,
            )
        else:
            h = self.encoder(
                edge_index=edge_index,
                x=x,
                batch_indices=batch_indices,
            )

        if add_feats is not None and self._add_feat_dim == 0:
            raise ValueError("'add_feats' found in input batch, but model was initialized without add_feats.")
        if add_feats is None and self._add_feat_dim > 0:
            raise ValueError("Model expects 'add_feats', but input batch does not contain 'add_feats'.")

        if add_feats is not None:
            add_f = mx.array(add_feats, dtype=h.dtype)
            if add_f.ndim == 1:
                add_f = add_f.reshape(h.shape[0], -1)
            if add_f.shape[0] != h.shape[0]:
                raise ValueError(f"Batch mismatch: x has {h.shape[0]}, add_feats has {add_f.shape[0]}.")
            h = mx.concatenate([h, add_f], axis=1)

        for layer in self.shared_mlp:
            h = layer(h)

        return {name: head(h) for name, head in zip(self._task_names, self.heads)}

    def _extract_targets_and_mask(self, batch) -> tuple[mx.array, mx.array]:
        if not hasattr(batch, "y") or batch.y is None:
            raise ValueError("Input batch does not contain target labels 'y'.")
        y = batch.y.astype(mx.float32)
        if y.ndim == 1:
            if len(self._task_names) == 1:
                y = mx.expand_dims(y, axis=1)
            else:
                raise ValueError(
                    f"Expected 2D targets [batch_size, {len(self._task_names)}] for tasks {self._task_names}, "
                    f"got 1D shape {y.shape}. Pass multi-target lists/arrays to MolToGraph."
                )
        if y.shape[1] != len(self._task_names):
            raise ValueError(
                f"Expected {len(self._task_names)} target columns for tasks {self._task_names}, "
                f"got {y.shape[1]}."
            )

        if hasattr(batch, "y_mask") and batch.y_mask is not None:
            y_mask = batch.y_mask.astype(mx.bool_)
            if y_mask.ndim == 1:
                y_mask = mx.expand_dims(y_mask, axis=1)
        else:
            y_mask = ~mx.isnan(y)
            y = mx.where(y_mask, y, mx.zeros_like(y))
        return y, y_mask

    def _compute_batch_loss(
        self,
        outputs: dict[str, mx.array],
        y: mx.array,
        y_mask: mx.array,
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        zero = mx.array(0.0, dtype=mx.float32)
        one = mx.array(1.0, dtype=mx.float32)

        weighted_sum = zero
        weight_denom = zero
        task_losses: list[mx.array] = []
        task_counts: list[mx.array] = []

        for idx, name in enumerate(self._task_names):
            cfg = self._task_configs[name]
            out = outputs[name]
            target_i = y[:, idx]
            mask_f = y_mask[:, idx].astype(mx.float32)
            count = mx.sum(mask_f)
            has_valid = count > 0
            safe_count = mx.where(has_valid, count, one)

            if cfg["type"] == "classification":
                if cfg["criterion"] == "bce_with_logits":
                    per_sample = mlx.nn.losses.binary_cross_entropy(
                        out.reshape(-1), target_i, with_logits=True, reduction="none"
                    )
                else:
                    per_sample = mlx.nn.losses.cross_entropy(
                        out, target_i.astype(mx.int32), reduction="none"
                    )
            else:
                mean_i = self._y_means.get(name, 0.0)
                std_i = self._y_stds.get(name, 1.0)
                target_std = (target_i - mean_i) / std_i
                pred_i = out.reshape(-1)
                crit = cfg["criterion"]
                if crit == "mse":
                    per_sample = mlx.nn.losses.mse_loss(pred_i, target_std, reduction="none")
                elif crit == "l1":
                    per_sample = mlx.nn.losses.l1_loss(pred_i, target_std, reduction="none")
                elif crit == "huber":
                    per_sample = mlx.nn.losses.huber_loss(pred_i, target_std, reduction="none")
                elif crit == "smooth_l1":
                    per_sample = mlx.nn.losses.smooth_l1_loss(pred_i, target_std, reduction="none")
                else:
                    raise ValueError(f"Unsupported regression criterion {crit!r}.")

            task_loss = mx.where(has_valid, mx.sum(per_sample * mask_f) / safe_count, zero)
            task_losses.append(mx.stop_gradient(task_loss))
            task_counts.append(mx.stop_gradient(count))

            if self.loss_weighting == "equal":
                weighted_sum = weighted_sum + task_loss
                weight_denom = weight_denom + mx.where(has_valid, one, zero)
            elif self.loss_weighting == "manual":
                w = mx.array(cfg["weight"], dtype=mx.float32)
                weighted_sum = weighted_sum + mx.where(has_valid, w * task_loss, zero)
                weight_denom = weight_denom + mx.where(has_valid, w, zero)
            else:
                # Kendall et al. (2018) homoscedastic uncertainty weighting
                s = mx.clip(self.log_vars[idx], -6.0, 6.0)
                scale = 0.5 if cfg["type"] == "regression" else 1.0
                term = scale * mx.exp(-s) * task_loss + 0.5 * s
                weighted_sum = weighted_sum + mx.where(has_valid, term, zero)
                weight_denom = weight_denom + mx.where(has_valid, one, zero)

        if self.loss_weighting == "uncertainty":
            total_loss = weighted_sum
        else:
            safe_denom = mx.where(weight_denom > 0, weight_denom, one)
            total_loss = mx.where(weight_denom > 0, weighted_sum / safe_denom, zero)

        return total_loss, task_losses, task_counts

    def _batch_forward_args(self, batch) -> dict:
        args = {
            "edge_index": batch.edge_index,
            "x": batch.node_features,
            "batch_indices": batch.batch_indices,
        }
        if self._encoder_uses_edges and hasattr(batch, "edge_features"):
            args["edge_features"] = batch.edge_features
        if hasattr(batch, "add_feats") and batch.add_feats is not None:
            args["add_feats"] = batch.add_feats
        return args

    def _loss_and_outputs(self, batch, y: mx.array, y_mask: mx.array):
        outputs = self(**self._batch_forward_args(batch))
        total_loss, task_losses, task_counts = self._compute_batch_loss(outputs, y, y_mask)
        outputs_list = [mx.stop_gradient(outputs[name]) for name in self._task_names]
        return total_loss, (outputs_list, task_losses, task_counts)

    def _fit_regression_scalers(self, train_loader: Dataloader) -> None:
        all_y = []
        all_mask = []
        for batch in train_loader:
            y, y_mask = self._extract_targets_and_mask(batch)
            all_y.append(np.array(y))
            all_mask.append(np.array(y_mask))
        y_np = np.concatenate(all_y, axis=0)
        mask_np = np.concatenate(all_mask, axis=0)

        self._y_means = {}
        self._y_stds = {}
        for idx, name in enumerate(self._task_names):
            if self._task_configs[name]["type"] != "regression":
                continue
            valid = y_np[mask_np[:, idx], idx]
            if valid.size == 0:
                self._y_means[name] = 0.0
                self._y_stds[name] = 1.0
            else:
                mean_i = float(np.mean(valid))
                std_i = float(np.std(valid, ddof=0))
                if std_i == 0.0 or np.isnan(std_i):
                    std_i = 1.0
                self._y_means[name] = mean_i
                self._y_stds[name] = std_i

    def _run_epoch(self, batches, loss_and_grad_fn=None, optimizer=None):
        is_train = optimizer is not None
        self.train() if is_train else self.eval()

        t_count = len(self._task_names)
        batch_losses = []
        task_loss_batches = [[] for _ in range(t_count)]
        task_count_batches = [[] for _ in range(t_count)]
        raw_preds = [[] for _ in range(t_count)]
        all_y = []
        all_mask = []

        for batch in batches:
            y, y_mask = self._extract_targets_and_mask(batch)
            if is_train:
                (loss, (outputs_list, task_losses, task_counts)), grads = loss_and_grad_fn(batch, y, y_mask)
                optimizer.update(self, grads)
                mx.eval(self.parameters(), optimizer.state, loss, outputs_list, task_losses, task_counts)
            else:
                loss, (outputs_list, task_losses, task_counts) = self._loss_and_outputs(batch, y, y_mask)
                mx.eval(loss, outputs_list, task_losses, task_counts)

            batch_losses.append(loss)
            for i in range(t_count):
                task_loss_batches[i].append(task_losses[i])
                task_count_batches[i].append(task_counts[i])
                raw_preds[i].append(outputs_list[i])
            all_y.append(y)
            all_mask.append(y_mask)

        avg_loss = float(mx.mean(mx.stack(batch_losses)).item())
        per_task_loss = {}
        raw_preds_np = {}
        for i, name in enumerate(self._task_names):
            tl = np.array(mx.stack(task_loss_batches[i]), dtype=np.float64)
            tc = np.array(mx.stack(task_count_batches[i]), dtype=np.float64)
            total_obs = float(tc.sum())
            per_task_loss[name] = float((tl * tc).sum() / total_obs) if total_obs > 0 else np.nan
            raw_preds_np[name] = np.array(mx.concatenate(raw_preds[i], axis=0))

        y_np = np.array(mx.concatenate(all_y, axis=0))
        mask_np = np.array(mx.concatenate(all_mask, axis=0))

        task_metrics = _compute_task_metrics_np(
            y_np, mask_np, raw_preds_np, self._task_names, self._task_configs, self._y_means, self._y_stds
        )
        return avg_loss, per_task_loss, task_metrics

    def _record_epoch_metrics(self, train_loss, train_task_loss, train_metrics, test_loss, test_task_loss, test_metrics):
        self.metrics.train_loss.append(train_loss)
        self.metrics.test_loss.append(test_loss)
        for name in self._task_names:
            tm = self.metrics.tasks[name]
            tm.train_loss.append(train_task_loss[name])
            tm.test_loss.append(test_task_loss[name])
            tr_m = train_metrics[name]
            te_m = test_metrics[name]
            if self._task_configs[name]["type"] == "classification":
                tm.train_accuracy.append(tr_m["accuracy"])
                tm.test_accuracy.append(te_m["accuracy"])
                tm.train_precision.append(tr_m["precision"])
                tm.test_precision.append(te_m["precision"])
                tm.recall.append(te_m["recall"])
                tm.f1_score.append(te_m["f1"])
                tm.mcc.append(te_m["mcc"])
                tm.roc_auc.append(te_m["roc_auc"])
            else:
                tm.train_mae.append(tr_m["mae"])
                tm.test_mae.append(te_m["mae"])
                tm.train_rmse.append(tr_m["rmse"])
                tm.test_rmse.append(te_m["rmse"])
                tm.train_r2.append(tr_m["r2"])
                tm.test_r2.append(te_m["r2"])

    def fit(
        self,
        train_loader: Union[Dataloader, list],
        test_loader: Union[Dataloader, list],
        epochs: int = 10,
        optimizer: str = "adam",
        lr: float = 0.01,
        wt_decay: float = 0.0,
        batch_size: int = 1024,
        device: str = "auto",
        estop: Optional[Union[EarlyStopping, int]] = None,
        pbar: bool = True,
        verbose: bool = False,
        device_warn: bool = True,
        seed: int = 42,
        restore_best: bool = True,
    ):
        """
        Train MLX MultiTaskGNN with masked per-task losses, per-task regression standardization, and best-epoch restoration.
        """
        self.metrics = MultiTaskMetrics(self._task_names)
        set_seed(seed)

        fit_device = get_device(device)
        mx.set_default_device(fit_device)

        # Snapshot initial weights before first training pass so reset_params() always works
        if self._initial_params is None:
            self.store_initial_params()

        if optimizer in MLX_OPTI:
            if optimizer in NO_WT_DECAY:
                opt = MLX_OPTI[optimizer](learning_rate=lr)
            else:
                opt = MLX_OPTI[optimizer](learning_rate=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! Only {list(MLX_OPTI.keys())} allowed!")

        np.random.seed(seed)
        if isinstance(train_loader, list):
            train_loader = Dataloader(train_loader, batch_size=batch_size, shuffle=True)
        if isinstance(test_loader, list):
            test_loader = Dataloader(test_loader, batch_size=batch_size, shuffle=False)
        # Pre-collate static validation batches once across epochs
        test_batches = list(test_loader)

        if device_warn:
            if device == "auto" or device == "gpu":
                print(f"Model running on {state_apple_silicon_device()}")
            else:
                print("Model running on CPU")

        self._fit_regression_scalers(train_loader)

        if estop is not None and isinstance(estop, EarlyStopping):
            estop.reset()
        elif estop is not None and isinstance(estop, int):
            estop = EarlyStopping(monitor="val_loss", patience=estop, mode="min")

        monitor_metric = estop.monitor if estop is not None else "val_loss"
        if monitor_metric != "val_loss":
            raise ValueError("MultiTaskGNN EarlyStopping monitor must be 'val_loss'.")
        monitor_mode = estop.mode if estop is not None else "min"
        best_score = float("inf") if monitor_mode == "min" else -float("inf")
        is_better = estop.monitor_op if estop is not None else (lambda cur, best: cur < best)
        best_epoch = None
        best_state = None

        loss_and_grad_fn = mlx.nn.value_and_grad(self, self._loss_and_outputs)
        mx.eval(self.parameters())

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            tr_loss, tr_task_loss, tr_metrics = self._run_epoch(
                train_loader, loss_and_grad_fn=loss_and_grad_fn, optimizer=opt
            )
            te_loss, te_task_loss, te_metrics = self._run_epoch(
                test_batches, loss_and_grad_fn=None, optimizer=None
            )
            self._record_epoch_metrics(tr_loss, tr_task_loss, tr_metrics, te_loss, te_task_loss, te_metrics)

            if verbose:
                print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")

            if is_better(te_loss, best_score):
                best_score = te_loss
                best_epoch = epoch
                if restore_best:
                    mx.eval(self.parameters())
                    best_state = [(k, mx.array(v)) for k, v in tree_flatten(self.parameters())]

            if estop is not None:
                estop(te_loss)
                if estop.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs!")
                    break

        if restore_best and best_state is not None:
            self.update(tree_unflatten(best_state))
            mx.eval(self.parameters())

        self.is_fitted = True
        best_train_epoch = int(np.argmin(self.metrics.train_loss))
        best_test_epoch = best_epoch if best_epoch is not None else int(np.argmin(self.metrics.test_loss))
        self.metrics.best_epoch = best_test_epoch
        for name in self._task_names:
            self.metrics.tasks[name].best_epoch = best_test_epoch

        if device_warn:
            print(f"Best Training Loss: {round(self.metrics.train_loss[best_train_epoch], 3)}")
            print(f"Best Test Loss: {round(self.metrics.test_loss[best_test_epoch], 3)}")
            for name in self._task_names:
                tm = self.metrics.tasks[name]
                if self._task_configs[name]["type"] == "classification":
                    print(f"  [{name}] Test Acc: {round(tm.test_accuracy[best_test_epoch], 3)} | "
                          f"ROC_AUC: {round(tm.roc_auc[best_test_epoch], 3)}")
                else:
                    print(f"  [{name}] Test MAE: {round(tm.test_mae[best_test_epoch], 3)} | "
                          f"R2: {round(tm.test_r2[best_test_epoch], 3)}")

    def _collect_raw_predictions(
        self,
        data: Union[list, Dataloader],
        batch_size: int = 1024,
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        pbar: bool = False,
        collect_targets: bool = False,
    ):
        self.eval()
        if isinstance(data, list):
            loader = Dataloader(data, batch_size=batch_size, shuffle=False)
        elif isinstance(data, Dataloader):
            loader = data
        else:
            raise TypeError(f"data must be a list of graphs or Dataloader (got {type(data).__name__}).")

        if add_feats is not None and self._add_feat_dim == 0:
            raise ValueError("'add_feats' passed to predict/evaluate, but model was initialized without add_feats.")

        raw_preds = {name: [] for name in self._task_names}
        all_y = []
        all_mask = []

        for batch in tqdm(loader, disable=not pbar):
            outputs = self(**self._batch_forward_args(batch))
            mx.eval(*outputs.values())
            for name in self._task_names:
                raw_preds[name].append(np.array(outputs[name]))
            if collect_targets:
                y, y_mask = self._extract_targets_and_mask(batch)
                all_y.append(np.array(y))
                all_mask.append(np.array(y_mask))

        raw_preds_np = {name: np.concatenate(raw_preds[name], axis=0) for name in self._task_names}
        if collect_targets:
            return raw_preds_np, np.concatenate(all_y, axis=0), np.concatenate(all_mask, axis=0)
        return raw_preds_np

    def predict(
        self,
        data: Union[list, Dataloader],
        batch_size: int = 1024,
        binary: Union[bool, str, None] = True,
        cutoff: float = 0.5,
        return_type: str = "dict",
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        pbar: bool = False,
    ):
        """
        Generate per-task predictions on molecular graphs.
        Returns a dict `{task_name: np.ndarray}` (default) or `pd.DataFrame` when `return_type="dataframe"`.
        """
        raw_preds_np = self._collect_raw_predictions(
            data=data, batch_size=batch_size, add_feats=add_feats, pbar=pbar, collect_targets=False
        )
        return _format_predictions_np(
            raw_preds_np,
            self._task_names,
            self._task_configs,
            self._y_means,
            self._y_stds,
            binary=binary,
            cutoff=cutoff,
            return_type=return_type,
        )

    def evaluate(
        self,
        data: Union[list, Dataloader],
        batch_size: int = 1024,
        cutoff: float = 0.5,
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        pbar: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate the model on labeled multi-task graphs and return a per-task metrics DataFrame.
        Missing (`NaN`) labels are excluded per task via `y_mask`.
        """
        raw_preds_np, y_np, mask_np = self._collect_raw_predictions(
            data=data, batch_size=batch_size, add_feats=add_feats, pbar=pbar, collect_targets=True
        )
        task_metrics = _compute_task_metrics_np(
            y_np,
            mask_np,
            raw_preds_np,
            self._task_names,
            self._task_configs,
            self._y_means,
            self._y_stds,
            cutoff=cutoff,
        )
        return _build_evaluation_df(task_metrics, self._task_names)

    def store_initial_params(self):
        """Snapshot initial parameters for later reset (e.g. across cross-validation folds)."""
        if self._initial_params is None:
            mx.eval(self.parameters())
            self._initial_params = [(k, mx.array(v)) for k, v in tree_flatten(self.parameters())]

    def reset_params(self):
        """Reset model weights to the initial parameter snapshot and clear regression scalers."""
        if self._initial_params is None:
            raise ValueError("Initial parameters not stored! Call store_initial_params() or fit() first.")
        restored = [(k, mx.array(v)) for k, v in self._initial_params]
        self.update(tree_unflatten(restored))
        mx.eval(self.parameters())
        self._y_means = {}
        self._y_stds = {}
        self.is_fitted = False

    def save(self, filename: str):
        """
        Save MLX MultiTaskGNN weights, task configuration metadata, and per-task regression scalers to a `.npz` file.
        """
        mx.eval(self.parameters())
        save_dict = dict(tree_flatten(self.parameters()))
        meta = {
            "format": CHECKPOINT_FORMAT,
            "task_names": list(self._task_names),
            "task_configs": copy.deepcopy(self._task_configs),
            "loss_weighting": self.loss_weighting,
            "y_means": dict(self._y_means),
            "y_stds": dict(self._y_stds),
        }
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        save_dict[META_KEY] = mx.array(meta_bytes)
        mx.savez(filename, **save_dict)
        return "Model saved!"

    def load(self, model_path: str):
        """
        Load MLX MultiTaskGNN weights and per-task regression scalers from a `.npz` file after validating compatibility.
        """
        weights = mx.load(model_path)
        if META_KEY not in weights:
            raise ValueError(
                "Mismatch! Checkpoint is not a valid MultiTaskGNN '.npz' file (missing multitask metadata)."
            )
        meta_bytes = np.array(weights[META_KEY], dtype=np.uint8).tobytes()
        meta = json.loads(meta_bytes.decode("utf-8"))
        _validate_multitask_checkpoint(
            meta, self._task_names, self._task_configs, self.loss_weighting
        )

        model_params = [(k, v) for k, v in weights.items() if k != META_KEY]
        self.update(tree_unflatten(model_params))
        mx.eval(self.parameters())

        self._y_means = {k: float(v) for k, v in meta.get("y_means", {}).items()}
        self._y_stds = {k: float(v) for k, v in meta.get("y_stds", {}).items()}
        self.is_fitted = True
        self.eval()
        print("Model loaded!")
        return self


if __name__ == "__main__":
    import doctest

    doctest.testmod()
