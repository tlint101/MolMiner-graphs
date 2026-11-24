import copy
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mminer_graphs.model.graph_modeling import EarlyStopping, Metrics
from mminer_graphs.model.model_utils import (
    OPTIMIZERS,
    get_device,
    make_generator,
    seed_worker,
    state_apple_silicon_device,
)

VALID_CLS_CRITERIA = ("bce_with_logits", "cross_entropy")
VALID_REG_CRITERIA = ("mse", "l1", "huber", "smooth_l1")
VALID_LOSS_WEIGHTINGS = ("equal", "manual", "uncertainty")
CHECKPOINT_FORMAT = "mminer_multitask_v1"


def _parse_tasks(
    tasks: Union[list[str], tuple[str, ...], dict],
    default_task: str = "classification",
) -> tuple[list[str], dict[str, dict]]:
    """
    Normalize a task specification (list/tuple of names or dict of configs) into
    an ordered list of task names and a normalized per-task config dictionary.
    """
    default_norm = str(default_task).strip().lower()
    if default_norm not in ("classification", "regression"):
        raise ValueError(
            f"default_task must be 'classification' or 'regression' (got {default_task!r})."
        )

    if isinstance(tasks, (list, tuple)):
        if len(tasks) == 0:
            raise ValueError("tasks cannot be empty.")
        raw_items = [(name, {"type": default_norm}) for name in tasks]
    elif isinstance(tasks, dict):
        if len(tasks) == 0:
            raise ValueError("tasks cannot be empty.")
        raw_items = list(tasks.items())
    else:
        raise TypeError(
            f"tasks must be a list/tuple of task names or a dict of task configs (got {type(tasks).__name__})."
        )

    task_names: list[str] = []
    task_configs: dict[str, dict] = {}

    for name, raw_cfg in raw_items:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Task names must be non-empty strings (got {name!r}).")
        if name in task_configs:
            raise ValueError(f"Duplicate task name: {name!r}.")

        if isinstance(raw_cfg, str):
            cfg_in = {"type": raw_cfg}
        elif raw_cfg is None:
            cfg_in = {"type": default_norm}
        elif isinstance(raw_cfg, dict):
            cfg_in = dict(raw_cfg)
        else:
            raise TypeError(
                f"Config for task {name!r} must be a dict, str, or None (got {type(raw_cfg).__name__})."
            )

        task_type = str(cfg_in.get("type", default_norm)).strip().lower()
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"Task {name!r} has invalid type {task_type!r}; expected 'classification' or 'regression'."
            )

        weight = float(cfg_in.get("weight", 1.0))
        if weight <= 0 or np.isnan(weight) or np.isinf(weight):
            raise ValueError(f"Task {name!r} weight must be a finite positive number (got {weight!r}).")

        if task_type == "classification":
            num_classes = cfg_in.get("num_classes", 2)
            if not isinstance(num_classes, (int, np.integer)) or isinstance(num_classes, bool) or int(num_classes) < 2:
                raise ValueError(
                    f"Classification task {name!r} requires integer num_classes >= 2 (got {num_classes!r})."
                )
            num_classes = int(num_classes)
            default_crit = "bce_with_logits" if num_classes == 2 else "cross_entropy"
            criterion = str(cfg_in.get("criterion", default_crit)).strip().lower()
            if criterion == "bce":
                criterion = "bce_with_logits"
            if criterion not in VALID_CLS_CRITERIA:
                raise ValueError(
                    f"Classification task {name!r} criterion must be one of {VALID_CLS_CRITERIA} (got {criterion!r})."
                )
            if criterion == "bce_with_logits" and num_classes != 2:
                raise ValueError(
                    f"Task {name!r} uses 'bce_with_logits' which only supports num_classes=2 (got {num_classes})."
                )
            out_dim = 1 if criterion == "bce_with_logits" else num_classes
            task_configs[name] = {
                "type": "classification",
                "num_classes": num_classes,
                "out_dim": out_dim,
                "criterion": criterion,
                "weight": weight,
            }
        else:
            criterion = str(cfg_in.get("criterion", "mse")).strip().lower()
            if criterion not in VALID_REG_CRITERIA:
                raise ValueError(
                    f"Regression task {name!r} criterion must be one of {VALID_REG_CRITERIA} (got {criterion!r})."
                )
            task_configs[name] = {
                "type": "regression",
                "num_classes": None,
                "out_dim": 1,
                "criterion": criterion,
                "weight": weight,
            }
        task_names.append(name)

    return task_names, task_configs


def _resolve_add_feat_dim(add_feats: Optional[Union[np.ndarray, list, int]]) -> int:
    if add_feats is None:
        return 0
    if isinstance(add_feats, (int, np.integer)) and not isinstance(add_feats, bool):
        dim = int(add_feats)
        if dim < 0:
            raise ValueError(f"add_feats int must be >= 0 (got {dim}).")
        return dim
    if isinstance(add_feats, (list, tuple, np.ndarray)):
        arr = np.asarray(add_feats)
        if arr.ndim == 2:
            return int(arr.shape[1])
        if arr.ndim == 1:
            return int(arr.shape[0])
        raise ValueError(f"add_feats array must be 1D or 2D (got shape {arr.shape}).")
    raise TypeError(f"Unsupported add_feats type {type(add_feats).__name__}.")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def _compute_task_metrics_np(
    y_true: np.ndarray,
    y_mask: np.ndarray,
    raw_preds: dict[str, np.ndarray],
    task_names: list[str],
    task_configs: dict[str, dict],
    y_means: dict[str, float],
    y_stds: dict[str, float],
    cutoff: float = 0.5,
) -> dict[str, dict]:
    """
    Compute per-task classification or regression metrics over observed (y_mask == True) entries.
    """
    results: dict[str, dict] = {}
    for idx, name in enumerate(task_names):
        cfg = task_configs[name]
        obs = y_mask[:, idx].astype(bool)
        n_obs = int(obs.sum())
        raw = raw_preds[name]

        if cfg["type"] == "classification":
            if n_obs == 0:
                results[name] = {
                    "type": "Classification",
                    "n_observed": 0,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "mcc": np.nan,
                    "roc_auc": np.nan,
                }
                continue

            yt = y_true[obs, idx].astype(int)
            raw_obs = raw[obs]
            num_classes = cfg["num_classes"]

            if num_classes == 2:
                if cfg["out_dim"] == 1:
                    proba_pos = _sigmoid_np(raw_obs.reshape(-1))
                else:
                    proba_pos = _softmax_np(raw_obs)[:, 1]
                yp = (proba_pos >= cutoff).astype(int)
                avg = "binary"
                if len(np.unique(yt)) >= 2:
                    roc_auc = float(roc_auc_score(yt, proba_pos))
                else:
                    roc_auc = np.nan
            else:
                probs = _softmax_np(raw_obs)
                yp = np.argmax(probs, axis=1)
                avg = "macro"
                try:
                    roc_auc = float(roc_auc_score(yt, probs, multi_class="ovr", average="macro"))
                except ValueError:
                    roc_auc = np.nan

            results[name] = {
                "type": "Classification",
                "n_observed": n_obs,
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, average=avg, zero_division=0)),
                "recall": float(recall_score(yt, yp, average=avg, zero_division=0)),
                "f1": float(f1_score(yt, yp, average=avg, zero_division=0)),
                "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) >= 2 else np.nan,
                "roc_auc": roc_auc,
            }
        else:
            if n_obs == 0:
                results[name] = {
                    "type": "Regression",
                    "n_observed": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                }
                continue

            yt = y_true[obs, idx].astype(float)
            mean_i = y_means.get(name, 0.0)
            std_i = y_stds.get(name, 1.0)
            yp = raw[obs].reshape(-1).astype(float) * std_i + mean_i

            results[name] = {
                "type": "Regression",
                "n_observed": n_obs,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(root_mean_squared_error(yt, yp)),
                "r2": float(r2_score(yt, yp)) if n_obs >= 2 else np.nan,
            }

    return results


def _format_predictions_np(
    raw_preds: dict[str, np.ndarray],
    task_names: list[str],
    task_configs: dict[str, dict],
    y_means: dict[str, float],
    y_stds: dict[str, float],
    binary: Union[bool, str, None] = True,
    cutoff: float = 0.5,
    return_type: str = "dict",
):
    """
    Convert raw per-task logits/standardized outputs into final predictions as a dict or DataFrame.
    """
    if binary not in (True, False, None, "multi", "proba"):
        raise ValueError("binary must be True, False, None, 'multi', or 'proba'.")
    if return_type not in ("dict", "dataframe"):
        raise ValueError(f"return_type must be 'dict' or 'dataframe' (got {return_type!r}).")

    formatted: dict[str, np.ndarray] = {}
    for name in task_names:
        cfg = task_configs[name]
        raw = raw_preds[name]

        if cfg["type"] == "regression":
            mean_i = y_means.get(name, 0.0)
            std_i = y_stds.get(name, 1.0)
            formatted[name] = raw.reshape(-1).astype(np.float32) * std_i + mean_i
        else:
            num_classes = cfg["num_classes"]
            if num_classes == 2:
                if cfg["out_dim"] == 1:
                    proba_pos = _sigmoid_np(raw.reshape(-1))
                else:
                    proba_pos = _softmax_np(raw)[:, 1]

                if binary is True or binary == "multi":
                    formatted[name] = (proba_pos >= cutoff).astype(int)
                elif binary is False or binary is None:
                    formatted[name] = proba_pos
                elif binary == "proba":
                    formatted[name] = np.column_stack([1.0 - proba_pos, proba_pos])
            else:
                probs = _softmax_np(raw)
                if binary == "proba":
                    formatted[name] = probs
                else:
                    formatted[name] = np.argmax(probs, axis=1)

    if return_type == "dict":
        return formatted

    # DataFrame output: expand 2D probability columns cleanly as "<task>_class_<c>"
    cols = {}
    for name in task_names:
        arr = formatted[name]
        if arr.ndim == 1:
            cols[name] = arr
        else:
            for c in range(arr.shape[1]):
                cols[f"{name}_class_{c}"] = arr[:, c]
    return pd.DataFrame(cols)


def _build_evaluation_df(task_metrics: dict[str, dict], task_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in task_names:
        m = task_metrics[name]
        rows.append(
            {
                "Task": name,
                "TaskType": m["type"],
                "N_Observed": m["n_observed"],
                "Accuracy": m.get("accuracy", np.nan),
                "Precision": m.get("precision", np.nan),
                "Recall": m.get("recall", np.nan),
                "F1": m.get("f1", np.nan),
                "MCC": m.get("mcc", np.nan),
                "ROC_AUC": m.get("roc_auc", np.nan),
                "MAE": m.get("mae", np.nan),
                "RMSE": m.get("rmse", np.nan),
                "R2": m.get("r2", np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("Task")


def _validate_multitask_checkpoint(
    saved_meta: dict,
    expected_task_names: list[str],
    expected_task_configs: dict[str, dict],
    expected_loss_weighting: str,
) -> None:
    if not isinstance(saved_meta, dict) or saved_meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "Mismatch! Checkpoint is not a valid MultiTaskGNN checkpoint."
        )
    saved_names = saved_meta.get("task_names")
    if saved_names != expected_task_names:
        raise ValueError(
            f"Mismatch! Checkpoint tasks {saved_names} do not match model tasks {expected_task_names}."
        )
    saved_configs = saved_meta.get("task_configs", {})
    for name in expected_task_names:
        sc = saved_configs.get(name, {})
        ec = expected_task_configs[name]
        if sc.get("type") != ec["type"] or sc.get("out_dim") != ec["out_dim"]:
            raise ValueError(
                f"Mismatch! Task {name!r} config in checkpoint ({sc.get('type')}, out_dim={sc.get('out_dim')}) "
                f"does not match model ({ec['type']}, out_dim={ec['out_dim']})."
            )
    saved_weighting = saved_meta.get("loss_weighting")
    if (saved_weighting == "uncertainty") != (expected_loss_weighting == "uncertainty"):
        raise ValueError(
            f"Mismatch! Checkpoint loss_weighting={saved_weighting!r} is incompatible with "
            f"model loss_weighting={expected_loss_weighting!r}."
        )


class MultiTaskMetrics:
    """
    Training history container for MultiTaskGNN.
    Stores aggregate train_loss/test_loss/best_epoch alongside per-task Metrics objects on `self.tasks[task_name]`.
    """

    def __init__(self, task_names: list[str]):
        self.best_epoch: Optional[int] = None
        self.train_loss: list[float] = []
        self.test_loss: list[float] = []
        self.tasks: dict[str, Metrics] = {name: Metrics() for name in task_names}


class MultiTaskGNN(nn.Module):
    """
    Composable multi-task molecular graph neural network (PyTorch / PyTorch Geometric).

    Wraps any latent-space GNN encoder (`GCN`, `GIN`, `GINE`, `GSAGE`, `GAT`) with an optional shared MLP trunk
    and per-task prediction heads supporting mixed binary classification, multi-class classification, and
    regression tasks with `NaN`-masked sparse labels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tasks: Union[list[str], tuple[str, ...], dict],
        default_task: str = "classification",
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        shared_mlp_layers: int = 0,
        head_mlp_layers: int = 1,
        mlp_dropout: float = 0.0,
        loss_weighting: str = "equal",
    ):
        """
        :param encoder: nn.Module
            Graph encoder from `mminer_graphs.model.latent_space` exposing `.out_dim` (`GCN`, `GIN`, `GINE`, `GSAGE`, `GAT`).
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
                "(use an encoder from mminer_graphs.model.latent_space)."
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
        self._task_names, self._task_configs = _parse_tasks(tasks, default_task=default_task)
        self.task = "MultiTask"
        self.loss_weighting = loss_weighting
        self.mlp_dropout = float(mlp_dropout)
        self.add_feats = add_feats
        self._add_feat_dim = _resolve_add_feat_dim(add_feats)
        self.is_fitted = False
        self._y_means: dict[str, float] = {}
        self._y_stds: dict[str, float] = {}
        self.metrics = MultiTaskMetrics(self._task_names)

        # Optional shared MLP trunk
        trunk_dim = int(encoder.out_dim) + self._add_feat_dim
        self.shared_mlp = nn.ModuleList()
        dim = trunk_dim
        for _ in range(shared_mlp_layers):
            next_dim = max(1, round(dim / 2))
            self.shared_mlp.append(
                nn.Sequential(
                    nn.Linear(dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.ReLU(),
                    nn.Dropout(self.mlp_dropout),
                )
            )
            dim = next_dim

        # Per-task prediction heads (indexed 0..T-1 to allow arbitrary characters in task names)
        self.heads = nn.ModuleList()
        for name in self._task_names:
            out_dim = self._task_configs[name]["out_dim"]
            layers: list[nn.Module] = []
            h_dim = dim
            for _ in range(head_mlp_layers - 1):
                next_dim = max(1, round(h_dim / 2))
                layers.extend(
                    [
                        nn.Linear(h_dim, next_dim),
                        nn.BatchNorm1d(next_dim),
                        nn.ReLU(),
                        nn.Dropout(self.mlp_dropout),
                    ]
                )
                h_dim = next_dim
            layers.append(nn.Linear(h_dim, out_dim))
            self.heads.append(nn.Sequential(*layers))

        # Learnable homoscedastic log-variances for Kendall uncertainty weighting
        if self.loss_weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self._task_names), dtype=torch.float32))
        else:
            self.log_vars = None

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)

    @property
    def task_configs(self) -> dict[str, dict]:
        return copy.deepcopy(self._task_configs)

    @property
    def y_means(self) -> dict[str, float]:
        return dict(self._y_means)

    @property
    def y_stds(self) -> dict[str, float]:
        return dict(self._y_stds)

    def get_head(self, task_name: str) -> nn.Module:
        """Return the prediction head module for a given task name."""
        if task_name not in self._task_configs:
            raise KeyError(f"Unknown task {task_name!r}; available tasks: {self._task_names}.")
        return self.heads[self._task_names.index(task_name)]

    def forward(self, data) -> dict[str, torch.Tensor]:
        h = self.encoder(data)

        has_add = hasattr(data, "add_feats") and data.add_feats is not None
        if has_add and self._add_feat_dim == 0:
            raise ValueError("'add_feats' found in input batch, but model was initialized without add_feats.")
        if not has_add and self._add_feat_dim > 0:
            raise ValueError("Model expects 'add_feats', but input batch does not contain 'add_feats'.")

        if has_add:
            add_f = data.add_feats
            if not isinstance(add_f, torch.Tensor):
                add_f = torch.tensor(add_f, device=h.device, dtype=h.dtype)
            else:
                add_f = add_f.to(device=h.device, dtype=h.dtype)
            if add_f.dim() == 1:
                add_f = add_f.unsqueeze(0)
            if add_f.size(0) == 1 and h.size(0) > 1:
                add_f = add_f.expand(h.size(0), -1)
            elif add_f.size(0) != h.size(0):
                raise ValueError(f"add_feats batch size {add_f.size(0)} does not match graph batch size {h.size(0)}.")
            h = torch.cat([h, add_f], dim=1)

        for layer in self.shared_mlp:
            h = layer(h)

        return {name: head(h) for name, head in zip(self._task_names, self.heads)}

    def _extract_targets_and_mask(self, data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if data.y is None:
            raise ValueError("Input batch does not contain target labels 'y'.")
        y = data.y.to(device=device, dtype=torch.float32)
        if y.dim() == 1:
            if len(self._task_names) == 1:
                y = y.unsqueeze(1)
            else:
                raise ValueError(
                    f"Expected 2D targets [batch_size, {len(self._task_names)}] for tasks {self._task_names}, "
                    f"got 1D shape {tuple(y.shape)}. Pass multi-target lists/arrays to MolToGraph."
                )
        if y.shape[1] != len(self._task_names):
            raise ValueError(
                f"Expected {len(self._task_names)} target columns for tasks {self._task_names}, "
                f"got {y.shape[1]}."
            )

        if hasattr(data, "y_mask") and data.y_mask is not None:
            y_mask = data.y_mask.to(device=device, dtype=torch.bool)
            if y_mask.dim() == 1:
                y_mask = y_mask.unsqueeze(1)
        else:
            y_mask = ~torch.isnan(y)
            y = torch.where(y_mask, y, torch.zeros_like(y))
        return y, y_mask

    def _compute_batch_loss(
        self,
        outputs: dict[str, torch.Tensor],
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        device = y.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        one = torch.ones((), device=device, dtype=torch.float32)

        weighted_sum = zero
        weight_denom = zero
        task_losses: list[torch.Tensor] = []
        task_counts: list[torch.Tensor] = []

        for idx, name in enumerate(self._task_names):
            cfg = self._task_configs[name]
            out = outputs[name]
            target_i = y[:, idx]
            mask_f = y_mask[:, idx].to(dtype=torch.float32)
            count = mask_f.sum()
            has_valid = count > 0
            safe_count = torch.where(has_valid, count, one)

            if cfg["type"] == "classification":
                if cfg["criterion"] == "bce_with_logits":
                    per_sample = F.binary_cross_entropy_with_logits(
                        out.squeeze(-1), target_i, reduction="none"
                    )
                else:
                    per_sample = F.cross_entropy(out, target_i.long(), reduction="none")
            else:
                mean_i = self._y_means.get(name, 0.0)
                std_i = self._y_stds.get(name, 1.0)
                target_std = (target_i - mean_i) / std_i
                pred_i = out.squeeze(-1)
                crit = cfg["criterion"]
                if crit == "mse":
                    per_sample = F.mse_loss(pred_i, target_std, reduction="none")
                elif crit == "l1":
                    per_sample = F.l1_loss(pred_i, target_std, reduction="none")
                elif crit == "huber":
                    per_sample = F.huber_loss(pred_i, target_std, reduction="none")
                elif crit == "smooth_l1":
                    per_sample = F.smooth_l1_loss(pred_i, target_std, reduction="none")
                else:
                    raise ValueError(f"Unsupported regression criterion {crit!r}.")

            task_loss = torch.where(has_valid, (per_sample * mask_f).sum() / safe_count, zero)
            task_losses.append(task_loss.detach())
            task_counts.append(count.detach())

            if self.loss_weighting == "equal":
                weighted_sum = weighted_sum + task_loss
                weight_denom = weight_denom + torch.where(has_valid, one, zero)
            elif self.loss_weighting == "manual":
                w = torch.tensor(cfg["weight"], device=device, dtype=torch.float32)
                weighted_sum = weighted_sum + torch.where(has_valid, w * task_loss, zero)
                weight_denom = weight_denom + torch.where(has_valid, w, zero)
            else:
                # Kendall et al. (2018) homoscedastic uncertainty weighting
                s = torch.clamp(self.log_vars[idx], -6.0, 6.0)
                scale = 0.5 if cfg["type"] == "regression" else 1.0
                term = scale * torch.exp(-s) * task_loss + 0.5 * s
                weighted_sum = weighted_sum + torch.where(has_valid, term, zero)
                weight_denom = weight_denom + torch.where(has_valid, one, zero)

        if self.loss_weighting == "uncertainty":
            total_loss = weighted_sum
        else:
            safe_denom = torch.where(weight_denom > 0, weight_denom, one)
            total_loss = torch.where(weight_denom > 0, weighted_sum / safe_denom, zero)

        return total_loss, task_losses, task_counts

    def _fit_regression_scalers(self, train_loader: DataLoader) -> None:
        all_y = []
        all_mask = []
        for batch in train_loader:
            y, y_mask = self._extract_targets_and_mask(batch, device=torch.device("cpu"))
            all_y.append(y.numpy())
            all_mask.append(y_mask.numpy())
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

    def _run_epoch(
        self,
        loader: DataLoader,
        fit_device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        is_train = optimizer is not None
        self.train() if is_train else self.eval()

        total_loss_sum = 0.0
        n_batches = 0
        t_count = len(self._task_names)
        task_loss_weighted = np.zeros(t_count, dtype=np.float64)
        task_obs_total = np.zeros(t_count, dtype=np.float64)

        raw_preds = {name: [] for name in self._task_names}
        all_y = []
        all_mask = []

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for data in loader:
                data = data.to(fit_device)
                y, y_mask = self._extract_targets_and_mask(data, fit_device)

                if is_train:
                    optimizer.zero_grad()

                outputs = self(data)
                loss, task_losses, task_counts = self._compute_batch_loss(outputs, y, y_mask)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss_sum += float(loss.detach().item())
                n_batches += 1
                for i in range(t_count):
                    cnt_i = float(task_counts[i].item())
                    task_loss_weighted[i] += float(task_losses[i].item()) * cnt_i
                    task_obs_total[i] += cnt_i

                for name in self._task_names:
                    raw_preds[name].append(outputs[name].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_mask.append(y_mask.detach().cpu().numpy())

        avg_loss = total_loss_sum / max(n_batches, 1)
        per_task_loss = {
            name: (float(task_loss_weighted[i] / task_obs_total[i]) if task_obs_total[i] > 0 else np.nan)
            for i, name in enumerate(self._task_names)
        }
        raw_preds_np = {name: np.concatenate(raw_preds[name], axis=0) for name in self._task_names}
        y_np = np.concatenate(all_y, axis=0)
        mask_np = np.concatenate(all_mask, axis=0)

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
        train_loader: Union[DataLoader, list],
        test_loader: Union[DataLoader, list],
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
        restore_best: bool = True,
    ):
        """
        Train MultiTaskGNN with masked per-task losses, per-task regression standardization, and best-epoch restoration.
        """
        self.metrics = MultiTaskMetrics(self._task_names)
        fit_device = get_device(device)
        self.to(fit_device)

        if optimizer in OPTIMIZERS:
            opt = OPTIMIZERS[optimizer](self.parameters(), lr=lr, weight_decay=wt_decay)
        elif optimizer is None:
            raise ValueError("No optimizer defined!")
        else:
            raise ValueError(f"Optimizer not recognized! Only {list(OPTIMIZERS.keys())} allowed!")

        if isinstance(train_loader, list):
            train_loader = DataLoader(
                train_loader,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=make_generator(42),
            )
        if isinstance(test_loader, list):
            test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

        if device_warn:
            if device == "mps":
                print(f"Model and Tensor running with MPS on {state_apple_silicon_device()}")
            else:
                print(f"Model and Tensor running on {fit_device}")

        # Fit per-task regression mean/std on observed training labels only
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

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not pbar):
            tr_loss, tr_task_loss, tr_metrics = self._run_epoch(train_loader, fit_device, optimizer=opt)
            te_loss, te_task_loss, te_metrics = self._run_epoch(test_loader, fit_device, optimizer=None)
            self._record_epoch_metrics(tr_loss, tr_task_loss, tr_metrics, te_loss, te_task_loss, te_metrics)

            if verbose:
                print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")

            if is_better(te_loss, best_score):
                best_score = te_loss
                best_epoch = epoch
                if restore_best:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            if estop is not None:
                estop(te_loss)
                if estop.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs!")
                    break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            self.to(fit_device)

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
        data: Union[list, DataLoader],
        batch_size: int = 1024,
        device: str = "auto",
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        pbar: bool = False,
        collect_targets: bool = False,
    ):
        fit_device = get_device(device)
        self.to(fit_device)
        self.eval()

        if isinstance(data, list):
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        elif isinstance(data, DataLoader):
            loader = data
        else:
            raise TypeError(f"data must be a list of graphs or DataLoader (got {type(data).__name__}).")

        if add_feats is not None and self._add_feat_dim == 0:
            raise ValueError("'add_feats' passed to predict/evaluate, but model was initialized without add_feats.")

        raw_preds = {name: [] for name in self._task_names}
        all_y = []
        all_mask = []

        with torch.no_grad():
            for batch in tqdm(loader, disable=not pbar):
                batch = batch.to(fit_device)
                outputs = self(batch)
                for name in self._task_names:
                    raw_preds[name].append(outputs[name].detach().cpu().numpy())
                if collect_targets:
                    y, y_mask = self._extract_targets_and_mask(batch, fit_device)
                    all_y.append(y.detach().cpu().numpy())
                    all_mask.append(y_mask.detach().cpu().numpy())

        raw_preds_np = {name: np.concatenate(raw_preds[name], axis=0) for name in self._task_names}
        if collect_targets:
            return raw_preds_np, np.concatenate(all_y, axis=0), np.concatenate(all_mask, axis=0)
        return raw_preds_np

    def predict(
        self,
        data: Union[list, DataLoader],
        batch_size: int = 1024,
        device: str = "auto",
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
            data=data, batch_size=batch_size, device=device, add_feats=add_feats, pbar=pbar, collect_targets=False
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
        data: Union[list, DataLoader],
        batch_size: int = 1024,
        device: str = "auto",
        cutoff: float = 0.5,
        add_feats: Optional[Union[np.ndarray, list, int]] = None,
        pbar: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate the model on labeled multi-task graphs and return a per-task metrics DataFrame.
        Missing (`NaN`) labels are excluded per task via `y_mask`.
        """
        raw_preds_np, y_np, mask_np = self._collect_raw_predictions(
            data=data, batch_size=batch_size, device=device, add_feats=add_feats, pbar=pbar, collect_targets=True
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

    def reset_params(self):
        """Reset all learnable parameters (encoder, shared MLP, task heads, and uncertainty log-variances)."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.log_vars is not None:
            with torch.no_grad():
                self.log_vars.zero_()
        self._y_means = {}
        self._y_stds = {}
        self.is_fitted = False

    def save(self, filepath: str):
        """
        Save MultiTaskGNN weights, task configuration metadata, and per-task regression scalers to a `.pth` file.
        """
        payload = {
            "format": CHECKPOINT_FORMAT,
            "task_names": list(self._task_names),
            "task_configs": copy.deepcopy(self._task_configs),
            "loss_weighting": self.loss_weighting,
            "y_means": dict(self._y_means),
            "y_stds": dict(self._y_stds),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        torch.save(payload, filepath)
        return "Model saved!"

    def load(self, model_path: str, device: str = "auto"):
        """
        Load MultiTaskGNN weights and per-task regression scalers from a `.pth` file after validating task compatibility.
        """
        fit_device = get_device(device)
        checkpoint = torch.load(model_path, map_location=fit_device, weights_only=True)
        _validate_multitask_checkpoint(
            checkpoint, self._task_names, self._task_configs, self.loss_weighting
        )

        self.load_state_dict(checkpoint["state_dict"])
        self.to(fit_device)
        self._y_means = {k: float(v) for k, v in checkpoint.get("y_means", {}).items()}
        self._y_stds = {k: float(v) for k, v in checkpoint.get("y_stds", {}).items()}
        self.is_fitted = True
        self.eval()
        print("Model loaded!")
        return self


if __name__ == "__main__":
    import doctest

    doctest.testmod()
