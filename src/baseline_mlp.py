
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


# -----------------------------
# Config
# -----------------------------
@dataclass
class BaselineConfig:
    proportions: np.ndarray = np.linspace(0.1, 1.0, 10)
    runs: int = 5
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    dropout_rate: float = 0.1
    seed: int = 42
    save_checkpoints: bool = True


# -----------------------------
# Data utilities
# -----------------------------
def prepare_tensors(
    df_x_tune: pd.DataFrame,
    df_y_tune: pd.Series | pd.DataFrame,
    df_x_test: pd.DataFrame,
    df_y_test: pd.Series | pd.DataFrame,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
   
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # y may be Series or 1-col DataFrame; ensure 1D
    y_full_np = np.asarray(df_y_tune).squeeze()
    y_test_np = np.asarray(df_y_test).squeeze()

    X_full = torch.tensor(df_x_tune.values, dtype=torch.float32, device=device)
    y_full = torch.tensor(y_full_np, dtype=torch.long, device=device)

    X_test = torch.tensor(df_x_test.values, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test_np, dtype=torch.long, device=device)

    num_classes = int(len(torch.unique(y_full)))

    return X_full, y_full, X_test, y_test, num_classes, device


# -----------------------------
# Model
# -----------------------------
class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Train/Eval
# -----------------------------
def baseline_single_run(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.1,
) -> Tuple[float, nn.Module]:
    """
    Train one baseline MLP and return (test_accuracy, model).
    """
    model = BaselineMLP(input_dim=X_train.shape[1], num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(X_train.device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        pred_labels = preds.argmax(dim=1)

    acc = accuracy_score(
        y_test.detach().cpu().numpy(),
        pred_labels.detach().cpu().numpy()
    )

    return float(acc), model


# -----------------------------
# Main experiment
# -----------------------------
def run_baseline_experiment(
    df_x_tune: pd.DataFrame,
    df_y_tune: pd.Series | pd.DataFrame,
    df_x_test: pd.DataFrame,
    df_y_test: pd.Series | pd.DataFrame,
    output_dir: str,
    cfg: BaselineConfig = BaselineConfig(),
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
   
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility (torch + numpy)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Data
    X_full, y_full, X_test, y_test, num_classes, device = prepare_tensors(
        df_x_tune, df_y_tune, df_x_test, df_y_test, device=device
    )

    print("Device:", device)
    print("Train shape:", tuple(X_full.shape))
    print("Test shape:", tuple(X_test.shape))
    print("Num classes:", num_classes)

    # Setup paths
    ckpt_dir = os.path.join(output_dir, "checkpoints_baseline")
    if cfg.save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)

    results: Dict[float, List[float]] = {}
    best_acc = 0.0
    best_state = None

    # For sklearn splitting we need numpy arrays on CPU
    X_np = X_full.detach().cpu().numpy()
    y_np = y_full.detach().cpu().numpy()

    for p in cfg.proportions:
        p = float(p)
        results[p] = []
        print(f"\n========== Proportion {int(p*100)}% ==========")

        if p == 1.0:
            # Use FULL dataset; repeat runs (different init each time)
            for run_id in range(1, cfg.runs + 1):
                acc, model = baseline_single_run(
                    X_full, y_full,
                    X_test, y_test,
                    num_classes=num_classes,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    dropout_rate=cfg.dropout_rate,
                )
                results[p].append(acc)
                print(f" Run {run_id}: acc = {acc:.4f}")

                if cfg.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"baseline_p100_run{run_id}.pth"))

                if acc > best_acc:
                    best_acc = acc
                    best_state = model.state_dict()

        else:
            splitter = StratifiedShuffleSplit(
                n_splits=cfg.runs, train_size=p, random_state=cfg.seed
            )

            for run_id, (idx, _) in enumerate(splitter.split(X_np, y_np), start=1):
                X_sub = X_full[idx]
                y_sub = y_full[idx]

                acc, model = baseline_single_run(
                    X_sub, y_sub,
                    X_test, y_test,
                    num_classes=num_classes,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    dropout_rate=cfg.dropout_rate,
                )
                results[p].append(acc)
                print(f" Run {run_id}: acc = {acc:.4f}")

                if cfg.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"baseline_p{int(p*100)}_run{run_id}.pth"))

                if acc > best_acc:
                    best_acc = acc
                    best_state = model.state_dict()

        print(f"Mean acc @ {int(p*100)}% = {np.mean(results[p]):.4f}")

    # Save best model (optional but handy)
    if cfg.save_checkpoints and best_state is not None:
        torch.save(best_state, os.path.join(ckpt_dir, "baseline_best.pth"))

    # Build results DF
    rows = []
    for p, accs in results.items():
        row = {
            "proportion": p,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        }
        for i, a in enumerate(accs, start=1):
            row[f"run{i}"] = float(a)
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("proportion")
    out_csv = os.path.join(output_dir, "baseline_results.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("Best model accuracy:", best_acc)

    return results_df, best_acc
