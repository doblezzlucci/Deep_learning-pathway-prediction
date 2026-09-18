import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

class SSLFineTuneModel(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.classifier(z)
        return out
    
def finetune_single_run(
        encoder, X_train, y_train, X_test, y_test,
        latent_dim, num_classes,
        freeze_encoder=True,
        epochs=20,
        batch_size=32,
        lr=1e-3):

    # Copy encoder to avoid overwriting during multiple runs
    encoder_ft = encoder

    # Freeze or unfreeze encoder
    if freeze_encoder:
        for p in encoder_ft.parameters():
            p.requires_grad = False
    else:
        for p in encoder_ft.parameters():
            p.requires_grad = True

    # Build full model
    model = SSLFineTuneModel(
        encoder_ft, latent_dim=latent_dim, num_classes=num_classes
    )

    # Optimizer ONLY trains trainable params
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    # Training
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Test accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        acc = accuracy_score(y_test.cpu(), preds.cpu())

    return acc, model

def run_multiproportion_finetuning_unfrozen(
        encoder,
        X_full, y_full,
        X_test, y_test,
        latent_dim,
        num_classes,
        save_dir="finetune_ckpts",
        proportions=np.linspace(0.1, 1.0, 10),
        runs=5,
        freeze_encoder=False):

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    X_np, y_np = X_full.numpy(), y_full.numpy()

    for p in proportions:
        results[p] = []
        print(f"\n===== Fine-tuning on {int(p*100)}% data =====")

        # ------------------------------------
        # CASE 1 → p == 1.0 (full dataset)
        # ------------------------------------
        if p == 1.0:
            print("  Using FULL dataset — no StratifiedSplit")

            X_sub = X_full
            y_sub = y_full

            for run_id in range(1, runs + 1):

                acc, model = finetune_single_run(
                    encoder=encoder,
                    X_train=X_sub, y_train=y_sub,
                    X_test=X_test, y_test=y_test,
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                    freeze_encoder=freeze_encoder,
                    epochs=20,
                    batch_size=32,
                    lr=1e-3
                )

                results[p].append(acc)
                print(f" Run {run_id}: acc={acc:.4f}")

                # Save checkpoint
                ckpt = os.path.join(save_dir,
                    f"finetune_p100_run{run_id}.pth")
                torch.save(model.state_dict(), ckpt)

            continue  # skip stratified splitter

        # ------------------------------------
        # CASE 2 → 0 < p < 1 (normal train-size)
        # ------------------------------------
        splitter = StratifiedShuffleSplit(
            n_splits=runs, train_size=p, random_state=42
        )

        for run_id, (idx, _) in enumerate(splitter.split(X_np, y_np), start=1):

            # Subset data
            X_sub = X_full[idx]
            y_sub = y_full[idx]

            # Train single run
            acc, model = finetune_single_run(
                encoder=encoder,
                X_train=X_sub, y_train=y_sub,
                X_test=X_test, y_test=y_test,
                latent_dim=latent_dim,
                num_classes=num_classes,
                freeze_encoder=freeze_encoder,
                epochs=20,
                batch_size=32,
                lr=1e-3
            )

            results[p].append(acc)
            print(f" Run {run_id}: acc={acc:.4f}")

            ckpt = os.path.join(
                save_dir, f"finetune_p{int(p*100)}_run{run_id}.pth")
            torch.save(model.state_dict(), ckpt)

    # -----------------------
    # Save results CSV
    # -----------------------
    rows = []
    for p, accs in results.items():
        rows.append({
            "proportion": p,
            **{f"run{i+1}": accs[i] for i in range(len(accs))},
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "finetune_results_unfrozen.csv"), index=False)
    print("Saved →", os.path.join(save_dir, "finetune_results_unfrozen.csv"))

    return results


    
    
