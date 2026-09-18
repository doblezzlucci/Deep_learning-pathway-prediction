import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # optional but recommended
from sklearn.preprocessing import StandardScaler
# ----------------------------
# 1) Your Encoder definition (MUST match training)
# ----------------------------
import torch.nn as nn

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_pretrained_encoder(save_dir):
    with open(os.path.join(save_dir,"config.json"), "r") as f:
        config = json.load(f)

    encoder = Encoder(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"]
    )

    encoder.load_state_dict(
        torch.load(os.path.join(save_dir, "encoder.pt"), map_location="cpu")
    )
    encoder.eval()
    return encoder, config

# ----------------------------
# 2) Load encoder + your data
# ----------------------------



def TSNE_pretrained(save_dir, dataset_name, label_col):
    SAVE_DIR = os.path.join(save_dir, "ssl_pretrained")
    encoder, config = load_pretrained_encoder(SAVE_DIR)
    if type(dataset_name)==str : 
      df = pd.read_csv(os.path.join(save_dir, dataset_name))
    else:
      df=dataset_name
    # --- OPTIONAL: keep labels for coloring
    labels = None
    if label_col is not None and label_col in df.columns:
        labels = df[label_col].astype(str).values

    # --- Drop common metadata columns if they exist
    drop_cols = [c for c in ["caseID", "cancer_type"] if c in df.columns]
    X_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns (avoids object dtype issues)
    X_df = X_df.select_dtypes(include=[np.number])

    X = X_df.to_numpy(dtype=np.float32)

    assert X.shape[1] == config["input_dim"], (
        f"X has {X.shape[1]} features but encoder expects {config['input_dim']}. "
        f"Check that you dropped non-gene columns and kept the same gene set/order as training."
    )

    # --- Encode
    X_t = torch.from_numpy(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    X_t = X_t.to(device)

    BATCH = 256
    Z_list = []
    encoder.eval()

    with torch.no_grad():
        for i in range(0, X_t.shape[0], BATCH):
            z = encoder(X_t[i:i+BATCH])
            Z_list.append(z.detach().cpu())

    Z = torch.cat(Z_list, dim=0).numpy()
    print("Latent Z shape:", Z.shape)

    # --- PCA -> (optional) Standardize -> t-SNE
    Z_pca = PCA(n_components=min(50, Z.shape[1]), random_state=42).fit_transform(Z)
    Z_pca = StandardScaler().fit_transform(Z_pca)

    n = Z_pca.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))  # safe default if n is small

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    Z_tsne = tsne.fit_transform(Z_pca)

    # --- Plot
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    if labels is None:
        ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=6, alpha=0.7, rasterized=True)
    else:
        # Color by label (categorical)
        uniq = pd.unique(labels)
        for lab in uniq:
            idx = labels == lab
            ax.scatter(Z_tsne[idx, 0], Z_tsne[idx, 1], s=6, alpha=0.7, label=lab, rasterized=True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    ax.set_title("t-SNE of pretrained encoder latent space")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_png = os.path.join(save_dir, "tsne_plot_pretrained.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")  # save BEFORE show
    plt.show()

    print("Saved:", out_png)
    return Z_tsne
