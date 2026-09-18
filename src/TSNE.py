import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # optional but recommended

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
def TSNE_pretrained(save_dir,dataset_name):
  SAVE_DIR=os.path.join(save_dir,'ssl_pretrained' )# folder containing encoder.pt + config.json
  encoder, config = load_pretrained_encoder(SAVE_DIR)
  X=pd.read_csv(os.path.join(save_dir,dataset_name))  # shape (N, G)
  X = X.to_numpy(dtype=np.float32)
#X = np.load("X.npy")  # shape (N, G)
  assert X.shape[1] == config["input_dim"], f"X has {X.shape[1]} features but encoder expects {config['input_dim']}"

  X_t = torch.tensor(X, dtype=torch.float32)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  encoder = encoder.to(device)
  X_t = X_t.to(device)

# ----------------------------
# 3) Encode -> latent vectors
# ----------------------------
  BATCH = 256
  Z_list = []

  with torch.no_grad():
      for i in range(0, X_t.shape[0], BATCH):
          z = encoder(X_t[i:i+BATCH])
          Z_list.append(z.detach().cpu())

  Z = torch.cat(Z_list, dim=0).numpy()  # (N, latent_dim)
  print("Latent Z shape:", Z.shape)

# ----------------------------
# 4) t-SNE (optionally PCA first)
# ----------------------------
# PCA to 50 dims usually makes t-SNE faster + more stable
  Z_pca = PCA(n_components=min(50, Z.shape[1]), random_state=42).fit_transform(Z)

  tsne = TSNE(
      n_components=2,
      perplexity=30,       # adjust based on N (try 5–50)
      learning_rate="auto",
      init="pca",
      random_state=42
  )
  Z_tsne = tsne.fit_transform(Z_pca)  # (N, 2)

# ----------------------------
# 5) Plot
# ----------------------------
  plt.figure(figsize=(7, 6))
  plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=10)
  plt.title("t-SNE of pretrained encoder latent space")
  plt.xlabel("t-SNE-1")
  plt.ylabel("t-SNE-2")
  plt.tight_layout()
  plt.show()
  plt.savefig(os.path.join(save_dir,"tsne_plot_pretrained.png"), dpi=30)