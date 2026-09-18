import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

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


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class PathwayAutoencoder(nn.Module):
    def __init__(self, input_dim, pathway_dim, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, pathway_dim)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat, z

#training loop 

def pretrain_autoencoder(
    X,            # Tensor: gene expression (N x G)
    Y_pathway,    # Tensor: pathway profiles (N x P)
    input_dim,
    pathway_dim,
    latent_dim=256,
    lr=1e-3,
    epochs=50,
    batch_size=64
):
    model = PathwayAutoencoder(input_dim, pathway_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y_pathway)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_hat, _ = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss / len(X):.5f}")

    return model




def save_pretrained_model(encoder, config, save_dir):
  
    os.makedirs(save_dir, exist_ok=True)
     
    # Save weights
    torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))

    # Save JSON config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"[OK] Saved encoder + config inside {save_dir}")



def load_pretrained_encoder(save_dir):
    # Load config
    with open(os.path.join(save_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Recreate encoder
    encoder = Encoder(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"]
    )
    
    # Load weights
    encoder.load_state_dict(
        torch.load(os.path.join(save_dir, "encoder.pt"), map_location="cpu")
    )

    print(f"[OK] Loaded pretrained encoder from {save_dir}")
    return encoder, config



