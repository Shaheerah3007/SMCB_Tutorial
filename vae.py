import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder,
        decoder,
        criterion,
        train_data,
        val_data,
        beta=1,
        learning_rate=1e-4,
        batch_size=64,
        epochs=50,
        shuffle=True,
        thresh=1e-4
    ):
        super(VAE, self).__init__()

        # Incoming components
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

        # Train settings
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.thresh = thresh
        self.beta = beta

        # Add μ and logσ² layers on top of encoder output
        encoder_output_dim = list(self.encoder.children())[-2].out_features
        self.mu = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar = nn.Linear(encoder_output_dim, latent_dim)

    # --------- VAE Core Methods ---------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    # --------- Training Function ---------
    def fit(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        train_loader = DataLoader(
            TensorDataset(torch.Tensor(self.train_data)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            TensorDataset(torch.Tensor(self.val_data)),
            batch_size=self.batch_size,
            shuffle=False,
        )

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        prev_loss = float("inf")
        stop_count = 0

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0

            for x_batch, in train_loader:
                x_batch = x_batch.to(device)

                optimizer.zero_grad()
                reconstructed, mu, logvar = self(x_batch)

                recon_loss = self.criterion(reconstructed, x_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + self.beta * kl_loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # -------- validation --------
            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x_batch, in val_loader:
                    x_batch = x_batch.to(device)

                    reconstructed, mu, logvar = self(x_batch)

                    recon_loss = self.criterion(reconstructed, x_batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    loss = recon_loss + self.beta * kl_loss
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Early stopping
            if abs(val_loss - prev_loss) < self.thresh:
                print("Early stopping (converged)")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= 2:
                    print("Early stopping (no improvement)")
                    break

            prev_loss = val_loss

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f}"
            )
