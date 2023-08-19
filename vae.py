import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from encoderdecoder import ImgCapDataset


# Define the Conv-VAE architecture
class ConvVAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# Initialize the Conv-VAE model with custom parameters
conv_vae = ConvVAE(3, 32)

# Define optimizer
optimizer = optim.Adam(conv_vae.parameters(), lr=0.001)

# dataset
dataset = ImgCapDataset("shapesdata", img_transform=lambda x: x.float() / 255)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# Training loop
def train_conv_vae(data_loader, num_epochs):
    conv_vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = conv_vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader.dataset)}")


# Train the Conv-VAE
train_conv_vae(train_loader, num_epochs=10)


def display(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()
