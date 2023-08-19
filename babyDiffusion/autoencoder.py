from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AutoEncoder(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_chn: int, latent_bits: int, **kwargs):
        super().__init__()
        self._in_chn = in_chn
        self._latent_bits = latent_bits
        self._dropout = kwargs.get("dropout", False)
        self._batch_norm = kwargs.get("batch_norm", False)
        self._device = kwargs.get("device", torch.device("cpu"))
        self._layers = kwargs.get("layers", 3)
        self._kwargs = kwargs

        self._encoder, self._decoder = self._build()

    @abstractmethod
    def _build(self) -> tuple["nn.Module", "nn.Module"]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, B: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError

    @abstractmethod
    def forward(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        raise NotImplementedError

    def inference(self, X: "torch.Tensor") -> "torch.Tensor":
        with torch.inference_mode():
            Y = self.forward(X)
            if isinstance(Y, tuple): Y = Y[0]
            return Y.detach().cpu()

    def inference_encode(self, X: "torch.Tensor") -> "torch.Tensor":
        with torch.inference_mode():
            E = self.encode(X)
            if isinstance(E, tuple): E = E[0]
            return E.detach().cpu()

    def inference_decode(self, B: "torch.Tensor") -> "torch.Tensor":
        with torch.inference_mode():
            return self.decode(B).detach().cpu()


class VariationalAutoEncoder(AutoEncoder):
    @staticmethod
    def reparameterize(mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @abstractmethod
    def _build_encoder(self) -> "nn.Module":
        raise NotImplementedError

    @abstractmethod
    def _build_decoder(self) -> "nn.Module":
        raise NotImplementedError

    def _build(self) -> tuple["nn.Module", "nn.Module"]:
        encoder = self._build_encoder()
        self._fc_mu = nn.LazyLinear(self._latent_bits)
        self._fc_logvar = nn.LazyLinear(self._latent_bits)
        self._un_fc = None
        decoder = self._build_decoder()
        self._encode, self.encode = self.encode, self._encode
        return encoder, decoder

    def _encode(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        E = self._encoder(X)

        # Lazy initialization
        if self._un_fc is None:
            self._un_fc = nn.Linear(self._latent_bits, torch.prod(torch.tensor(E.shape[1:])).item())
            flatten = nn.Flatten()
            self._encoder.append(flatten)
            un_flatten = nn.Unflatten(1, E.shape[1:])
            self._decoder.insert(0, un_flatten)
            E = flatten(E)
            self._encode, self.encode = self.encode, self._encode

        mu = self._fc_mu(E)
        logvar = self._fc_logvar(E)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        E = self._encoder(X)
        mu, logvar = self._fc_mu(E), self._fc_logvar(E)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, B: "torch.Tensor") -> "torch.Tensor":
        RECON_X = self._decoder(self._un_fc(B))
        return RECON_X

    def forward(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        z, mu, logvar = self.encode(X)
        RECON_X = self.decode(z)
        return RECON_X, mu, logvar


def vae_loss(out, X):
    RECON_X, mu, logvar = out
    recon_loss = nn.functional.mse_loss(RECON_X, X, reduction='sum')
    kl_loss = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train(
        model: "AutoEncoder",
        ds: "Dataset",
        optimizer: "optim.Optimizer",
        criterion,
        ne: int = 10, bs: int = 32,
):
    for epoch in (e_prog := tqdm(range(ne))):
        loss_sum = 0
        e_prog.set_postfix({
            "Epoch": f"{epoch + 1}/{ne}",
            "Batch": f"?/?",
            "Cost": f"?/?"
        })
        for batch, X in enumerate(DataLoader(ds, batch_size=bs, shuffle=True)):
            optimizer.zero_grad()
            loss = criterion(model(X), X)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            e_prog.set_postfix({
                "Epoch": f"{epoch + 1}/{ne}",
                "Batch": f"{batch + 1}/{len(ds) // bs}",
                "Cost": f"{loss_sum / (batch + 1):.4f}"
            })


def test(
        model: "AutoEncoder",
        ds: "Dataset",
        criterion,
        bs: int = 32,
):
    loss_sum = 0
    for batch, (X, Y) in enumerate(b_prog := tqdm(DataLoader(ds, batch_size=bs, shuffle=True))):
        loss = criterion(model(X), Y)
        loss_sum += loss.item()
        b_prog.set_postfix({
            "Batch": f"{batch + 1}/{len(ds) // bs}",
            "Cost": f"{loss_sum / (batch + 1):.4f}"
        })
    return loss_sum / (batch + 1)
