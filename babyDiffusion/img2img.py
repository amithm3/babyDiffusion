from typing import Union

import torch
from torch import nn

from .autoencoder import VariationalAutoEncoder


class ConvVAEEncoder(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, batch_norm: bool = False) -> None:
        super().__init__()
        self._in_chn = in_chn
        self._out_chn = out_chn
        self._batch_norm = batch_norm

        self._conv1 = nn.Conv2d(self._in_chn, self._out_chn, 4, 2, 1)
        self._conv2 = nn.Conv2d(self._out_chn, self._out_chn, 3, 1, 1)
        self._batch_norm = nn.BatchNorm2d(self._out_chn) if batch_norm else None
        self._leaky_relu = nn.LeakyReLU()

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        X = self._conv1(X)
        X = self._conv2(X)
        if self._batch_norm is not None: X = self._batch_norm(X)
        X = self._leaky_relu(X)
        return X


class ConvVAEDecoder(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, batch_norm: bool = False, sigmoid: bool = False) -> None:
        super().__init__()
        self._in_chn = in_chn
        self._out_chn = out_chn
        self._batch_norm = batch_norm

        self._conv1 = nn.ConvTranspose2d(self._in_chn, self._out_chn, 4, 2, 1)
        self._conv2 = nn.ConvTranspose2d(self._out_chn, self._out_chn, 3, 1, 1)
        self._batch_norm = nn.BatchNorm2d(self._out_chn) if batch_norm else None
        self._act = nn.LeakyReLU() if not sigmoid else nn.Sigmoid()

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        X = self._conv1(X)
        X = self._conv2(X)
        if self._batch_norm is not None: X = self._batch_norm(X)
        X = self._act(X)
        return X


class ConvVAE(VariationalAutoEncoder):
    def _build(self) -> tuple["nn.Module", "nn.Module"]:
        # set lazy initialization
        self._encode, self.encode = self.encode, self._encode
        return super()._build()

    def _encode(self, X: "torch.Tensor") -> Union[tuple["torch.Tensor", ...], "torch.Tensor"]:
        """
        Lazy Initialization of Flatten and Linear Layers
        :param X:
        :return:
        """
        E = self._encoder(X)

        flatten = nn.Flatten()
        linear = nn.Linear(self._latent_bits, torch.prod(torch.tensor(E.shape[1:])).item(), device=self._device)
        un_flatten = nn.Unflatten(1, E.shape[1:])

        self._encoder.append(flatten)
        self._un_fc.insert(0, linear)
        self._decoder.insert(0, un_flatten)

        self._encode, self.encode = self.encode, self._encode

        return super().encode(X)

    def _build_encoder(self) -> "nn.Module":
        dims = self._in_chn, *self._layers
        layers = []
        for i, dim in enumerate(dims[1:]):
            layers.append(ConvVAEEncoder(dims[i], dim, self._batch_norm))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> "nn.Module":
        dims = *self._layers[::-1], self._in_chn
        layers = []
        for i, dim in enumerate(dims[1:]):
            layers.append(ConvVAEDecoder(dims[i], dim, self._batch_norm, sigmoid=i == len(dims) - 2))
        return nn.Sequential(*layers)
