from torch import nn

from .autoencoder import VariationalAutoEncoder


class ConvVAE(VariationalAutoEncoder):
    def _build_encoder(self) -> "nn.Module":
        layers = [nn.Conv2d(self._in_chn, 32, 4, 2, 1)]
        if self._batch_norm: layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        if self._dropout: layers.append(nn.Dropout2d(self._dropout))
        for i in range(self._layers - 1):
            layers.append(nn.Conv2d(32 * 2 ** i, 64 * 2 ** i, 4, 2, 1))
            if self._batch_norm: layers.append(nn.BatchNorm2d(64 * 2 ** i))
            layers.append(nn.ReLU())
            if self._dropout: layers.append(nn.Dropout2d(self._dropout))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> "nn.Module":
        layers = []
        for i in range(self._layers - 1):
            layers.append(nn.ConvTranspose2d(64 * 2 ** (self._layers - i - 2), 32 * 2 ** (self._layers - i - 2), 4, 2, 1))
            if self._batch_norm: layers.append(nn.BatchNorm2d(32 * 2 ** (self._layers - i - 2)))
            layers.append(nn.ReLU())
            if self._dropout: layers.append(nn.Dropout2d(self._dropout))
        layers.append(nn.ConvTranspose2d(32, self._in_chn, 4, 2, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
