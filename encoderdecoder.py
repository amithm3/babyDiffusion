import json
import os
import re
from typing import Literal, Union

import torch
from tqdm import tqdm as tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class ImgCapDataset(Dataset):
    @property
    def captions(self):
        return "; ".join(self._captions.values())

    def __init__(self, DIR: str, img_transform=None, cap_transform=None):
        super().__init__()

        self.DIR = DIR
        self._files = os.listdir(DIR)
        self._files.remove("captions.json")
        self._files.remove("captions.py")
        self._img_transform = img_transform
        self._cap_transform = cap_transform
        self._mode = "both"

        with open(os.path.join(self.DIR, "captions.json"), "r") as f:
            self._captions = json.load(f)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_datapoint(self._files[idx])
        elif isinstance(idx, slice):
            return [self.get_datapoint(file) for file in self._files[idx]]
        elif isinstance(idx, list):
            return [self.get_datapoint(self._files[i]) for i in idx]
        else:
            raise TypeError(f"Indexing type {type(idx)} not supported")

    def mode(self, mode: Literal["images", "captions", "both"]):
        if mode not in ["images", "captions", "both"]:
            raise ValueError(f"Mode {mode} not supported")
        self._mode = mode
        return self

    def get_datapoint(self, file: str) -> Union[tuple["torch.Tensor", str], "torch.Tensor", str]:
        if self._mode == "images":
            img = read_image(os.path.join(self.DIR, file))
            if self._img_transform: img = self._img_transform(img)
            return img
        elif self._mode == "captions":
            cap = self._captions[file] if file in self._captions else ""
            if self._cap_transform: cap = self._cap_transform(cap)
            return cap
        elif self._mode == "both":
            img = read_image(os.path.join(self.DIR, file))
            if self._img_transform: img = self._img_transform(img)
            cap = self._captions[file] if file in self._captions else ""
            if self._cap_transform: cap = self._cap_transform(cap)
            return img, cap


class ConvAutoEncoder(nn.Module):
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def __init__(self, in_chn: int, encode_bits: int):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(in_chn, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64 * 8 * 8, encode_bits)
        self.fc_logvar = nn.Linear(64 * 8 * 8, encode_bits)

        self._decoder = nn.Sequential(
            nn.Linear(encode_bits, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(32, in_chn, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, X):
        hidden = self._encoder(X)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def bits(self, X):
        mu, logvar = self.encode(X)
        return self.reparameterize(mu, logvar)

    def decode(self, X):
        return self._decoder(X)

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def fit(self, ds: "ImgCapDataset", ep: int = 10, mb: int = 1, *, optimizer=None):
        if optimizer is None: optimizer = optim.Adam(self.parameters(), lr=0.001)

        ds.mode("images")
        for epoch in (prog := tqdm(range(ep))):
            los = 0
            i = 0
            for batch, X in enumerate(DataLoader(ds, batch_size=mb, shuffle=True)):
                optimizer.zero_grad()
                Y, mu, logvar = self(X)
                loss = vae_loss(Y, X, mu, logvar)
                loss.backward()
                optimizer.step()
                los += loss.item()
                i += 1
            prog.set_description(f"Epoch: {epoch} Loss: {los / i}")


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def isiter(foo):
    try:
        return tuple(foo)
    except TypeError:
        return False


class Tokenizer:
    def __init__(self, texts):
        self._w2i = {
            "<unk>": 0,  # unknown
            "<pad>": 1,  # padding
            "<sos>": 2,  # start of sentence
            "<eos>": 3,  # end of sentence
            "<cap>": 6,  # capital
            " ": 7,
            ",": 8,
            ".": 9,
            "!": 10,
            "?": 11,
            ";": 12,
            ":": 13,
        }
        self._vre = "|".join(w if w not in ("?", ".") else f"\\{w}" for w in self._w2i)

        for word in re.split(f"{self._vre}", texts.strip()):
            if word not in self._w2i: self._w2i[word.lower()] = len(self._w2i)

        self._i2w = {i: w for w, i in self._w2i.items()}

    def __call__(self, text):
        text = f"<sos>{text.strip()}<eos>"
        text = re.sub(f"({self._vre})([A-Z])", r"\1<cap>\2", text).lower()
        tokens = re.split(f"({self._vre})", text)
        return [t for t in tokens if t]

    def __len__(self):
        return len(self._w2i)

    def __getitem__(self, token):
        if isinstance(token, str):
            return self._w2i.get(token.lower(), self._w2i["<unk>"])
        elif isinstance(token, int):
            return self._i2w.get(token, "<unk>")
        elif token := isiter(token):
            if all(isinstance(word, str) for word in token):
                return [self[word] for word in token]
            elif all(isinstance(word, int) for word in token):
                return [self[word] for word in token]
            else:
                raise TypeError(f"Expected str / int or its iterable")
        else:
            raise TypeError(f"Expected str / int or its iterable")
