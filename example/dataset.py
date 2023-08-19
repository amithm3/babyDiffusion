import json
import os
from typing import Literal, Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImgCapDataset(Dataset):
    def __init__(self, DIR: str, img_transform=None, cap_transform=None):
        super().__init__()

        self.DIR = DIR
        self._files = os.listdir(DIR)
        self._files.remove("captions.json")
        self._img_transform = img_transform
        self._cap_transform = cap_transform
        self._mode = "both"

        with open(os.path.join(self.DIR, "captions.json"), "r") as f:
            self._captions = json.load(f)

    @property
    def captions(self):
        return "\n".join(self._captions.values())

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
