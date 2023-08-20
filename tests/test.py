import os
import pickle
from typing import Callable

import torch
from torch import optim
from torch.utils.data import Dataset

from babyDiffusion import ConvVAE, VAELoss, train, test
from examples.dataset import ImgCapDataset


def check_cvae(model: "ConvVAE", dataset: "Dataset"):
    import matplotlib.pyplot as plt

    IMG = dataset[torch.randint(0, len(ds), (1,)).item()]
    RECON_IMG = model.inference(IMG[None])[0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(IMG.permute(1, 2, 0).detach().cpu().numpy())
    ax[1].imshow(RECON_IMG.permute(1, 2, 0).numpy())
    plt.show()


def sample_cvae(model: "ConvVAE", latent_bits: int):
    import matplotlib.pyplot as plt

    with torch.no_grad():
        z = torch.randn(1, latent_bits)
        img = model.decode(z)

        plt.imshow(img[0].permute(1, 2, 0).detach().numpy())
        plt.show()


class CIFAR10(Dataset):
    base_folder = "cifar-10-batches-py"
    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    def __init__(
            self,
            root: str,
            transform: Callable = None,
            target_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data = []

        # now load the picked numpy arrays
        for file_name in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(torch.tensor(entry["data"]))

        self.data = torch.vstack(self.data).reshape((-1, 3, 32, 32))

    def __getitem__(self, index: int) -> "torch.Tensor":
        img = self.data[index]
        if self.transform is not None: img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = ConvVAE(3, 1 << 6, batch_norm=True, layers=3, device=device).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    loss_fnc = VAELoss()
    # ds = CIFAR10("examples/cifar10", transform=lambda x: x.float().to(device) / 255)
    ds = ImgCapDataset("examples/shapesdata", img_transform=lambda x: x.float().to(device) / 255)
    ds.mode("images")

    cvae(ds[0][None])
    cvae.inference(ds[0][None])
    print(cvae)

    train(cvae, ds, optimizer, loss_fnc, ne=1 << 7, bs=1 << 2)
    test(cvae, ds, loss_fnc, bs=1 << 10)
