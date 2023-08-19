import torch
from torch import optim

from babyDiffusion import ConvVAE, vae_loss, train, test
from example.dataset import ImgCapDataset


def check_cvae(model: "ConvVAE", dataset: "ImgCapDataset"):
    import matplotlib.pyplot as plt

    ds.mode("both")
    IMG, CAP = dataset[torch.randint(0, len(ds), (1,)).item()]
    RECON_IMG = model.inference(IMG[None])[0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(IMG.permute(1, 2, 0).detach().numpy())
    ax[1].imshow(RECON_IMG.permute(1, 2, 0).detach().numpy())
    fig.suptitle(CAP)
    plt.show()


if __name__ == '__main__':
    cvae = ConvVAE(3, 1 << 4, dropout=False, batch_norm=False, layers=4)
    ds = ImgCapDataset("example/shapesdata", img_transform=lambda x: x.float() / 255)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

    ds.mode("images")
    train(cvae, ds, optimizer, vae_loss, ne=1 << 4, bs=1 << 3)

    check_cvae(cvae, ds)
