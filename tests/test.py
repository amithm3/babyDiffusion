import torch
from torch import optim

from babyDiffusion import ConvVAE, VAELoss, train, test
from examples.dataset import ImgCapDataset


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
    cvae = ConvVAE(3, 1 << 5, dropout=False, batch_norm=True, layers=3)
    ds = ImgCapDataset("examples/shapesdata", img_transform=lambda x: x.float() / 255)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    loss_fnc = VAELoss()

    ds.mode("images")
    train(cvae, ds, optimizer, loss_fnc, ne=1 << 5, bs=1 << 1)

    check_cvae(cvae, ds)
