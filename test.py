import torch
import matplotlib.pyplot as plt

from encoderdecoder import ImgCapDataset, ConvAutoEncoder, Tokenizer


def genai(cae, tse, prompt: str):
    with torch.no_grad():
        z, _ = tse(prompt)
        img = cae.decode(z[0] * 10)
        display(img[0])
        print(prompt)


def display(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()


def checkai(cae, ds: "ImgCapDataset", idx):
    import matplotlib.pyplot as plt

    ds.mode("both")
    with torch.no_grad():
        X, cap = ds[idx]
        z = cae.bits(X[None])
        img = cae.decode(z)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(X.permute(1, 2, 0).detach().numpy())
        ax[1].imshow(img[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        print(cap)


def sample(cae, eBits):
    with torch.no_grad():
        z = torch.randn(1, eBits)
        img = cae.decode(z)

        plt.imshow(img[0].permute(1, 2, 0).detach().numpy())
        plt.show()


if __name__ == '__main__':
    eBits = 1 << 5
    dataset = ImgCapDataset("shapesdata", img_transform=lambda x: x.float() / 255)

    cae = ConvAutoEncoder(3, eBits)
    tok = Tokenizer(dataset.captions)
    # tse = TextSequenceEncoder(100, eBits, tok)

    cae.fit(dataset, 1 << 6, 1)
    # tse.fit_full(cae, dataset, 25, 1)
