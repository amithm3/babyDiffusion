import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from encoderdecoder import ImgCapDataset


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
            "<cap>": 4,  # capital
            " ": 5,
            ",": 6,
            ".": 7,
            "!": 8,
            "?": 9,
            ";": 10,
            ":": 11,
            "-": 12,
        }
        self._vre = "|".join(w if w not in ("?", ".") else f"\\{w}" for w in self._w2i)

        for word in self(texts):
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
                return torch.tensor([self[word] for word in token])
            elif all(isinstance(word, int) for word in token):
                return [self[word] for word in token]
            else:
                raise TypeError(f"Expected str / int or its iterable")
        else:
            raise TypeError(f"Expected str / int or its iterable")


class LSTMEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 p: float = 0.2):
        super().__init__()

        self._dropout = nn.Dropout(p)
        self._embedding = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=p)

    def forward(self, x):
        embedded = self._dropout(self._embedding(x))
        _, (hidden, cell) = self._lstm(embedded)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 p: float = 0.2):
        super().__init__()

        self._dropout = nn.Dropout(p)
        self._embedding = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=p)
        self._fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedded = self._dropout(self._embedding(x))
        output, (hidden, cell) = self._lstm(embedded, (hidden, cell))
        prediction = self._fc(output.squeeze(0))
        return prediction, hidden, cell


class LSTMAutoEncoder(nn.Module):
    def __init__(self,
                 tokenizer: "Tokenizer",
                 embedding_dim: int = 128,
                 hidden_dim: int = 1024,
                 num_layers: int = 2,
                 p: float = 0.2):
        super().__init__()

        self.tokenizer = tokenizer
        self._encoder = LSTMEncoder(len(tokenizer), embedding_dim, hidden_dim, num_layers, p)
        self._decoder = LSTMDecoder(len(tokenizer), embedding_dim, hidden_dim, num_layers, p)

    def forward(self, source, tfr=0.5):
        batch_size = source.shape[1]
        target_len = source.shape[0]
        target_vocab_size = len(self.tokenizer)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        hidden, cell = self._encoder(source)

        x = source[0]
        for t in range(1, target_len):
            output, hidden, cell = self._decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = source[t] if torch.rand(1).item() < tfr else best_guess

        return outputs


def test(
        lae: "LSTMAutoEncoder",
        text: str,
):
    tok = lae.tokenizer
    tokens = tok[tok(text)]
    tokens = tokens.unsqueeze(1)
    outputs = lae(tokens, tfr=0)
    outputs = outputs.argmax(2).squeeze(1)
    return "".join(tok[map(int, outputs)])


def train(
        lae: "LSTMAutoEncoder",
        ds: "ImgCapDataset",
        optimizer,
        criterion,
        ne: int = 10, bs: int = 32,
):
    tok = lae.tokenizer
    for epoch in (prog := tqdm(range(ne))):
        dataset.mode("captions")
        los = 0
        i = 0
        for batch, captions in enumerate(DataLoader(ds, batch_size=bs, shuffle=True)):
            tokens = pad_sequence([tok[tok(cap)] for cap in captions], padding_value=tok["<pad>"])
            outputs = lae(tokens)

            outputs = outputs[1:].reshape(-1, outputs.shape[2])
            targets = tokens[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            los += loss.item()
            i += 1
        prog.set_description(f"Epoch [{epoch + 1}/{ne}] Loss: {los / i:.4f}")


if __name__ == '__main__':
    dataset = ImgCapDataset("shapesdata")
    tok = Tokenizer(dataset.captions)
    lae = LSTMAutoEncoder(tok)

    optimizer = optim.Adam(lae.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tok["<pad>"])

    train(lae, dataset, optimizer, criterion, ne=10, bs=32)
