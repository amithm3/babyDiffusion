import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 p: float = 0.2,
                 bid: bool = False):
        super().__init__()

        self._dropout = nn.Dropout(p)
        self._embedding = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=p, bidirectional=bid)

    def forward(self, x):
        embedded = self._dropout(self._embedding(x))
        _, (hidden, cell) = self._lstm(embedded)
        return hidden, cell


class BiLSTMDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 p: float = 0.2,
                 bid: bool = False):
        super().__init__()

        self._dropout = nn.Dropout(p)
        self._embedding = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=p, bidirectional=bid)
        self._fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedded = self._dropout(self._embedding(x))
        output, (hidden, cell) = self._lstm(embedded, (hidden, cell))
        prediction = self._fc(output.squeeze(0))
        return prediction, hidden, cell


class BiLSTMAutoEncoder(nn.Module):
    def __init__(self,
                 tokenizer: "Tokenizer",
                 embedding_dim: int = 128,
                 hidden_dim: int = 1024,
                 num_layers: int = 2,
                 p: float = 0.2,
                 bid: bool = False):
        super().__init__()

        self.tokenizer = tokenizer
        self._encoder = BiLSTMEncoder(len(tokenizer), embedding_dim, hidden_dim, num_layers, p, bid)
        self._decoder = BiLSTMDecoder(len(tokenizer), embedding_dim, hidden_dim, num_layers, p, bid)

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
