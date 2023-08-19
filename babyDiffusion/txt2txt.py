import re

import torch


class Tokenizer:
    def __init__(self, texts):
        self._w2i = {
            # ===TAGS===
            "<unk>": 0,  # unknown
            "<pad>": 1,  # padding
            "<sos>": 2,  # start of sentence
            "<eos>": 3,  # end of sentence
            "<cap>": 4,  # capital

            # ===SPACED DELIMITERS===
            ",": 5,
            ".": 6,
            "!": 7,
            "?": 8,
            ";": 9,
            ":": 10,

            # ===NON-SPACED DELIMITERS===
            " ": 12,
            "-": 13,

            # === SPECIAL CHARACTERS ===
            "\n": 13,
            "\t": 14,
            "\r": 15,
        }

        for word in self(texts):
            if word not in self._w2i: self._w2i[word.lower()] = len(self._w2i)

        self._i2w = {i: w for w, i in self._w2i.items()}

    def __call__(self, text):
        if isinstance(text, str):
            text = f"<sos> {text.strip()} <eos>"
            text = re.sub(f"([,.!?;: -])([A-Z])", r"\1<cap> \2", text).lower()
            tokens = re.split(f"([,.!?;: -])", text)
            return [t for t in tokens if t not in ("", " ")]
        elif (isinstance(text, tuple) or isinstance(text, list)) and all(isinstance(t, str) for t in text):
            text = [t.capitalize() if text[i] == "<cap>" else t for i, t in enumerate(text[1:-1]) if t != "<cap>"]
            text = " ".join(text)
            text = re.sub(f" +([,.!?;:])", r"\1", text)
            return re.sub(f" +([ -]) +", r"\1", text)

    def __len__(self):
        return len(self._w2i)

    def __getitem__(self, token):
        if isinstance(token, str):
            return self._w2i.get(token.lower(), self._w2i["<unk>"])
        elif isinstance(token, int):
            return self._i2w.get(token, "<unk>")
        elif isinstance(token, torch.Tensor):
            return [self[word] for word in map(int, token)]
        elif isinstance(token, tuple) or isinstance(token, list):
            if all(isinstance(word, str) for word in token):
                return torch.tensor([self[word] for word in token])
            elif all(isinstance(word, int) for word in token):
                return [self[word] for word in token]

        raise TypeError(f"Expected str / int or its tuple / list / torch.Tensor")
