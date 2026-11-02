
class CharacterTokenizer():
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding= "utf-8") as file:
            text= file.read()

        chars= sorted(list(set(text)))
        self.vocab_size= len(chars)
        self.stoi= {ch: i for i, ch in enumerate(chars)}
        self.itos= {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, l: list[int]) -> str:
        return "".join(self.itos[i] for i in l)