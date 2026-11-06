import tiktoken

class BPETokenizer:

    def __init__(self, encoding_name: str= "gpt2"):

        self.enc= tiktoken.get_encoding(encoding_name)
        self.vocab_size= self.enc.n_vocab

    def encode(self, s: str) -> list[int]:
        return self.enc.encode(s, allowed_special= {"<|endoftext|>"})

    def decode(self, l: list[int]) -> str:
        return self.enc.decode(l)