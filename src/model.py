import json
import torch
import torch.nn as nn
import torch.nn.functional as F


with open("configs/default_config.json", "r") as f:
    config= json.load(f)


class TransformerBlock(nn.Module):
    def __init__(self, N_EMBD: int, N_HEAD: int, dropout: float):
        super().__init__()
        self.layer= nn.TransformerEncoderLayer(d_model= N_EMBD, nhead= N_HEAD, dim_feedforward= 4 * N_EMBD, dropout= dropout, batch_first= True)

    def forward(self, x, src_mask):
        return self.layer(x, src_mask= src_mask)

class DeepLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.BLOCK_SIZE= config["block_size"]
        self.token_embedding_table= nn.Embedding(vocab_size, config["n_embd"])
        self.position_embedding_table= nn.Embedding(config["block_size"], config["n_embd"])
        self.drop= nn.Dropout(dropout)

    # multiple transformer blocks
        self.blocks= nn.ModuleList([TransformerBlock(config["n_embd"], config["n_head"], dropout) for _ in range(config["n_layer"])])
        self.ln_f= nn.LayerNorm(config["n_embd"])
        self.lm_head= nn.Linear(config["n_embd"], vocab_size)

    def forward(self, idx):
        B, T= idx.shape
        tok_emb= self.token_embedding_table(idx)
        pos_emb= self.position_embedding_table(torch.arange(T, device= idx.device))

        x= self.drop(tok_emb + pos_emb)

        causal_mask= nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        for block in self.blocks:
            x= block(x, src_mask= causal_mask)

            x= self.ln_f(x)
            return self.lm_head(x)
        