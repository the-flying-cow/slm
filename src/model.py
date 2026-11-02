import torch
import torch.nn as nn

class SmallLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, n_emb: int, block_size: int):
        super().__init__()
        self.block_size= block_size
        self.token_embedding_table= nn.Embedding(vocab_size, n_emb)
        self.posi_embedding_table= nn.Embedding(block_size, n_emb)

        self.blocks= nn.TransformerEncoderLayer(d_model= n_emb, nhead= 4, dim_feedforward= 128, batch_first= True)
        self.lm_head= nn.Linear(n_emb, vocab_size)

    def forward(self, idx):
        B, T= idx.shape

        tok_emb= self.token_embedding_table(idx)
        pos_emb= self.posi_embedding_table(torch.arange(T, device= idx.device))
        x= tok_emb + pos_emb

        causal_mask= nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x= self.blocks(x, src_mask= causal_mask)

        return self.lm_head(x)