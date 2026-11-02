import torch
import torch.nn.functional as F
from tokenizer import CharacterTokenizer
from model import SmallLanguageModel

block_size= 8
n_embd= 32
tokenizer= CharacterTokenizer("../data/input.txt")

model= SmallLanguageModel(tokenizer.vocab_size, n_embd, block_size)
model.load_state_dict(torch.load("../checkpoints/tiny_shakespeare_model.pt"))
model.eval()


def generate_text(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond= idx[:, -block_size:]
        with torch.no_grad():
            logits= model(idx_cond)

        logits= logits[:, -1, :]
        probs= F.softmax(logits, dim= -1)
        idx_next= torch.multinomial(probs, num_samples= 1)
        idx= torch.cat((idx, idx_next), dim= 1)

    return idx

context= torch.zeros((1, 1), dtype= torch.long)
generated_indices= generate_text(model, context, max_new_tokens= 500)
print(tokenizer.decode(generated_indices[0].tolist()))