import json
import torch
import torch.nn.functional as F
from model import DeepLanguageModel
from tokenizer import BPETokenizer


with open("configs/default_config.json", "r") as f:
    config= json.load(f)

tokenizer= BPETokenizer()
model= DeepLanguageModel(
    vocab_size=config["vocab_size"],
    n_embd=config["n_embd"],
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    block_size=config["block_size"],
    dropout=config["dropout"]
)

checkpt_path= "checkpoints/v2_model.pt"

try:
    model.load_state_dict(torch.load(checkpt_path, weights_only= True))
    print(f"Loaded model checkpoint")

except FileNotFoundError:
    print(f"No checkpoint found")
    exit()

def generate_text(model, start_prompt: str, max_new_token: int, temperature: float= 0.8, top_k: int= 40):
    model.eval()
    tokens= tokenizer.encode(start_prompt)
    idx= torch.tensor([tokens], dtype= torch.long)

    for _ in range(max_new_token):
        idx_cond= idx[:, -config["block_size"]:]

        with torch.no_grad():
            logits= model(idx_cond)

        logits= logits[:, -1, :] / temperature

        if top_k is not None:
            v, _= torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]]= -float('Inf')

        probs= F.softmax(logits, dim= -1)
        idx_next= torch.multinomial(probs, num_samples= 1)

        idx= torch.cat((idx, idx_next), dim= 1)

    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    prompt= input("Enter starting prompt: ")
    print("\nGenerating...\n")
    output= generate_text(model, start_prompt=prompt, max_new_token=100)
    print(output)