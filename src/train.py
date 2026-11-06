import json
import os
import torch
import torch.nn.functional as F
from model import DeepLanguageModel
from tokenizer import BPETokenizer
from evaluate import estimate_loss

with open("configs/default_config.json", "r") as f:
    config= json.load(f)

# define tokenizer
tokenizer= BPETokenizer()
with open(config["data_path"], "r", encoding= "utf-8") as f:
    text= f.read()



print(f"___Encoding dataset___")
encoded_data= tokenizer.encode(text)
data_tensor= torch.tensor(encoded_data, dtype= torch.long)


# split data into train and val
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split, batch_size, block_size):
    data= train_data if split == "train" else val_data

    idx= torch.randint(len(data) - block_size, (batch_size, ))
    x= torch.stack([data[i: i+block_size] for i in idx])
    y= torch.stack([data[i+1: i + block_size + 1] for i in idx])

    return x, y

# define model
model= DeepLanguageModel(vocab_size=config["vocab_size"],
    n_embd=config["n_embd"],
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    block_size=config["block_size"],
    dropout=config["dropout"])


optimizer= torch.optim.AdamW(model.parameters(), lr= config["learning_rate"])


print(f"__Starting training__")
for step in range(config["max_steps"]):
    if step % config["eval_interval"] == 0 or step == config["max_steps"] - 1:
        
        metrics= estimate_loss(model, get_batch, train_data, val_data, config["batch_size"], config["block_size"], config["eval_iters"])
        print(f"Step {step:5d} | Train Loss: {metrics['train']['loss']:.4f} (PPL: {metrics['train']['perplexity']:8.2f}) | Val Loss: {metrics['val']['loss']:.4f} (PPL: {metrics['val']['perplexity']:8.2f})")


    xb, yb= get_batch('train', config["batch_size"], config["block_size"])

    logits= model(xb)
    B, T, C= logits.shape
    loss= F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()


# save model
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path= "checkpoints/v2_model.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"\nTraining complete. Model checkpoint saved to {checkpoint_path}")