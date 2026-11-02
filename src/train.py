import torch
import torch.nn.functional as F
from tokenizer import CharacterTokenizer
from model import SmallLanguageModel

batch_size= 4
block_size= 8
n_embd= 32
learning_rate= 1e-3
train_steps= 1000
data_path= "../data/input.txt"

tokenizer= CharacterTokenizer(data_path)
with open(data_path, "r", encoding= "utf-8") as file:
    raw_text= file.read()

data_tensor= torch.tensor(tokenizer.encode(raw_text), dtype= torch.long)
n= int(0.9 * len(data_tensor))

train_data= data_tensor[:n]

def get_batch(data, batch_size, block_size):
    ix= torch.randint(len(data) - block_size, (batch_size,))
    x= torch.stack([data[i: i+block_size] for i in ix])
    y= torch.stack([data[i+1: i+block_size+1] for i in ix])

    return x, y


model= SmallLanguageModel(tokenizer.vocab_size, n_embd, block_size)
optimizer= torch.optim.AdamW(model.parameters(), lr= learning_rate)

for step in range(train_steps):
    xb, yb= get_batch(train_data, batch_size, block_size)

    logits= model(xb)
    B, T, C= logits.shape

    loss= F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print(f"Final Step {train_steps-1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../checkpoints/tiny_shakespeare_model.pt")
print("Model saved to checkpoints")