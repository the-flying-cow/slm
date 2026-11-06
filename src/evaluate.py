
import torch
import math
import torch.nn.functional as F

@torch.no_grad()
def estimate_loss(model, get_batch , train_data, val_data, batch_size, block_size, eval_iters= 20):
    out= {}
    model.eval()

    for split, data in [('train', train_data), ('val', val_data)]:
        losses= torch.zeros(eval_iters)
        for k in range(eval_iters):

            X, Y= get_batch(split, batch_size, block_size)
            logits= model(X)

            B, T, C= logits.shape
            loss= F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k]= loss.item()

        mean_loss= losses.mean().item()

        perplexity= math.exp(mean_loss) if mean_loss < 20 else float('inf')
        out[split]= {'loss': mean_loss, 'perplexity': perplexity}

    model.train()

    return out