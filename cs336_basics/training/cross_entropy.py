import torch

def cross_entropy(logits, targets):
    """
    Compute the cross entropy loss.
    Take in predicted logits o_i (... batch_size vocab_size) as well as
    targets x_{i+1} (... batch_size)

    Subtract the largest element for numerical stability.
    Cancel out log and exp whenever possible.
    Handle any additional batch dimensions and return the average across the batch. 
    Assume batch-like dimensions always come first, before the vocabulary size dimension
    """

    # subtract the max of each (vocab_size) tensor
    maxes = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - maxes

    # NLL is log(sum exp(o_i[a])) - o_i[x_{i+1}]
    nll_loss = torch.log(torch.sum(torch.exp(logits), dim=-1))
    nll_loss -= torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)

    return torch.mean(nll_loss)
