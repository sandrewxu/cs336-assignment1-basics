import torch
from cs336_basics.training.sgd import SGD

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e3)

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

# lr = 1e1 - decays slower, ends ~ 11
# lr = 1e2 - decays faster, around 20 steps to 0 loss
# lr = 1e3 - diverges