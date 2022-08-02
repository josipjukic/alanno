import torch

print(torch.cuda.is_available())
device = torch.device("cuda:0")
a = torch.tensor(5, device=device)
print(a)