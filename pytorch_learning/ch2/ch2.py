import torch

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=torch.float, requires_grad=True)
print(X)

f = X.pow(2).sum()
print(f)

f.backward()
print(X.grad)


