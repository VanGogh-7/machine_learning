import torch

ndim_1_Tensor = torch.Tensor([1, 2, 3])
print(ndim_1_Tensor)

m , n = 2, 3

zeros_Tensor = torch.zeros(m,n)

ones_Tensor = torch.ones(m,n)

full_Tensor = torch.full((m,n), 10)

print('zeros Tensor:\n', zeros_Tensor)
print('ones  Tensor:\n', ones_Tensor)
print('full  Tensor:\n', full_Tensor)









