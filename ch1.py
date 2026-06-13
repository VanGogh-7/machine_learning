import torch
import os
import pandas as pd

ndim_1_Tensor = torch.Tensor([1, 2, 3])
print(ndim_1_Tensor)

m , n = 2, 3

zeros_Tensor = torch.zeros(m,n)

ones_Tensor = torch.ones(m,n)

full_Tensor = torch.full((m,n), 10)

print('zeros Tensor:\n', zeros_Tensor)
print('ones  Tensor:\n', ones_Tensor)
print('full  Tensor:\n', full_Tensor)


a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))

print(a+b)

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.fillna(inputs.mean())

print(inputs)

X = torch.Tensor(inputs.to_numpy(dtype=float))
y = torch.Tensor(outputs.to_numpy(dtype=float))
print(X, "\n" ,y)


ones_tensor1 = torch.ones(2,3,4)
print(ones_tensor1, "\n",len(ones_tensor1))


z = torch.arange(4.0).requires_grad_(True)

r = 2 * torch.dot(z,z)
r.backward()
print(z.grad)
z.grad.zero_()







