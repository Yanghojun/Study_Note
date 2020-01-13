import torch
import torch.optim

A = torch.Tensor([[1, 2]])
B = torch.Tensor([[2], [3]])

print(A)
print(B)

# result = torch.matmul(A,B)
result = A.matmul(B)
print(result)