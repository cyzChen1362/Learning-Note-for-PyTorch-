# %% 2.3.5

import torch

A = torch.arange(100).reshape((5,5,4))
print(A)

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)

A_sum_axis2 = A.sum(axis=2)
print(A_sum_axis2)

print(A.mean())
print(A.sum()/A.numel())
print(A.mean(axis=0))
print(A.sum(axis=0)/A.shape[0])