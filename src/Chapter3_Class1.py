import torch
from time import time

a = torch.ones(1000)  # a是一个样本，有1000个全为1的特征
b = torch.ones(1000)  # b是一个样本，有1000个全为1的特征
# print(a)
# print(b)

start = time()
c = torch.zeros(1000) # c是一个样本，有1000个全为0的特征
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

start = time()
d = a + b
print(time() - start)


