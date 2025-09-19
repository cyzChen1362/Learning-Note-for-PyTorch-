"""
学习笔记

原始工程来源：
    ShusenTang / Dive-into-DL-PyTorch
    仓库地址：https://github.com/ShusenTang/Dive-into-DL-PyTorch

原始文献引用：
    @book{zhang2019dive,
        title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
        note={\url{http://www.d2l.ai}},
        year={2020}
    }

用途说明：
    本文件基于该工程，加入了个人理解与注释，用作学习笔记，不用于商业用途。

许可协议：
    原工程遵循 Apache-2.0 许可证。:contentReference[oaicite:1]{index=1}
"""

# %% 2.1.1
import torch

x = torch.arange(12)
print(x)
print(type(x))
print(x.shape)
print(x.numel())
X = x.reshape(3,2,2)
print(X)
print(X.shape)
print(X.numel())

print("\n")

print(torch.zeros((2,3,4)))
print(torch.ones((2,3,4)))
print(torch.randn(3,4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# %% 2.1.2
import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())

a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
print(a)
print(b)
print(a + b)

# %% 2.1.3
import torch

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a)
print(b)
print(a + b)

# %% 2.1.4
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(X[-1])
print(X[1:3])

X[1,2] = 9
print(X)

X[0:2,:] = 12
print(X)

# %% 2.1.5
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
before = id(Y)
Y[:] = Y + X
print(id(Y) == before)
Y += X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print(id(Z))
Z[:] = X + Y
print(id(Z))
Z[:] = Z + Y
print(id(Z))

# %% 2.1.6

import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
A = X.numpy()
B = torch.tensor(A)
print(A, "\n", type(A))
print(B, "\n", type(B))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
