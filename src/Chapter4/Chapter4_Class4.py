import torch
from torch import nn

"""
    自定义层
"""

# ===================================
# 1. 不含模型参数的自定义层
# ===================================
# 和4.1节中使用Module类构造模型类似
# 构造一个CenteredLayer层
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

# 实例化并做前向运算
layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)) # tensor([-2., -1.,  0.,  1.,  2.])

# 构造更复杂的模型
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

# 打印
y = net(torch.rand(4, 8))
y.mean().item() # 0.0

# ===================================
# 2. 含模型参数的自定义层
# ===================================
# Parameter类其实是Tensor的子类，
# 如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里
# 和Module -- ModuleList -- ModuleDict关系一样
# 同样可以Parameter -- ParameterList -- ParameterDict
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyDense()
print(net)



