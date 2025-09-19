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

# 使用ParameterList
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        # 列表，三个完全一样的参数list
        # 这里也是同样的道理，传进去的是一个Parameter类，它就自动添加到模型的参数列表里了
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        # 新增一个参数
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            # 这个迭代就挺帅
            # 但注意，这里只是存粹的输入和参数的做法
            # 并没有什么“层”的概念
            x = torch.mm(x, self.params[i])
        return x
net = MyListDense()
print(net)
"""
    # 这里并不是什么“层”的名称
    # 只是列出参数 
    # (index):Parameter containing: [torch.FloatTensor of size 4x4]
    MyDense(
      (params): ParameterList(
          (0): Parameter containing: [torch.FloatTensor of size 4x4]
          (1): Parameter containing: [torch.FloatTensor of size 4x4]
          (2): Parameter containing: [torch.FloatTensor of size 4x4]
          (3): Parameter containing: [torch.FloatTensor of size 4x1]
      )
    )
"""

# 使用ParameterDict
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                # 参数名: 参数形
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        # 同样是dict.update()方法
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        # 在运行时选择不同的“线性层”（其实只是矩阵）
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)
"""
    MyDictDense(
      (params): ParameterDict(
          (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
          (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
          (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
      )
    )
"""
# therefore...
x = torch.ones(1, 4)
print(net(x, 'linear1')) # tensor([[1.5082, 1.5574, 2.1651, 1.2409]], grad_fn=<MmBackward>)
print(net(x, 'linear2')) # tensor([[-0.8783]], grad_fn=<MmBackward>)
print(net(x, 'linear3')) # tensor([[ 2.2193, -1.6539]], grad_fn=<MmBackward>)

# 用自定义层构建模型：
net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))

"""
    # 这个net分两层，每层有几个参数
    Sequential(
      (0): MyDictDense(
        (params): ParameterDict(
            (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
            (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
            (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
        )
      )
      (1): MyListDense(
        (params): ParameterList(
            (0): Parameter containing: [torch.FloatTensor of size 4x4]
            (1): Parameter containing: [torch.FloatTensor of size 4x4]
            (2): Parameter containing: [torch.FloatTensor of size 4x4]
            (3): Parameter containing: [torch.FloatTensor of size 4x1]
        )
      )
    )
"""
