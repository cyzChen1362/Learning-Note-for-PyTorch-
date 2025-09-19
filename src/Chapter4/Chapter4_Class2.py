r"""
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
from torch.nn import init

"""
    模型参数的访问、初始化和共享
"""

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
print(net(X).sum())

# =========================
# 1. 访问模型参数
# =========================
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
"""
    <class 'generator'>
    # 自动加上了层数的索引作为前缀
    0.weight torch.Size([3, 4])
    0.bias torch.Size([3])
    2.weight torch.Size([1, 3])
    2.bias torch.Size([1])
"""

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))
"""
    weight torch.Size([3, 4]) <class 'torch.nn.parameter.Parameter'>
    bias torch.Size([3]) <class 'torch.nn.parameter.Parameter'>
"""

# 下面的代码中weight1在参数列表中但是weight2却没在参数列表中
# 因为Parameter是Tensor，即Tensor拥有的属性它都有
# 比如可以根据data来访问参数数值，用grad来访问参数梯度

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name) # weight1

weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
"""
    tensor([[-0.4040, -0.0805,  0.4940, -0.2924],
            [-0.3461,  0.4942, -0.0268,  0.1254],
            [-0.1322, -0.4686, -0.3485,  0.1263]])
"""

print(weight_0.grad) # 反向传播前梯度为None
"""
    None
"""

Y = net(X).sum()
Y.backward()
print(weight_0.grad)
"""
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.1746, -0.3004, -0.4105, -0.6236],
            [ 0.0129,  0.0140,  0.0235,  0.0377]])
"""

# =========================
# 2. 初始化模型参数
# =========================
# 使用init模块进行初始化
# 使用正态分布初始化权重
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    """
        0.weight tensor([[-0.0049,  0.0066,  0.0064, -0.0128],
            [ 0.0092, -0.0063, -0.0013,  0.0082],
            [-0.0168, -0.0054,  0.0001, -0.0043]])
        2.weight tensor([[-0.0141, -0.0011, -0.0029]])

    """

# 使用常数初始化权重
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)
    """
        0.bias tensor([0., 0., 0.])
        2.bias tensor([0.])
    """

# =========================
# 3. 自定义初始化方法
# =========================
# PyTorch是怎么实现这些初始化方法的，例如torch.nn.init.normal_：
"""
def normal_(tensor, mean=0, std=1):
    # 不记录梯度
    with torch.no_grad():
        # 每个函数都使用tensor.normal方法
        return tensor.normal_(mean, std)

"""

# 自定义参数初始化方法
# 令权重有一半概率初始化为0，
# 有另一半概率初始化为[−10,−5]和[5,10]两个区间里均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        # 这里还挺聪明
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)
"""
    0.weight tensor([[ 0.0000,  8.5069,  0.0000, -7.3966],
            [ 9.9446,  9.6018, -8.5192, -8.2891],
            [ 7.1540,  7.7833,  0.0000,  7.1986]])
    2.weight tensor([[0.0000, 7.6770, 6.0126]])
    
"""

# 同样，可以根据.data进行模型参数的修改，从而不影响梯度
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
"""
    0.bias tensor([1., 1., 1.])
    2.bias tensor([1.])

"""

# ======================
# 4. 共享模型参数
# ======================
# 在上一节中，共享模型参数就是Module类的forward函数里多次调用同一个层；
# 在这里，传入Sequential的模块是同一个Module实例，参数也是共享的
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
"""
    Sequential(
      (0): Linear(in_features=1, out_features=1, bias=False)
      (1): Linear(in_features=1, out_features=1, bias=False)
    )
    
"""
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
"""
    # 虽然有两个层，但也就只有一个参数
    0.weight tensor([[3.]])
    
"""
print(id(net[0]) == id(net[1])) # True
print(id(net[0].weight) == id(net[1].weight)) # True

x = torch.ones(1, 1)
# net有两层，每层都是乘以三
y = net(x).sum()
print(y) # 9
y.backward()
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6

# **************************************************
# 这里解释一下：
# h1 = w * x = 3
# y = w * h1 = w * (w * x) = w^2 * x = 9
# 反过来求梯度是对w求导，而不是对x
# dy/dw = x*2*w = 6
# **************************************************
