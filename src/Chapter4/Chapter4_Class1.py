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
from collections import OrderedDict

"""
    模型构造
"""

# =======================
# 1. 继承Module类来构造模型
# =======================

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    # 声明里面层的顺序是没有关系的，真正执行的顺序由forward控制
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”（下一节）将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    # 必须是这个名字
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

# 实例化MLP类得到模型变量net
X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)
print(net(X))
# 往net里传入X，相当于调用下面这一句
# out = net.__call__(X)
# print(out)

# =====================================================
# 2. 实现一个与Sequential类有相同功能的MySequential类
# =====================================================

class MySequential(nn.Module):
    # from collections import OrderedDict
    def __init__(self, *args):
        # super是固定格式，调用父类 nn.Module 的构造函数，确保模型正确注册其子模块
        # 构造函数接受任意数量的位置参数args；可以传入：
        # 若干个网络层（如MySequential(nn.Linear(10, 20), nn.ReLU(), ...)）
        # 一个OrderedDict（如MySequential(OrderedDict([("fc1", nn.Linear(10, 20)), ("act", nn.ReLU())]))）
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                # 自动给每个模块起名字："0", "1", "2", ...并添加
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input

"""
在 nn.Module 的源码中，add_module() 方法定义如下（简化版）：

def add_module(self, name: str, module: Optional["Module"]) -> None:
    if not isinstance(module, Module) and module is not None:
        raise TypeError(...)
    self._modules[name] = module
    
你可以看到它直接往 self._modules 赋值，而这个 _modules 是在 nn.Module.__init__() 中定义的：

self._modules = OrderedDict()

"""

# 实例化MySequential类得到模型变量net
net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        )
print(net)
net(X)
print(net(X))

# =========================
# 3. ModuleList类
# =========================

# **********************************************************************
# ModuleList仅仅是一个储存各种模块的列表
# 这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配）
# 而且没有实现forward功能需要自己实现
# 所以下面执行net(torch.zeros(1, 784))会报NotImplementedError
# **********************************************************************

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError

# ModuleList的出现只是让网络定义前向传播时更加灵活，见下面官网的例子
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 使用 nn.ModuleList 保存了10个 nn.Linear(10, 10) 层，索引为0 ~ 9
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        # 遍历；i 是索引0 ~ 9
        # l 是当前的 Linear(10, 10) 层，等价于 self.linears[i]
        for i, l in enumerate(self.linears):
            # 除法，向下取整
            # 用两个层来运算并相加
            x = self.linears[i // 2](x) + l(x)
        return x

# 可以比较一下普通的Module类下的一个list和ModuleList：
class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        # 这里包含了两个参数并自动创建了：
        # 权重矩阵10*10, 偏置矩阵1*10
        self.linears = nn.ModuleList([nn.Linear(10, 10)])

class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]

# 加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
net1 = Module_ModuleList()
# 这个并不会自动有参数
net2 = Module_List()

print("net1:")
for p in net1.parameters():
    print(p.size())

print("net2:")
for p in net2.parameters():
    print(p)

# =========================
# 4. ModuleDict类
# =========================
# 和ModuleList类似

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
# Linear(in_features=784, out_features=256, bias=True)
print(net.output) # 访问
# Linear(in_features=256, out_features=10, bias=True)
print(net)
# ...
# 同理并不会自动有forward方法
# net(torch.zeros(1, 784)) # 会报NotImplementedError

# =========================
# 5. 构造复杂的模型
# =========================

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        # x.norm() 是张量的L2范数（欧几里得范数）
        # 对整个张量中所有元素求平方和后开方，返回一个标量张量
        # 也就是缩放一下输出量，防止梯度消失或爆炸
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))

# 嵌套调用一下

class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
print(net(X))





