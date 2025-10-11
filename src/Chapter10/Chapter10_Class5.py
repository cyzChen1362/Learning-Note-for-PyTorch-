r"""
学习笔记

原始工程来源：
    D2L (Dive into Deep Learning) 中文版
    仓库地址：https://github.com/d2l-ai/d2l-zh
    官方网站：https://zh.d2l.ai/

原始文献引用：
    @book{zhang2019dive,
        title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
        note={\url{https://zh.d2l.ai/}},
        year={2020}
    }

用途说明：
    本文件基于《动手学深度学习》中文版（d2l-zh）及其代码进行学习与注释，
    仅作个人学习笔记与交流之用，不用于商业用途。

许可协议：
    原工程遵循 Apache-2.0 许可证。
"""

"""
    注意力汇聚：Nadaraya-Watson 核回归
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html
# b站：https://www.bilibili.com/video/BV1264y1i7R1/?spm_id_from=333.1387.collection.video_card.click&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import torch
from torch import nn
import d2lzh_pytorch as d2l

# ========================
# 生成数据集
# ========================

# 根据下面的非线性函数生成一个人工数据集， 其中加入的噪声项为：
# yi = 2sin(xi) + xi^0.8 + e
# e服从均值为0和标准差为0.5的正态分布
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)

# 绘制所有的训练样本（样本由圆圈表示）
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# ========================
# 平均汇聚
# ========================

# torch.repeat_interleave(x, n) 会重复元素 n 次
# 即把训练集标签的平均值复制 n_test 次，作为测试集预测结果
# 也就是每个不同的x，结果f(x)都是训练样本输出值的平均值
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

# ========================
# 非参数注意力汇聚
# ========================

# X_repeat的形状:(n_test,n_train),
# 复制n_train行，每一行都包含着相同的测试输入（例如：同样的查询），然后reshape
# 得到的X_repeat每一行都是这一个测试输入的位置复制n_train次
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

# 观察注意力的权重
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

# ========================
# 带参数注意力汇聚
# ========================

# 批量矩阵乘法
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
# weights.unsqueeze(1) -- (2, 1, 10)
# values.unsqueeze(-1) -- (2, 10, 1)
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        # 原本queries是[q1, q2, q3,...]，然后变成了：
        # [q1, q1, q1, ...
        #  q2, q2, q2, ...
        #  ...
        #  ]
        # 接下来就不用过多解释了
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # self.attention_weights为(num_queries, num_keys)
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(num_queries, num_keys)
        # 相当于 (num_queries, 1, num_keys) * (num_queries, num_keys, 1) = (num_queries, 1, 1)
        # 最终reshape为(num_queries)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# 训练

# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))

# 用 mask 取出所有非对角元素（也就是排除自身的样本），再按行 reshape 回去，使得：
# keys[i] 包含“第 i 个样本对应的所有其他输入 x”；
# values[i] 包含“第 i 个样本对应的所有其他标签 y”；
# 因为排除了自己，所以每行长度是 n_train-1
mask = (1 - torch.eye(n_train)).type(torch.bool)   # True=非对角元素, False=对角(自身)
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[mask].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[mask].reshape((n_train, -1))

# ********************************************************************************
# 这个mask是很重要的：
# 例如下面将x_train作为queries输入，并输入keys和values
# 在NWKernelRegression中queries会被捣鼓成：
# [x1, x1, x1, ...
#  x2, x2, x2, ...
#  ...
#  ]
# 然后keys的形状是：
# [x2, x3, x4, ...
#  x1, x3, x4, ...
#  ...
#  ]
# 所以queries - keys是：
# [x1 - x2, x1 - x3, x1 - x4, ...
#  x2 - x1, x2 - x3, x2 - x4, ...
#  ...
#  ]
# 如果不使用mask，你将会见到x1-x1, x2-x2, ...
# 那训练的时候就几乎没误差了，没什么好训练的
# 然后掩码之后，后面的剧情就是：
# output[1] = K(x1-x2)*value[2] + K(x1-x3)*value[3] + ...
# output[2] = K(x2-x1)*value[1] + K(x2-x3)*value[3] + ...
# ...
# ********************************************************************************

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')