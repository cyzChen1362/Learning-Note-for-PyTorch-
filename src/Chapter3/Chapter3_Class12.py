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
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    高维线性回归实验
"""

# =======================
# 1. 生成数据集
# =======================
# 真实值：y = 0.05 + Σ 0.01xi + e
# 这里的e就是随机噪声

# 考虑过拟合问题，训练集样本数为20，输入维度为200
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
# 特征，随机生成整数
features = torch.randn((n_train + n_test, num_inputs))
# 标签
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 划分训练集、测试集
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# =======================
# 2. 初始化模型参数
# =======================
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# =======================
# 3. 定义L2范数惩罚项
# =======================
def l2_penalty(w):
    # 这个就是 (w1^2 + w2^2)/2，λ项为正则化系数，η为batch_size到时再除
    return (w**2).sum() / 2

# =======================
# 4. 定义训练和测试
# =======================

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

"""
def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): 
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2
    
"""
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# 这里传入正则化系数作为惩罚项的超参数
def fit_and_plot(lambd):
    # 初始化w,b参数，初始化训练集/测试集loss
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            # 这一批次的loss求和
            l = l.sum()

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            # 反向传播
            l.backward()
            # 这里优化器会除以batch_size，所以上面的sum()是必要的
            d2l.sgd([w, b], lr, batch_size)
            """
            def sgd(params, lr, batch_size):
                for param in params:
                    param.data -= lr * param.grad / batch_size

            """

        # 由于这里epoch=100，所以即使是在外循环append而不是每个batch_size都append也没关系
        # 况且这里batch_size=1
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())

    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'],figsize=(6, 4))
    print('L2 norm of w:', w.norm().item())

# 不使用权重衰减
fit_and_plot(lambd=0)
d2l.plt.show()

# 使用权重衰减
fit_and_plot(lambd=3)
d2l.plt.show()

# =======================
# 5. 简洁实现
# =======================
# 上面的代码只对权重w进行衰减，下面这段可以灵活对偏置b也衰减
def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    # 权重和偏置分开做优化器即可
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # torch的SGD默认不会除以batch_size，所以这里mean()即可
            l = loss(net(X), y).mean()

            # 优化器中梯度清零
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            # 反向传播
            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'],figsize=(6, 4))
    print('L2 norm of w:', net.weight.data.norm().item())
    d2l.plt.show()

# 不使用权重衰减
fit_and_plot_pytorch(0)

# 使用权重衰减
fit_and_plot_pytorch(3)

