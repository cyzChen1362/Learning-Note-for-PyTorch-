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
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    多项式函数拟合实验
"""

# =======================
# 1. 生成数据集
# =======================
# 真实值：y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + e
# 这里的e就是随机噪声

# 样本数，参数等
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# 特征，随机生成整数
features = torch.randn((n_train + n_test, 1))
# torch.cat(..., 1) 表示按列拼接
# 总特征变为n行3列，第0列是x，第1列是x^2，第2列是x^3
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)

# 算理想值
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
# 添加高斯噪声（模拟真实数据）
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# =======================
# 2. 定义、训练和测试模型
# =======================

"""

# x_vals	第一条曲线的 x 数据（通常为 epoch）
# y_vals	第一条曲线的 y 数据（如训练误差）
# x_label	x 轴标题
# y_label	y 轴标题
# x2_vals	第二条曲线的 x 数据（可选）
# y2_vals	第二条曲线的 y 数据（可选）
# legend	图例（例如 ['train', 'test']）
# figsize	图像大小，默认为 (3.5, 2.5) 英寸

"""

# 作图函数
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(6, 4)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)

# 模型定义
# 训练100次，loss用平方损失函数
num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # Linear层，输入特征数为 train_features.shape[-1]（即列数），输出为1个值
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])
    # 创建一个数据集并用 DataLoader 随机批量加载数据
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # 训练集和测试集的loss
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
        for X, y in train_iter:
            # net并算loss
            l = loss(net(X), y.view(-1, 1))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 参数优化
            optimizer.step()
        # 标签调整为n行1列
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        # 计算这一轮优化参数后的loss并放入列表准备画图
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())

    # 绘图与模型参数输出
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    d2l.plt.show()
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)

# =======================
# 3. 三阶多项式函数拟合（正常）
# =======================

# 输入特征为x, x^2, x^3
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])


# =======================
# 4. 线性函数拟合（欠拟合）
# =======================

# 输入特征为x
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])

# =======================
# 5. 训练样本不足（过拟合）
# =======================

# 输入特征为x, x^2, x^3，训练集样本个数为2
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])

# =======================
# 6. 五阶多项式函数拟合（过拟合）
# =======================

# 总特征变为n行5列，第0列是x，第1列是x^2，第2列是x^3，第3列是x^4
poly_features_overfitting = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3), torch.pow(features, 4)), 1)

# 输入特征为x, x^2, x^3, x^4
fit_and_plot(poly_features_overfitting[:n_train, :], poly_features_overfitting[n_train:, :],
            labels[:n_train], labels[n_train:])
