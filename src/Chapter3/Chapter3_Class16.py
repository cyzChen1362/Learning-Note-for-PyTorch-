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
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    实战Kaggle比赛：房价预测
"""

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# =======================
# 1. 获取和读取数据集
# =======================

# 读取数据
train_data = pd.read_csv('../../data/Kaggle_House/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../../data/Kaggle_House/house-prices-advanced-regression-techniques/test.csv')
# 训练集和测试集形状
print(train_data.shape, test_data.shape)
# 前四个样本的前4个特征、后2个特征和标签（SalePrice）
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 第一个特征是id，可以帮助模型记住每个训练样本但难以推广到测试样本，所以不使用
# 连结特征，训练集不含SalePrice标签
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features.shape)

# =======================
# 2. 预处理数据
# =======================
# all_features.dtypes != 'object'将所有列类型不是object的置为True
# 外面的all_features.dtypes将True的几列筛出来
# .index再返回列索引
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# all_features[numeric_features]是一个DataFrame
# .apply(lambda x: ...)默认参数axis=0对列操作
# 数值标准化：将该特征的每个值先减去μ再除以σ
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值（相当于均值替换）
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转成指示特征
# 例如特征MSZoning里面有两个不同的离散值RL和RM
# 那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

# 训练集特征/标签tensor，测试集特征tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# =======================
# 3. 训练模型
# =======================
# 平方损失函数
loss = torch.nn.MSELoss()

# 设定网络，封装成一个函数
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    # 这一句时为了证明，K折交叉验证时每一折的训练都是参数清零重新开始的
    print('new net params:', [p.data.mean().item() for p in net.parameters()])
    return net

# 比赛用的对数均方根误差Log_RMSE
# sqrt( (1/n) * sum_{i=1}^n (log(y_i) - log(y_hat_i))^2 )
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        # 防止log0或者log负数
        # **************************************************
        # 这里比较妙的一点是：SalePrice标签本身就很大
        # 所以不用考虑说，如果预测值是0.x且和真实值距离较近的情况
        # 也就是“预测值小于1时强制设为1”并不会有什么误差
        # 相反，刚开始迭代的时候还可避免不稳定的问题
        # **************************************************
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

# 训练函数
# 使用Adam优化算法，对学习率相对不那么敏感
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    # 迭代器
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法，同样有权重衰减
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 这里指的是将参数转为浮点数
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 计算loss
            l = loss(net(X.float()), y.float())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 优化
            optimizer.step()
        # 训练误差列表更新
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            # 如果有传测试集标签进来，也把测试误差列表算了并更新
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# =======================
# 4. K折交叉验证
# =======================
# get_k_fold_data返回第i折交叉验证时所需要的训练和验证数据
# 就比如说分5折交叉验证，现在轮到了i=3作为验证集
# 那么get_k_fold_data就会返回1，2，4，5拼接的训练集，以及3的验证集
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    # 总得大于1折吧
    assert k > 1
    # 分K折，每折的样本数就是总样本数/K
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 取下标，slice(start, stop)：Python 的切片对象，等价于 start:stop
        # slice的返回值是一个slice对象，记录了起始、终止、步长
        """
            lst = [10, 20, 30, 40, 50, 60]
            s = slice(2, 5)   # 表示从下标2到4
            print(lst[s])     # 输出: [30, 40, 50]
        """
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            # data_train中原先有数据才能拼接
            # 否则不能做下面else的拼接
            X_train, y_train = X_part, y_part
        else:
            # 沿行上下拼接，将非第i折的样本拼接到data_train中
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

# K折交叉验证
# 这一部分并不是用来训练参数的
# 只是为了表示这个网络泛化能力很好
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        # 取出K折交叉验证中第i折交叉时所需的数据
        # 返回[X_train, y_train, X_valid, y_valid]
        # valid也就是输入train的测试集了
        data = get_k_fold_data(k, i, X_train, y_train)
        # 设定网络，输入为训练集特征数即dim=1维度大小
        # **************************************************
        # K折交叉验证时每一折的训练都是参数清零重新开始的
        # K折交叉验证是为了评估不同数据划分情况下的泛化能力
        # 不是用来选超参数或者训练参数的
        # **************************************************
        net = get_net(X_train.shape[1])
        # 将这第i折的数据塞进去训练
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # 这里返回的将是第i折交叉验证时训练集和验证集的误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 这里只作出K折交叉验证中第0折交叉时的损失下降图
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'], figsize=(6, 4))
            d2l.plt.show()
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

# =======================
# 5. 模型选择
# =======================
# 5折交叉验证，每折迭代轮次100，学习率5，不使用权重衰减，批次64
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

# =======================
# 6. 预测并在Kaggle提交结果
# =======================
# 训练及预测函数
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 创建网络
    net = get_net(train_features.shape[1])
    # 塞入训练集去训练
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 画出训练集的损失下降图
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', figsize=(6, 4))
    d2l.plt.show()
    print('train rmse %f' % train_ls[-1])
    # 将这个网络应用在测试集中，将会返回测试集预测值
    # .detach():从当前计算图中“分离”出来，意思是这个张量不再需要反向传播（梯度）
    preds = net(test_features).detach().numpy()
    # 把预测结果 preds 转成一维（保证行数和测试集对应），存到 test_data 的 SalePrice 列
    # 这里是为了把二维数组压平成一维数组（即一行），好让 Pandas 接收它为一个「列」
    # 因为你不能把一个n行1列的二维数组传到'SalePrice'列中
    # 所以你要将得到的列向量preds（列向量有n行1列所以是二维的）转成1行n列
    # 此时结果是 (1, n)，是一个二维数组。为了得到一维向量 (n,)，就取第一行 [0]：
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # 把 test_data 的 Id 和预测的 SalePrice 两列拼接成新的 DataFrame submission
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # 将 submission DataFrame 保存为 CSV 文件，不带行索引，文件名为 submission.csv
    submission.to_csv('./submission.csv', index=False)

# 最终执行
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)




