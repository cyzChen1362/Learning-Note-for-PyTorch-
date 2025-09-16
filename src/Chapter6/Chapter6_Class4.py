"""
    循环神经网络的从零开始实现
"""

# ========================
# 读取数据集
# ========================

import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 见Chapter6_Class3
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# ========================
# one-hot向量
# ========================
# 当然其实RNN并不常用独热码，只是这里用了
# n_class：类别总数
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    # 如果x有n行，那么res行数和x一样；
    # 如果x只有一行，那么res行数等于x的第一个维度的元素数
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    # dim=1：沿列方向写
    # x.view(-1,1)：把类别索引变成形状 (batch,1)
    # src=1：要写入的值是 1
    # 对每一行，根据 x[i] 指定的列，把 0 改成 1
    res.scatter_(1, x.view(-1, 1), 1)
    return res

x = torch.tensor([0, 2])
print(one_hot(x, vocab_size))

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    # 遍历，看第i列的batch_size行数据
    # 然后做one hot，输出batch_size个one hot编码结果
    # 一共输出X.shape[1]个tensor，每个tensor有batch_size个0-1矩阵
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
print(X)
inputs = to_onehot(X, 10)
print(len(inputs), inputs[0].shape)
print(inputs)

# ========================
# 初始化模型参数
# ========================

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

# 就是正态分布而已
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    # Ht = Xt * Wxh + Ht−1 * Whh + bh
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    # Ot = Ht * Whq + bq
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

# ========================
# 定义模型
# ========================

# 返回初始化的隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义了在一个时间步里如何计算隐藏状态和输出
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 如果是别的，例如LSTM，传进来的state可能是几个
    # 所以这样写方便扩展
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

# 如果对input的结构有疑问可以看这个：https://chatgpt.com/s/t_68c7c7c8c8008191856a6034e9625ab3
# 仔细琢磨一下是能看懂这个结构的
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

# ========================
# 定义预测函数
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# prefix：生成的起始文本，例如 "周杰"
# num_chars：要额外生成的字符数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    # 这里batch_size等于1
    state = init_rnn_state(1, num_hiddens, device)
    # 初始化输出，保证以prefix[0]开头
    output = [char_to_idx[prefix[0]]]
    # 循环的任务是：
    # 先把前缀的剩余字符“喂”给 RNN（但不用我们手动再写第一个字符）
    # 再生成需要的 num_chars 个新字符
    # 一共输出num_chars + len(prefix)个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        # 即取出上个循环最后一个预测的字符索引作为这个循环的第一个输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            # 一开始的时候，开头的输出强制为prefix
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # 经典操作
            output.append(int(Y[0].argmax(dim=1).item()))
    # 前面预测的全是index，现在转回char然后拼接成序列输出
    return ''.join([idx_to_char[i] for i in output])

# ========================
# 裁剪梯度
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

# ========================
# 困惑度
# ========================

# ********************************************************************************
# 损失函数使用的是 cross_entropy
# 而评估指标使用的是 exp(cross_entropy)
# 最佳情况下，模型总是把标签类别的概率预测为1，也就是cross_entropy=0，此时困惑度为1；
# 最坏情况下，模型总是把标签类别的概率预测为0，也就是cross_entropy=∞，此时困惑度为正无穷；
# 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
# ********************************************************************************

# ========================
# 定义模型训练函数
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    # 如果是随机采样（Chapter6_Class3）
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    # 如果是相邻采样（Chapter6_Class3）
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        # 这里的corpus_indices是原字符集的对应索引列表，放入fn里面做iter
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    # 相当于requires_grad=False
                    # grad_fn=None
                    s.detach_()

            """
                inputs[t] 的形状：(B, V)（第 t 个时间步的 one-hot 批）
                也就是说，一共有num_steps个时间步，在inputs列表里面就会有num_steps个子矩阵
                每个矩阵的行数是batch_size，列数是vocab_size，全是01变量
                
            """

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            # therefore, outputs是 num_steps * batch_size 行 vocab_size列独热码
            outputs = torch.cat(outputs, dim=0)

            # Y是num_steps * batch_size 行 1 列非独热码
            # 应和outputs相对应（而不是batch_size * num_steps）

            # torch.transpose(Y, 0, 1)交换维度 0 和 1
            # 结果形状： (num_steps, batch_size)（而不是batch_size,num_steps）
            # 这样.view才对得上
            # 然后.view(-1), 与 (num_steps * batch_size, vocab_size) 的行顺序完全对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()

            # 这里即使是随机采样仍然使用裁剪梯度
            # 如果想要区分开那θ设置大一点就好了
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均（经典batch_size问题）

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        # pred_period 用来控制训练过程中多长时间打印一次模型在文本生成上的效果
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

# ========================
# 训练模型并创作歌词
# ========================

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词

# 随机采样训练模型
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

# 相邻采样训练模型
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
