"""
    循环神经网络的简洁实现
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

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# ========================
# 定义模型
# ========================

num_hiddens = 256
# rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试

# ********************************************************************************
# 注意：对于不同的时间步输入的X，它们的隐藏层参数和输出层参数都是一样的
# 区别仅仅在于它们的输入和隐藏状态而已
# rnn_layer的输入形状为(时间步数, 批量大小, 输入个数)，输入个数即one-hot向量长度
# 隐藏状态矩阵为(时间步数, 批量大小, 隐藏单元个数)
#
# 注：
# 在 PyTorch 的 nn.RNN（以及 LSTM/GRU） 里，输出维度只等于hidden_size
# 也就是这个 RNN 层没有单独的「输出层」，它的“输出”指的就是每个时间步的隐藏状态
# 所以想要vocab_size的输出得自己再加一个线性层
#
# 另：
# 虽然 nn.RNN的两个输出都是隐藏层，但也是有区别的（“粒度不同”）
# output = “整条时间线每一刻的隐藏状态序列”
# h_n = “这条时间线最后一刻的隐藏状态（可续接）”
# output用来做序列到序列（sequence-to-sequence）任务，比如机器翻译、语音识别
# h_n用来做只关心最终结果的任务（例如文本分类、情感分析）或者分段输入
# ********************************************************************************

rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

# 测试
num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)

# 本类已保存在d2lzh_pytorch包中方便以后使用
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        # rnn_layer.bidirectional指的是双向RNN
        # 双向RNN指的是不仅仅从正向往后推，还从反向往前推
        # 适用于机器翻译这种输入就知道全文，输出改成英文的任务
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        # 全连接层or稠密层，just TensorFlow 习惯使用 dense
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = d2l.to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)
        # 它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

# ========================
# 训练模型
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# 这个model就是接入你的RNN模型了
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    # 同 Class4 先把输出的第一个字符固定为prefix
    output = [char_to_idx[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            # 看看state是不是元组？（LSTM是）
            if isinstance(state, tuple): # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        # 同 Class4，固定开头的几个字符输出
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

# 使用权重为随机值的模型来预测一次
model = RNNModel(rnn_layer, vocab_size).to(device)
print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

# 训练函数，和Class4一模一样
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 模型转到cuda
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        # 省得exp算出来无穷大
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

# 训练并输出
num_epochs, batch_size, lr, clipping_theta = 1000, 32, 1e-3, 1e-2 # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)

