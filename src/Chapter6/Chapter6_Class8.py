# ********************************************************************************
# 为什么候选隐藏状态/候选记忆细胞的计算中仍然要使用到上一时刻的隐藏状态/记忆细胞？
# 难道更新门/遗忘门不是已经控制了上一时刻的隐藏状态/记忆细胞对当前输出的影响了吗？

# Answer：
# 候选隐藏状态/候选记忆细胞代指的是这个时刻的事情对这个时刻的输出的影响
# 而这个时刻的事情当然会受到上个时刻的事情的影响

# Simple Explanation 4 LSTM：
# Ct = 今天记住的事情
# C~t = 今天实际新发生的事情（虽然你不一定全记得住）
# Ht = 今天你脑海里的想法
# Xt = 今天的机遇，例如今天的天气，今天遇到的人
# 首先，今天实际新发生的事情 = 今天的机遇 + 你昨天的想法
# 也就是 C~t = tanh(Xt Wxc + Ht−1 Whc + bc)
# 然后，今天记住的事情 = 部分昨天记住的事情 + 部分今天新发生的事情
# 也就是 Ct = Ft ⊙ Ct−1 + It ⊙ C~t
# 最后，今天的想法 = 总结今天记住的事情
# 也就是 Ht = Ot ⊙ tanh(Ct)
# ********************************************************************************

"""
    长短期记忆（LSTM）
"""

# ========================
# 读取数据集
# ========================

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
# 初始化模型参数
# ========================

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

# 和上一节一样
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

# ========================
# 定义模型
# ========================

# 毕竟一个Ct一个Ht都要初始化
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

# 根据长短期记忆的计算表达式定义模型
# 需要注意的是，只有隐藏状态会传递到输出层，而记忆细胞不参与输出层的计算
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    # 同前面几节
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

# ========================
# 训练模型并创作歌词
# ========================

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

# 每过40个迭代周期便根据当前训练的模型创作一段歌词
d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

# ========================
# 简洁实现
# ========================

lr = 1e-2 # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)

