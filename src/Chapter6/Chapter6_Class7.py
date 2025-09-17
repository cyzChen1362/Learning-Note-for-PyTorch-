# ********************************************************************************
# 重置门: Rt = σ(Xt Wxr + Ht−1 Whr + br)
# 更新门: Zt = σ(Xt Wxz + Ht−1 Whz + bz)
# 候选隐藏状态: H~t = tanh(Xt Wxh + (Rt ⊙ Ht−1) Whh + bh)
# 隐藏状态: Ht = Zt ⊙ Ht−1 + (1 − Zt) ⊙ H~t
#
# 很显然，如果学习到 Zt = 1，则相当于隐藏状态直接继承上一时刻的隐藏状态，完全不考虑这一时刻的Xt
# 如果学习到 Zt = 0，以及 Rt = 1，相当于完全不考虑上一时刻的隐藏状态
# 所以说，更新门可以控制长期记忆，决定旧隐藏状态ℎ𝑡−1有多少直接“复制”到新状态，长期依赖靠它实现
# 重置门可以捕捉短期模式，决定计算候选状态ℎ~𝑡时，前一时刻的隐藏状态ℎ𝑡−1影响有多大，短期依赖靠它实现
#
# 更新门提供接近恒等映射：梯度可以在时间方向无衰减地流动
# 重置门允许在需要时“清空”过去：防止无关历史噪声积累，减轻梯度爆炸
# ********************************************************************************

"""
    门控循环单元（GRU）
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

def get_params():
    # 单一参数矩阵初始化
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    # 两个单一参数矩阵初始化 + 0偏置初始化合并
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

# ========================
# 定义模型
# ========================

# 隐藏状态初始化函数
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 根据门控循环单元的计算表达式定义模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # 这里和 Class4 一样，X[t-1]是第t-1时间步的矩阵，X[t]是第t时间步的矩阵
    # X[t-1]高为batch_size，宽为input_num
    for X in inputs:
        # 更新门
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        # 重置门
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        # 候选隐藏状态
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        # 隐藏状态
        H = Z * H + (1 - Z) * H_tilda
        # 输出
        Y = torch.matmul(H, W_hq) + b_q
        # 同 Class4
        outputs.append(Y)
    return outputs, (H,)

# ========================
# 训练模型并创作歌词
# ========================

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

# 每过40个迭代周期便根据当前训练的模型创作一段歌词
d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

# ========================
# 简洁实现
# ========================

# 直接调用nn模块中的GRU类即可
lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)

