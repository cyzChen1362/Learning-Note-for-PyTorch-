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

# ********************************************************************************
# Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均
# 实际上是动量法与RMSProp算法的结合
# 经过前面几章的铺垫，其推导方式已经简单明了，看书就行
#
# 复习一下RMSProp算法：
# st ← gamma * st−1 + (1 - gamma) * gt ⊙ gt
# xt ← xt−1 − η ⊙ gt / (st + ϵ)^0.5
# 复习一下动量法：
# vt = gamma * vt-1 + eta * gt
# xt = xt-1 - vt
#
# Adam算法：
# vt = β1 * vt-1 + (1 - β1) * gt （算法作者建议β1设为0.9） （来自动量法）
# st = β2 * st−1 + (1 - β2) * gt ⊙ gt （算法作者建议β2设为0.999） （来自RMSProp算法）
# v~t = vt / (1 - β1^t)
# s~t = st / (1 - β2^t)
# g't = n * v~t / (s~t + ϵ)^0.5
# xt = xt-1 - g't
# ********************************************************************************

"""
    Adam算法
"""

# ========================
# 从零开始实现
# ========================

import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()

# train_ch7实际上就w和b两个参数
def init_adam_states():
    # v是每个特征的梯度的运算结果，那v矩阵的元素数自然是等于feature_num
    # v_w, v_b 分别是对w和b的v
    # s_w, s_b 同样分别是对w和b的s
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    # 思路和上一章完全一模一样，只是公式换了
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

# 学习率为0.01的Adam算法
d2l.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)

# ========================
# 简洁实现
# ========================

d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)
