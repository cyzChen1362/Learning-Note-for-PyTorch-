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

# ********************************************************************************
# AdaDelta算法其实就是RMSProp算法的改进
#
# 首先复习一下RMSProp算法：
# st ← gamma * st−1 + (1 - gamma) * gt ⊙ gt
# xt ← xt−1 − η ⊙ gt / (st + ϵ)^0.5
# 然后AdaDelta算法的改进其实就在于那个固定的n
# 把这个n换成某个不固定的Δxt−1(+ϵ)
# 而这个Δxt−1用的就是指数加权移动平均，也就是学习率随着梯度同样变化，这就很合理了
#
# 具体公式是：
# st ← p * st−1 + (1 - p) * gt ⊙ gt （这个不变）
# gt′ ←  (Δxt−1 + ϵ)^0.5 ⊙ gt / (st + ϵ)^0.5 （这里就是学习率换了）
# Δxt-1 ← ρΔxt−2 + (1 − ρ) gt-1′ ⊙ gt-1′ （就是学习率换成这个了）
# ********************************************************************************

"""
    AdaDelta算法
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
def init_adadelta_states():
    # s是每个特征的梯度的运算结果，那s矩阵的元素数自然是等于feature_num
    # s_w, s_b 分别是对w和b的s
    # delta_w, delta_b 同样分别是对w和b的delta
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    # params就是传两个参数进来的；states传入两个元组
    # 例如说zip的第一轮，读到了w和(s_w, delta_w)，那就继续处理
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        g =  p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g

d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)

# ========================
# 简洁实现
# ========================

d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)

