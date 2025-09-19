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
# 算法：
# st ← st−1 + gt ⊙ gt
# xt ← xt−1 − η ⊙ gt / (st + ϵ)^0.5
# 根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率
# 避免统一的学习率难以适应所有维度的问题
#
# 特点：
# 如果目标函数有关自变量中某个元素的偏导数一直都较大，那么该元素的学习率将下降较快
# 如果目标函数有关自变量中某个元素的偏导数一直都较小，那么该元素的学习率将下降较慢
# “当学习率在迭代早期降得较快且当前解依然不佳时
# AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解”
# 这句话的意思是说，在早期的梯度很大，然后st-1就很大，然后学习率就很低
# 这个时候还没有到最好的解，但st-1已经被压低了，所以学习率走不动了，就很难一个有用的解
# 就类似于刹车踩死了但其实还没到目标点
#
# 具体代码直接看Class7即可，Class是此算法的高级版
# ********************************************************************************

"""
    AdaGrad算法
"""

# ========================
# 观察迭代轨迹
# ========================

import math
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

# ========================
# 从零开始实现
# ========================

features, labels = d2l.get_data_ch7()

def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

# ========================
# 简洁实现
# ========================

d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
