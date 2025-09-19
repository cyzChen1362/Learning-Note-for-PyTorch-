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
# 梯度下降的问题在于
# 对于多元输入矩阵X，如果某xi方向的梯度过大而xj方向的梯度过小
# 那么在xi方向就容易越过最优解而在xj方向向最优解移动的方向过慢
#
# 还有另一个更直觉的问题就是
# 例如一个总的趋势是向下的一个大坑，但是下落的坡的过程中有一个凸起
# 如果仅仅使用当前位置的梯度进行迭代的话，遇到这个凸起就容易出错
# 所以还需要考虑前面几步的梯度，这就是动量法
#
# 动量法：
# vt = gamma * vt-1 + eta * gt
# xt = xt-1 - vt
#
# 指数加权移动平均：
# 例如这个式子：yt = gamma * yt-1 + (1 - gamma) * xt
# 展开：yt = (1 - gamma) * xt + (1 - gamma) * gamma * xt-1 + ...
#            + (1 - gamma) * gamma^n * xt-n
# 然后通过一堆变换，可以认为当gamma趋近于1时，高阶项可以忽略
# 例如取gamma=0.95，认为只考虑前20阶的加权平均，离当前时间步越近的xt的值获得的权重越大
# 也就是说，yt由一定阶次范围内的xt...xt-n决定
#
# 由指数加权移动平均理解动量法：
# vt = gamma * vt-1 + eta * gt = gamma * vt-1 + (1 - gamma) * (eta * gt / 1-gamma)
# 那就很显然动量受到一定阶次范围内的gt加权平均影响了（不仅仅是上一个时间步）
# ********************************************************************************

"""
    动量法
"""

# ========================
# 梯度下降
# ========================

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch

eta = 0.4 # 学习率

# f(x)
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

# f'(x)
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

# ========================
# 动量法
# ========================

def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, gamma = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

# ========================
# 从零开始实现
# ========================

# 就是airfoil_self_noise的数据包
features, labels = d2l.get_data_ch7()

def init_momentum_states():
    # 有多少个feature就有多少个梯度
    # 而梯度的形状和动量的形状是完全一致的
    # 所以动量的形状如下所示
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

# 动量法
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        # 加权和梯度下降
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

# 默认参数batch_size=10，所以是小批量梯度下降
# momentum设0.5
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.5}, features, labels)

# momentum增大到0.9
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.9}, features, labels)

# 学习率减小到原来的1/5
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.004, 'momentum': 0.9}, features, labels)

# ========================
# 简洁实现
# ========================
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)


