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


# %% 2.3.5

import torch

A = torch.arange(100).reshape((5,5,4))
print(A)

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)

A_sum_axis2 = A.sum(axis=2)
print(A_sum_axis2)

print(A.mean())
print(A.sum()/A.numel())
print(A.mean(axis=0))
print(A.sum(axis=0)/A.shape[0])