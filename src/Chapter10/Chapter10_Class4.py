r"""
学习笔记

原始工程来源：
    D2L (Dive into Deep Learning) 中文版
    仓库地址：https://github.com/d2l-ai/d2l-zh
    官方网站：https://zh.d2l.ai/

原始文献引用：
    @book{zhang2019dive,
        title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
        note={\url{https://zh.d2l.ai/}},
        year={2020}
    }

用途说明：
    本文件基于《动手学深度学习》中文版（d2l-zh）及其代码进行学习与注释，
    仅作个人学习笔记与交流之用，不用于商业用途。

许可协议：
    原工程遵循 Apache-2.0 许可证。
"""

"""
    注意力提示
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html
# b站：https://www.bilibili.com/video/BV1264y1i7R1/?spm_id_from=333.1387.collection.video_card.click&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

# ========================
# 注意力的可视化
# ========================

import torch
import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt

#@save
# 注意力权重热力图可视化
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(6, 5),
                  cmap='Reds'):
    """显示矩阵热图"""
    # 输入matrices的形状是 （要显示的行数，要显示的列数，查询的数目，键的数目）
    # 具体每个参数的解析如下：
    # 可以先参考Chapter10_Class6第三部分的注释，理解Q K V先
    # https://chatgpt.com/s/t_68ea1d737348819181ad6c85e39397bd
    # https://chatgpt.com/s/t_68ea1d818bdc8191863ad1dd622353dd
    # https://chatgpt.com/s/t_68ea1d90daf4819181a2c74f794e1336
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
    plt.show()

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
