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
    Transformer
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html
# b站：https://www.bilibili.com/video/BV1Kq4y1H7FL/?spm_id_from=333.1387.collection.video_card.click&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import math
import pandas as pd
import torch
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 基于位置的前馈网络
# ========================