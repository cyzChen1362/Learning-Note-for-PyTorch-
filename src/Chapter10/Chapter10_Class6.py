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
    注意力评分函数
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html
# b站：https://www.bilibili.com/video/BV1Tb4y167rb?spm_id_from=333.788.recommend_more_video.0&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import math
import torch
from torch import nn
import d2lzh_pytorch as d2l

# ========================
# 掩蔽softmax操作
# ========================

#@save
# 掩蔽softmax操作
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 例如输入X为(batch_size, num_queries, num_keys)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 如果valid_lens为(batch_size,)
        if valid_lens.dim() == 1:
            # 那么复制valid_lens为(batch_size*num_queries,)
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 否则直接将valid_lens变为一维(batch_size*num_queries,)
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        # 然后将遮蔽完的X给reshape回去，并沿最后一维num_keys进行softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)

print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# ========================
# 加性注意力
# ========================

#@save
# 加性注意力
# a(q,k)=Wv.T * tanh(Wq * q + Wk * k)，当然实现的时候直接套模块就可以了
# 注意不要混淆num_queries和query_size
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):

        # queries：(batch_size, num_queries, query_size)
        # num_queries相当于目标句子长度（decoder时间步数），query_size相当于每个query的维度
        # keys：(batch_size，num_keys，key_size)
        # num_keys相当于源句子长度（encoder时间步数），key_size相当于每个词经过encoder后得到的语义向量维度
        # values：(batch_size，num_keys，value_size)
        # num_keys相当于源句子中的词数，value_size相当于每个value的向量维度

        # 其中，queries来自 decoder 当前时刻的隐藏状态（或多个时间步的状态）。
        # 每个 query 向量代表“解码器当前要生成某个词时，它在问：我该关注源句子的哪个部分？”

        # Wq * q，queries：(batch_size, num_queries, num_hiddens)
        queries = self.W_q(queries)
        # Wk * k，keys：(batch_size，num_keys，num_hiddens)
        keys = self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，num_queries，1，num_hidden)
        # key的形状：(batch_size，1，num_keys，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，num_queries，num_keys)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，num_keys，value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

# ========================
# 缩放点积注意力
# ========================

# ********************************************************************************
# 这里讲一下我对 Scaled Dot-Product Attention 的理解：
#
# 向量化版本：
# Q:n*d; K:m*d; V:m*v;
#
# Q的每一行是一个样本，每个样本是d维向量；K的每一行是一个样本，每个样本是d维度向量
# Q*KT也就是Q[1]和K[1]做内积得到Q*KT[1][1]，Q[1]和K[2]做内积得到Q*KT[1][2]
# 也就是Q*KT[i][j]是Q[i]和K[j]的相似度（如果二者正好正交，内积结果一定是0）
# 然后除以√d相当于除以常数，避免过大
# 得到的注意力分数a(Q,K)=Q*KT/√d是n*m，其中a(Q,K)[i][j]可以理解为Q[i]和K[j]的相似度
#
# 然后对a(Q,K)的每一行进行softmax，
# 那么softmax(a(Q,K))[i]也就是对于Q[i]而言，根据和K每个样本的相似度算出来的权重向量
# 例如softmax(a(Q,K))[i][j]就是K[j]对于Q[i]的权重，softmax(a(Q,K))[i][j+1]就是K[j+1]对于Q[i]的权重
#
# 然后对于注意力池化，softmax(a(Q,K))*V
# softmax(a(Q,K))每一行都是权重向量，V的每一列都是对于某个维度，所有样本各自的值
# 那么输出结果的第i个样本的第j个维度的值，肯定是V所有样本在这一维度的值根据权重求和
# 也就是softmax(a(Q,K))的第i行乘以V的第j列
#
# 顺便一提，K和V是键值对，这俩是捆绑在一起的
# 权重矩阵softmax(a(Q,K))每一行都根据K算出权重，然后又跟对应位置的V算出最终的值
# 所以是一一对应的
# ********************************************************************************

#@save
# 缩放点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，num_queries，d)
    # keys的形状：(batch_size，num_keys，d)
    # values的形状：(batch_size，num_keys，value_size)
    # valid_lens的形状:(batch_size，)或者(batch_size，num_queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # self.attention_weights的形状：(batch_size，num_queries，num_keys)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
