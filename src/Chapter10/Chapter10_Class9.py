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
    多头注意力
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html
# b站：https://www.bilibili.com/video/BV1Kq4y1H7FL/?spm_id_from=333.1387.collection.video_card.click&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import math
import torch
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 实现
# ========================

#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 经过reshape后:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 经过permute后:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

#@save
# 多头注意力
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    # 这里小改了一下教材，最终的输出可以指定，不一定是num_hiddens这么多（当然默认是）
    # 然后增加了个注意力权重的输出
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, num_outputs = None, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        if num_outputs is not None:
            self.W_o = nn.Linear(num_hiddens, num_outputs, bias=bias)
        else:
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，query_size or keys_size or values_size)
        # valid_lens的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        # ********************************************************************************
        # 这里这样干，主要是因为想要用一个全连接层直接训练多个头的Q\K\V
        # 所以这里输出的num_hiddens其实是hiddens_for_each_head*num_heads
        # 然后，算注意力的时候，每个batch之间不能互相干扰，多个头之间每个头也不能互相干扰
        # 所以直接把num_heads维塞入batch_sizes维
        # 将(batch_size,查询或者“键－值”对的个数,hiddens_for_each_head*num_heads)
        # 变成(batch_size*num_heads,查询或者“键－值”对的个数,hiddens_for_each_head)
        # 之后放入self.attention时，d2l.DotProductAttention计算中batch_size维是不会相互干扰的
        # torch.bmm只作用于同一batch的(num_queries，d)@(d,num_keys)，不会跨batch相乘
        # 也就是把heads放入batch维度，同样有不会跨heads相乘的效果
        # 最后逆过程把heads从batch维还原出来即可
        # 同时复制valid_lens以适应batch_size*num_heads的形状，在attention中mask
        # ********************************************************************************
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 注意力权重
        self.attention_weights = transpose_output(self.attention.attention_weights, self.num_heads)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # 最终输出：(batch_size，查询的个数，num_outputs)
        return self.W_o(output_concat)

num_hiddens_of_MultiHeads, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens_of_MultiHeads, num_hiddens_of_MultiHeads, num_hiddens_of_MultiHeads,
#                                num_hiddens_of_MultiHeads, num_heads, 0.5)
# attention.eval()
#
# batch_size, num_queries = 2, 4
# num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
# X = torch.ones((batch_size, num_queries, num_hiddens_of_MultiHeads))
# Y = torch.ones((batch_size, num_kvpairs, num_hiddens_of_MultiHeads))
# print(attention(X, Y, Y, valid_lens).shape)

# ========================
# 训练
# ========================

class Seq2SeqAttentionDecoder(d2l.AttentionDecoder):
    # 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；
    # 解码器上一个时间步的最终层隐状态将用作查询；
    # 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            num_hiddens, num_hiddens, num_hiddens, num_hiddens_of_MultiHeads, num_heads, dropout, num_outputs = num_hiddens)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(num_steps,batch_size,num_hiddens)
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        # 这里的enc_valid_lens是怎么传进来的？见下文
        outputs, hidden_state = enc_outputs
        # permute之后的输出是(batch_size,num_steps,num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # 这里X输入的形状是(batch_size,num_steps)
        # 当然，在训练中X使用的是标签值，也就是强制教学 （见Y_hat, _ = net(X, dec_input, X_valid_len)）
        # 一开始承接的state是编码器输出的state
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens)
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)（permute后）
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        # for step in num_steps:
        for x in X:
            # ********************************************************************************
            # 第一个循环中，query是编码器最后一个输出的最顶层的隐状态
            # 随后，hidden_state在循环中被更新为这一时间步的最后一个输出的隐状态
            # 在下一个循环中，query将会是上一个循环的时间步的最后一个输出的最顶层隐状态
            # 在循环中，enc_outputs始终是编码器对源句子所有时间步的最终层隐状态，并不更新
            # 也就是说，每个循环的query都会被更新，而key和value是不更新的
            # 并且由于源句子后面可能有填充部分，所以传入enc_valid_lens作为遮蔽，不学习填充部分
            # ********************************************************************************
            # x形状为(batch_size,embed_size)
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            # 这里得到的context是由编码器隐状态应用了加性注意力的隐状态
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # rnn的state使用上一时间步输出的state，存
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 250

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
    1, 1, -1, num_steps))

# 加上一个包含序列结束词元
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
