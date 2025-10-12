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
    使用注意力机制的seq2seq
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html
# b站：https://www.bilibili.com/video/BV1v44y1C7Tg?spm_id_from=333.788.videopod.episodes&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import math
import torch
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 定义注意力解码器
# ========================

#@save
# 定义注意力解码器
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    # 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；
    # 解码器上一个时间步的最终层隐状态将用作查询；
    # 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
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

# ********************************************************************************
# 提问：Seq2SeqAttentionDecoder中的init_state中的enc_valid_lens是怎么传进来的？
#
# 首先在d2l.load_data_nmt中，返回的第一个参数是这样的：
# (src_array, src_valid_len, tgt_array, tgt_valid_len)
#
# 然后train_iter传进去d2l.train_seq2seq是这样的：
# for batch in data_iter:
#     X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
#     ......
#     Y_hat, _ = net(X, dec_input, X_valid_len)
#
# 然后EncoderDecoder类的net是这样的：
# def forward(self, enc_X, dec_X, *args):
#     dec_state = self.decoder.init_state(enc_outputs, *args)
# 也就是net传入的第三个参数X_valid_len即decoder.init_state传入的第二个参数*args
#
# 然后在训练流程中，for batch in data_iter中，
# 每net(X, dec_input, X_valid_len)一次，就传一次参给EncoderDecoder的forward
# 然后EncoderDecoder就forward一次；
# EncoderDecoder每forward一次，encoder和decoder就依次forward一次；
# 这样就算出一个batch的几个句子的结果。
# ********************************************************************************

encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape

# ========================
# 训练
# ========================

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
