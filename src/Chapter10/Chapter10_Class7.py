"""
下面这段代码记得改
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)

vocab = get_vocab_imdb(train_data)
'# words in vocab:', len(vocab)

为了避免不匹配这里将不使用torchtext，考虑其他方式复现代码，详细请看Chapter4_Class6

也就是开头的import torchtext也得换

反正整个Project中，torchtext也就用这么一点

"""