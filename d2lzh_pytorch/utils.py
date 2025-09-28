import collections
import math
import os
import random
import sys
import tarfile
import time
import json
import zipfile
from tqdm import tqdm
from PIL import Image
from collections import namedtuple

import hashlib
import requests
import pandas as pd
import shutil

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.ops import box_iou

# import torchtext
# import torchtext.vocab as Vocab
import numpy as np

# ###################### 3.2 ############################
def set_figsize(figsize=(6, 4)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j) 

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): 
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data



# ######################3##### 3.5 #############################
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # plt.show()

# 5.6 修改
# def load_data_fashion_mnist(batch_size, root='~/Datasets/FashionMNIST'):
#     """Download the fashion mnist dataset and then load into memory."""
#     transform = transforms.ToTensor()
#     mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
#     mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
#     if sys.platform.startswith('win'):
#         num_workers = 0  # 0表示不用额外的进程来加速读取数据
#     else:
#         num_workers = 4
#     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_iter, test_iter




# ########################### 3.6  ###############################
# (3.13节修改)
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))




# ########################### 3.7 #####################################3
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# ########################### 3.11 ###############################
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(6, 4)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    # plt.show()




# ############################# 3.13 ##############################
# 5.5 修改
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         if isinstance(net, torch.nn.Module):
#             net.eval() # 评估模式, 这会关闭dropout
#             acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#             net.train() # 改回训练模式
#         else: # 自定义的模型
#             if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
#                 # 将is_training设置成False
#                 acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
#             else:
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
#         n += y.shape[0]
#     return acc_sum / n






# ########################### 5.1 #########################
def corr2d(X, K):  
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y



# ############################ 5.5 #########################

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))



# ########################## 5.6 #########################3
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter



############################# 5.8 ##############################
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])



# ########################### 5.11 ################################
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output))) 
    return net



# ############################## 6.3 ##################################3
def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y





# ###################################### 6.4 ######################################
def one_hot(x, n_class, dtype=torch.float32): 
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehot(X, n_class):  
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else: 
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()
            
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

                
                
                
# ################################### 6.5 ################################################
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)  
                state = (state[0].to(device), state[1].to(device))
            else:   
                state = state.to(device)
            
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()
    
            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)
            
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))




# ######################################## 7.2 ###############################################
def train_2d(trainer):  
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):  
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    #
    plt.show()




# ######################################## 7.3 ###############################################
def get_data_ch7():  
    data = np.genfromtxt('../../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss
    
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)
    
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失
            
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
                
            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# 本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字
# 例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2 
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #
    plt.show()




############################## 8.3 ##################################
class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))





# ########################### 9.1 ########################################
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    # plt.show()
    return axes

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))




############################## 9.3 #####################
def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)




############################ 9.4 ###########################
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores. 
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores. 
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = [] # pair of (size, sqrt(ration))
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    
    pairs = np.array(pairs)
    
    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)
    
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    
    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)

# 这里我直接把MultiBoxPrior换成新的GPU版

def MultiBoxPrior_GPU(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    r"""
    按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    跟原版除了用GPU和tensor规避numpy之外基本没有不同
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    device = feature_map.device
    dtype  = feature_map.dtype

    pairs = [] # pair of (size, sqrt(ration))
    # 只对包含s1或r1的大小与宽高比的组合感兴趣
    for r in ratios:
        pairs.append((sizes[0], math.sqrt(r)))
    for s in sizes[1:]:
        pairs.append((s, math.sqrt(ratios[0])))

    # 将嵌套列表转为torch.tensor，才有“所有行的第一列”
    # 总长度为n+m-1
    pairs = torch.tensor(pairs, dtype=dtype, device=device)      # (n+m-1, 2)

    # 宽系数
    ss1 = pairs[:, 0] * pairs[:, 1]                              # size * sqrt(r)
    # 高系数
    ss2 = pairs[:, 0] / pairs[:, 1]                              # size / sqrt(r)

    # 生成形如 [xmin, ymin, xmax, ymax] 的二维数组，n+m-1行，4列
    # 这些坐标是相对于中心点的半宽半高，宽系数 0.5 → 左右坐标 ±0.25
    base_anchors = torch.stack([-ss1, -ss2, ss1, ss2], dim=1)    # (k, 4)
    base_anchors = base_anchors / 2.0

    # 取出倒数两个维度的大小
    h, w = feature_map.shape[-2:]
    # 横向锚框中心点坐标(归一化到0~1)
    shifts_x = torch.arange(w, dtype=dtype, device=device) / w
    # 纵向锚框中心点坐标(归一化到0~1)
    shifts_y = torch.arange(h, dtype=dtype, device=device) / h

    # shift_x：每一行都复制 shifts_x，表示每个网格点的横坐标
    # [[0, 1/w, 2/w, ...],
    #  [0, 1/w, 2/w, ...],
    #  ...
    # ]
    # shift_y：每一列都复制 shifts_y，表示每个网格点的纵坐标
    # [[0, 0, 0, ...],
    #  [1/h, 1/h, 1/h, ...],
    #  ...
    # ]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")  # (h,w)

    # 拉平成一维，[h*w, 4] -> (x, y, x, y)
    # 然后把 (x, y) 复制成 [x, y, x, y]
    # → 行
    # ↓ 列  [[0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #        [0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #       ]
    shifts = torch.stack([
        shift_x.reshape(-1),
        shift_y.reshape(-1),
        shift_x.reshape(-1),
        shift_y.reshape(-1)
    ], dim=1)

    # 广播相加得到最终 anchors
    # 等价于 anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    # shifts.reshape：行数仍然是h*w，第三维仍然是4，新增了1个第二维
    # 也就是shifts.reshape是(h*w,1,4)，base_anchors.reshape是(1,n+m-1,4)
    # 就能广播机制了
    # 生成的anchors是(h*w,n+m-1,4)
    # anchors的每一行都对应着同一个中心坐标，
    # 这一行的每一列都代表着一种不同的锚框大小，
    # 第三维4个通道说明每个中心坐标的每种锚框都由四个坐标点构成
    # 当然以上全是归一化的形式
    # 以及，这里当然是加法而不是惩罚（没错，就是在你认为的广播机制条件下）
    # 因为shifts的某一行是一个特征图上某个像素位置对应到原图的归一化中心点坐标 [cx, cy, cx, cy]
    # base_anchors的每一行则是某个归一化中心点的归一化偏移量
    # 总偏移后坐标 = （宽or高） * （归一化中心点 + 归一化偏移量）
    # 或者更形象来说，这里计算的偏移量是s√r和s/√r，而总偏移量是ws√r和hs/√r
    # 刚好总中心点是wcx和hcy，这里的shifts和base_anchors都提取出来了w和h项

    anchors = shifts[:, None, :] + base_anchors[None, :, :]      # (h*w, k, 4)
    anchors = anchors.reshape(1, -1, 4)                          # (1, h*w*k, 4)
    return anchors

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    device = anchor.device
    na = anchor.shape[0]
    nb = bb.shape[0]

    # 这一部分为了规避cpu，直接在gpu上计算
    # 源代码
    # jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    # 新代码
    jaccard = compute_jaccard(anchor, bb)  # (na, nb)

    assigned_idx = torch.full((na,), -1, dtype=torch.long, device=device)  # 初始全为-1
    
    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    # 这里因为改用Tensor，所以用.clone()而不是.copy()
    # 源代码
    # jaccard_cp = jaccard.copy()
    # 新代码
    jaccard_cp = jaccard.clone().to(jaccard.device)

    for j in range(nb):
        # 源代码
        # i = np.argmax(jaccard_cp[:, j])
        # 新代码
        i = torch.argmax(jaccard_cp[:, j]).item()
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行
     
    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            # 源代码
            # j = np.argmax(jaccard[i, :])
            # 新代码
            j = torch.argmax(jaccard[i, :]).item()
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    # 保证返回的张量与 anchor 在同一设备
    return assigned_idx

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns: 
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

# 这里我直接写原版的坐标变换了，更显式而且实现功能完全相同

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）

    Defined in :numref:`sec_bbox`"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    # print("anchor device:", anchor.device, "label device:", label.device)

    # 这里兼容旧版：旧版是无真实框行全为0，然后算IoU时必然为0，相当于跳过了
    # 新版这里应该也可以兼容旧数据集......所以就不改了

    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]
    
    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        MultiBoxTarget函数的辅助函数, 处理batch中的一个
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        an = anc.shape[0]
        # 上一级assign_anchor我也改了device
        assigned_idx = assign_anchor(lab[:, 1:], anc) # (锚框总数, )
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4)

        # 然后关于device不同导致出错的问题，这里也修改了一下

        device = anc.device
        cls_labels = torch.zeros(an, dtype=torch.long, device=device) # 0表示背景
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32, device=device) # 所有anchor对应的bb坐标
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0: # 即非背景
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1 # 注意要加一
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc) # (center_x, center_y, w, h)
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)

        return offset.view(-1), bbox_mask.view(-1), cls_labels
    
    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])
        
        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)
    
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)
    
    return [bbox_offset, bbox_mask, cls_labels]

@torch.no_grad()
def MultiBoxTarget_fast(anchor, label, jaccard_threshold=0.5, eps=1e-6):
    """
    更快的 MultiBoxTarget_fast 实现：
    使用 torchvision.ops.box_iou 替代 Python 循环，大幅减少 CPU 端开销。
    其余没变，只是改了一下代码写法

    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    assert anchor.ndim == 3 and label.ndim == 3
    device = anchor.device
    A = anchor.size(1)              # 锚框数
    anc = anchor[0]                 # 所有锚框的坐标(A, 4)
    center_anc = xy_to_cxcy(anc)

    batch_offsets, batch_masks, batch_cls = [], [], []
    B = label.size(0)

    for b in range(B):
        lab = label[b]
        # 过滤掉无效行（类别<0）
        valid = lab[:, 0] >= 0
        lab = lab[valid]
        if lab.numel() == 0:
            # 没有真实框：全部背景
            batch_offsets.append(torch.zeros(A * 4, device=device))
            batch_masks.append(torch.zeros(A * 4, device=device))
            batch_cls.append(torch.zeros(A, dtype=torch.long, device=device))
            continue

        gt_cls = lab[:, 0].long() + 1               # 正类从1开始
        gt_box = lab[:, 1:5]  # 真实框坐标

        # 其实这段代码和新书代码的逻辑是完全一样的，只是有点绕

        # 新书的逻辑是先为box分配anchor，
        # 然后剩余的anchor，看看和它最匹配的box是否符合阈值，符合就分配

        # 原书新代码的逻辑是先为anchor分配box，同时要符合阈值
        # 然后剩余的box再去分配anchor，并且可能覆盖之前的结果

        # 也就是说，这两种代码的逻辑都是box分配anchor的优先级最高
        # 然后anchor分配box的优先级次之，并且要符合逻辑
        # 新书的代码是在逻辑上确定优先级，旧书的新代码是以之后的结果能覆盖之前的结果确定优先级

        # IoU 计算 (A, G)，返回[N, M] 的张量，
        # 第 i 行、第 j 列元素表示 boxes1[i] 与 boxes2[j] 的 IoU
        ious = box_iou(anc, gt_box)

        # 先每个 anchor 找 IoU 最大的 GT
        # best_ious：[A] 每个锚框与所有真实框的最大 IoU。
        # best_gt：  [A] 给出最大 IoU 对应的真实框索引。
        best_ious, best_gt = ious.max(dim=1)
        assigned_idx = torch.full((A,), -1, dtype=torch.long, device=device)
        # 生成布尔张量 [A]，对于 IoU 大于等于阈值的锚框，对应位置为 True（认为是正样本）
        pos = best_ious >= jaccard_threshold
        # 把正样本锚框的位置替换为它们匹配到的真实框索引
        assigned_idx[pos] = best_gt[pos]

        # 保证每个 GT 至少分到一个 anchor
        best_anchor = ious.argmax(dim=0)
        # 例如这里bast_anchor是[2 0 1]，然后torch.arange返回是[0 1 2]
        # 那么就是assigned_idx[2]=0, assigned_idx[0]=1, assigned_idx[1]=2
        assigned_idx[best_anchor] = torch.arange(gt_box.size(0), device=device)

        # 构造标签和偏移
        cls_labels = torch.zeros(A, dtype=torch.long, device=device)
        assigned_bb = torch.zeros(A, 4, device=device)
        # 这里的pos是长为A的布尔变量
        pos = assigned_idx >= 0
        if pos.any():
            cls_labels[pos] = gt_cls[assigned_idx[pos]]
            assigned_bb[pos] = gt_box[assigned_idx[pos]]

        center_gt = xy_to_cxcy(assigned_bb)
        # 这里的公式见书本
        # 当然，书本是写了四项，这里是x y 两项算一次，w h 两项算一次，所以只有两条式子
        # 这里算x y偏移量
        offset_xy = 10.0 * (center_gt[:, :2] - center_anc[:, :2]) / (center_anc[:, 2:] + eps)
        # 同理，这里算w h偏移量，除以0.1标准差就是乘以10
        offset_wh =  5.0 * torch.log((center_gt[:, 2:] / (center_anc[:, 2:] + eps)).clamp(min=eps))
        # 拼起来；如果都没有分配锚框那就没有偏移量，掩码覆盖为0
        offsets = torch.cat([offset_xy, offset_wh], dim=1)
        # -1 表示保持原来的 N 不变，4 表示复制到 4 列
        mask = pos.unsqueeze(1).expand(-1, 4).float()

        batch_offsets.append((offsets * mask).reshape(-1))
        batch_masks.append(mask.reshape(-1))
        batch_cls.append(cls_labels)

    # 原先是一个Python list
    # 现在在新的第一维把这些小张量“堆叠”起来，形成(bn, N*4)的单个 torch.Tensor
    bbox_offset = torch.stack(batch_offsets)
    bbox_mask   = torch.stack(batch_masks)
    cls_labels  = torch.stack(batch_cls)
    return [bbox_offset, bbox_mask, cls_labels]


Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])
def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)
        
        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)
        
        iou = compute_jaccard(torch.tensor([best.xyxy]), 
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )
        
        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    # ********************************************************************************
    # 新书代码上其实有些坑，例如MultiBoxDetection偏移量编码用了log解码却没exp
    # 所以后面multibox_detection用原书的
    # 新书的MultiBoxDetection可以用作理解
    # ********************************************************************************

    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]
    
    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量
        
        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()
        
        pred_bb_info = [Pred_BB_Info(
                            index = i,
                            class_id = class_id[i] - 1, # 正类label从0开始
                            confidence = confidence[i],
                            xyxy=[*anc[i]]) # xyxy是个列表
                        for i in range(pred_bb_num)]
        
        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]
        
        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])
            
        return torch.tensor(output) # shape: (锚框个数, 6)
    
    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))
    
    return torch.stack(batch_output)



# ################################# 9.6 ############################
class PikachuDetDataset(torch.utils.data.Dataset):
    """皮卡丘检测数据集类"""
    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(data_dir, part, "images")
        
        with open(os.path.join(data_dir, part, "label.json")) as f:
            self.label = json.load(f)
            
        self.transform = torchvision.transforms.Compose([
            # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
            torchvision.transforms.ToTensor()])
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"
        
        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"], 
                         dtype="float32")[None, :]
        
        PIL_img = Image.open(os.path.join(self.image_dir, image_path)
                            ).convert('RGB').resize(self.image_size)
        img = self.transform(PIL_img)
        
        sample = {
            "label": label, # shape: (1, 5) [class, xmin, ymin, xmax, ymax]
            "image": img    # shape: (3, *image_size)
        }
        
        return sample

def load_data_pikachu(batch_size, edge_size=256, data_dir = '../../data/pikachu'):  
    """edge_size：输出图像的宽和高"""
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PikachuDetDataset(data_dir, 'val', image_size)
    

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

# ################################# 9.6.5 ############################

# 由于新书9.6-9.13仍有部分未施工完毕
# 所以9.6-9.13将会跟随原书学习，同时修改部分d2l代码
# 使用香蕉数据集，契合原书9.6-9.13

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['banana-detection'] = (
    DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

DATA_HUB['voc2012'] = (DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件

    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def read_data_bananas(is_train=True):
    # 先按照 d2l 默认规则下载
    tmp_dir = download_extract('banana-detection')

    # 希望的最终路径，在这个路径下创建文件夹
    target_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\bananas"
    os.makedirs(target_dir, exist_ok=True)

    # 如果目标目录为空，则移动一次即可
    # 将文件从默认规则下载的位置移动到目标路径
    if not os.listdir(target_dir):
        for item in os.listdir(tmp_dir):
            s = os.path.join(tmp_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 根据 is_train 决定读 bananas_train 还是 bananas_val
    sub_dir = 'bananas_train' if is_train else 'bananas_val'
    # label.csv 是标注文件，存储每张图片的边界框信息
    csv_fname = os.path.join(target_dir, sub_dir, 'label.csv')

    # 用 pandas 读入 csv，img_name 列设为索引
    # 每行形如：
    # img_name,x_min,y_min,x_max,y_max
    # 0001.png,48,240,195,371
    # ...
    # 用了set_index('img_name')，也就是label.csv的index就是img_name列
    csv_data = pd.read_csv(csv_fname).set_index('img_name')

    # for idx, row in csv_data.iterrows():
    # 每次迭代返回 (index, Series) 这样的二元组：
    # idx：该行的索引值（这里就是 img_name，因为前面用 set_index('img_name')）。
    # row：该行数据，类型是 pandas.Series，键为列名，值为该行对应的数值。
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        # img_path：拼出图片完整路径
        img_path = os.path.join(target_dir, sub_dir, 'images', img_name)
        # 读取一张图片并追加到 images 列表，返回形状[C, H, W]
        images.append(torchvision.io.read_image(img_path))
        targets.append(list(target))

    # targets转成torch.Tensor，形状为[N, 5] [batch, label xmin ymin xmax ymax]
    # 将坐标除以 256，实现归一化到 0–1 区间(数据集中图像尺寸是 256×256)
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集

    Defined in :numref:`sec_object-detection-dataset`"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True, num_workers=0, pin_memory=True)
                                            # , persistent_workers=True
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size, num_workers=0, pin_memory=True)
                                            # , persistent_workers=True
    return train_iter, val_iter

# ################################# 9.7 #########################

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(7, 4)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        # 👇 强制 y 轴更多刻度并设置显示格式
        ax = self.axes[0]
        ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 0,0.05,0.10,0.15,0.20
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.tight_layout()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_i, y_i, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_i, y_i, fmt, linewidth=1.5)
        self.config_axes()
        # 追加一次，确保每次刷新后刻度依旧
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        ax = self.axes[0]
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.tight_layout()
        display.display(self.fig)
        # display.clear_output(wait=True)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序
        Defined in :numref:`subsec_predicting-bounding-boxes-nms`
        Args:
            boxes: 各预测框的坐标
            scores: 各个框的预测概率（对了，交并比可不是预测概率）
            iou_threshold: 顾名思义
        Returns:
            所有锚框的信息, shape: (bn, 锚框个数, 6)
        """
    # 按置信度从高到低排序
    B = torch.argsort(scores, dim=-1, descending=True) # 类似于tensor([2, 0, 1, ...])
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        # 最高分框索引
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        # 这里并没有最高分框和自己IoU哈，看清楚一点代码
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 这里iou的第0个元素是原本B的第1个元素了，所以后面才会+1
        # torch.nonzero(...)：[True, False, True] → tensor([[0],[2]])
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """
    使用非极大值抑制来预测边界框
    注：这个代码其实并没有真正区分多类别，原书也没有真正区分
    可能后面多类别分类任务会先分开类别再搞，但我还是喜欢改detection代码......
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:[N, total_anchors, num_classes+1]
            offset_preds: 预测的各个锚框的偏移量, shape:[N, total_anchors * 4]，回忆一下书本里的偏移量，就是4个参数
            anchor: MultiBoxPrior输出的默认锚框, shape: [1, total_anchors, 4]
            nms_threshold: 非极大抑制中的阈值
            pos_threshold: 非背景预测的阈值
        Returns:
            output(batch_size, num_anchors, 6)

        """
    # 非极大值抑制的原理看新书

    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # cls_prob:[total_anchors, num_classes+1]
        # offset_pred:[total_anchors, 4]
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 在第 0 维度上取最大，也就是在所有类别（不包括背景）之间找一个最大值
        # 如果是torch.max(cls_prob[:], 0)，连背景都包括了，那就是预测背景了
        # conf：最大值； class_id：类别索引（从0开始）
        conf, class_id = torch.max(cls_prob[1:], 0)
        # 预测框：锚框+偏差
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 返回一列索引序列，类似于[2, 1, 6, ...]
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        # unique(return_counts=True)：
        # uniques：所有索引不重复地列一遍
        # counts ：每个索引出现的次数
        uniques, counts = combined.unique(return_counts=True)
        # 只出现了一次的必然是被NMS筛下去的锚框
        non_keep = uniques[counts == 1]
        # 先保留框，再接上被去掉的框
        all_id_sorted = torch.cat((keep, non_keep))
        # 被NMS筛下去的锚框分类为背景
        class_id[non_keep] = -1
        # 重新排列，保证一致
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        # 主要是某些确实通过NMS了，但conf很低的，也去掉吧，返回布尔列向量
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        # 既然某类的conf低，反过来就是背景的conf高
        conf[below_min_idx] = 1 - conf[below_min_idx]
        # pred_info：[num_anchors, 6]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        # 按每张图片append
        out.append(pred_info)
    return torch.stack(out)

# ################################# 9.9 #########################
def read_voc_images_old(root="../../data/VOCdevkit/VOC2012",
                    is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels # PIL image

# colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
# for i, colormap in enumerate(VOC_COLORMAP):
#     colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
def voc_label_indices_old(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop_old(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            feature, output_size=(height, width))
    
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)    

    return feature, label

class VOCSegDataset_old(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, 
                                             std=self.rgb_std)
        ])
        
        self.crop_size = crop_size # (h, w)
        features, labels = read_voc_images_old(root=voc_dir,
                                           is_train=is_train, 
                                           max_num=max_num)
        self.features = self.filter(features) # PIL image
        self.labels = self.filter(labels)     # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop_old(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        
        return (self.tsf(feature),
                voc_label_indices_old(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# ############################# 9.9.5 ##########################
def read_voc_images(voc_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"
                    ,is_train=True):
    """
    读取 PASCAL VOC 图像及分割标注
    1. 先检查本地固定目录是否已经存在数据
    2. 如果不存在，再调用 d2l.download_extract 下载并复制
    3. 最后按原逻辑读取图像和标注
    """
    target_dir = voc_dir

    # 如果目标目录不存在或内容为空，再去下载
    if not (os.path.exists(target_dir) and os.listdir(target_dir)):
        tmp_dir = download_extract('VOC2012')  # 只在需要时下载
        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(tmp_dir):
            s = os.path.join(tmp_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 根据 is_train 读取 train.txt/val.txt
    txt_fname = os.path.join(target_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        names = f.read().split()

    features, labels = [], []
    for fname in names:
        img_path = os.path.join(target_dir, 'JPEGImages', f'{fname}.jpg')
        lbl_path = os.path.join(target_dir, 'SegmentationClass', f'{fname}.png')
        features.append(torchvision.io.read_image(img_path))
        labels.append(torchvision.io.read_image(lbl_path, mode))
    return features, labels

# VOC颜色映射表
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# VOC类别名称列表
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# RGB标注图 —— 每个像素的类别索引图
def voc_colormap2label():
    """
        构建从RGB数到VOC类别索引的映射
        用途:
        给定一张 VOC 标注图 label_img（形状 H×W×3），可以快速得到类别图：
        def voc_label_indices(label_img, colormap2label):
             # label_img: (H, W, 3) uint8
            idx = (label_img[:,:,0]*256 + label_img[:,:,1])*256 + label_img[:,:,2]
            return colormap2label[idx]
    """
    # 建一个长度为 256^3 = 16777216 的一维张量，初值全 0
    # 每个可能的 RGB 组合对应一个唯一的索引位置
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    # 遍历 21 种 VOC 颜色
    for i, colormap in enumerate(VOC_COLORMAP):
        # 将一个 RGB 颜色映射到 0~16,777,215 的整数
        # 公式相当于把 (R,G,B) 当作三位 256 进制数：
        # index = R * 256^2 + G * 256 + B
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """
        将VOC标签中的RGB值映射到它们的类别索引。
        如果在CPU上使用numpy，与原逻辑一致；
        如果在GPU上使用tensor计算，避免数据搬运。
        Args:
            colormap:一张标注图的张量，形状通常是 (3, H, W)
            colormap2label:长度 256³ 的一维查表向量
        Return:
            colormap2label[idx]: 形状 (H, W)，每个元素是 0~20 的类别编号（与 VOC_CLASSES 对应）

    """
    if colormap.device.type == 'cuda:0':
        # ---- GPU: 全部使用tensor运算 ----
        colormap = colormap.long().permute(1, 2, 0)  # (H, W, 3)
        idx = (colormap[..., 0] * 256 + colormap[..., 1]) * 256 + colormap[..., 2]
        colormap2label = colormap2label.to(colormap.device)
        return colormap2label[idx]
    else:
        # ---- CPU: 保持原来的numpy逻辑 ----
        # permute(1, 2, 0)：把张量从 (C, H, W) 调整为 (H, W, C)，方便按像素访问 R/G/B
        colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        # idx 形状为 (H, W)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return colormap2label[idx]

# 随机裁剪特征(feature)和标签(label)图像
def voc_rand_crop(feature, label, height, width):
    """
        随机裁剪特征(feature)和标签(label)图像
        作用是：在数据增强时，对输入图像 feature 和对应的标注 label 同步做同样的随机裁剪
        Args:
            feature:待处理的输入图像（通常是 RGB 图），形状 (C, H, W)
            label:对应的分割标注图（每像素类别索引或 RGB），形状与 feature 高宽一致
            height:希望裁剪得到的输出图像的高度
            width:希望裁剪得到的输出图像的宽度
        Return:
            feature: 随机位置裁剪出来的图像张量，形状为 (C, height, width)
            label: 随机位置裁剪出来的标签张量，形状为 (C, height, width)

    """
    # rect：(top, left, new_height, new_width)，表示从feature中随机选取的裁剪区域
    # 其中top是从原图的最上边开始往下的像素数，left是从左往右
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    # functional.crop：按照给定的 top, left, height, width 参数裁剪图像
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# 自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # crop_size：例如 (320, 480)，后续随机裁剪时的高宽
        self.crop_size = crop_size
        # features：列表，其中每个元素都是一张原始RGB图像(C×H×W)
        # labels：列表，其中每个元素都是对应的标注图（每个像素是类别颜色的 RGB 图）
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        # filter：过滤掉尺寸小于 crop_size 的图像，确保随机裁剪时不会越界。
        # normalize_image：先把像素缩放到[0,1]，再做标准化(x-mean)/std，
        # 使用的是 ImageNet 常见均值和方差
        # colormap2label：生成颜色到类别索引的一维查表张量，用于后续把标注RGB转成类别ID
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    # 将 0–255 的 uint8 张量转为 float32，再除以 255 归一化到 0–1
    # 再用 Normalize 做标准化
    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    # 只保留高 ≥ crop_size[0] 且宽 ≥ crop_size[1] 的图
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    # 对第idx个图像随机裁剪voc_rand_crop，并对标签同步裁剪成 crop_size。
    # 标签转索引：voc_label_indices 把裁剪后的 RGB 标注图转成 (H, W) 的类别索引图
    # 每个元素都是这个像素对应的类别索引
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


# 整合所有组件，写成VOC数据集下载函数
def load_data_voc(batch_size, crop_size,
                  voc_dir=r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"):
    """
    加载VOC语义分割数据集
    1. 如果指定目录下已有数据且非空，则直接使用
    2. 否则下载并解压到默认位置
    """
    # 如果本地目录不存在或为空才下载
    if not (os.path.exists(voc_dir) and os.listdir(voc_dir)):
        print(f"[INFO] 未检测到数据集，开始下载...")
        voc_dir = download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    else:
        print(f"[INFO] 使用已有数据集：{voc_dir}")

    # DataLoader能传入VOCSegDataset类的原因：
    # DataLoader 需要的只是：
    # class MyDataset(torch.utils.data.Dataset):
    #     def __getitem__(self, idx):
    #         ... 给定一个索引 idx，返回单个样本 (feature, label)
    #     def __len__(self):
    #         ... 告诉 DataLoader 这个数据集有多少个样本

    num_workers = 0 # 因为我是高贵的windows所以不需要num_workers。

    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir),
        batch_size,
        drop_last=True,
        num_workers=num_workers
    )
    return train_iter, test_iter

# ############################# 9.11 ##########################

def train_batch_ch13(net, X, y, loss, trainer, device):
    """用多GPU进行小批量训练

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)

    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()

    # 这里返回的loss是sum起来的，所以后面会有一个metric[0] / metric[2]
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               device):
    """用单一GPU进行模型训练
        因为老子没有多GPU

    Defined in :numref:`sec_image_augmentation`"""
    # 使用 time 记录总时长
    total_time = 0.0

    # 记录曲线
    epochs, train_loss_list, train_acc_list, test_acc_list = [], [], [], []

    # 直接把模型搬到单个 device
    net = net.to(device)

    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            start_t = time.perf_counter()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            total_time += time.perf_counter() - start_t

        test_acc = evaluate_accuracy_gpu(net, test_iter, device)

        # 记录本轮结果
        epochs.append(epoch + 1)
        train_loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}: '
              f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / total_time:.1f} examples/sec on {device}')

    # ===== 用 matplotlib 画图 =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curve')
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

# ############################# 10.7 ##########################
def read_imdb(folder='train', data_root="/S1/CSCL/tangss/Datasets/aclImdb"): 
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

def get_tokenized_imdb(data):
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return torchtext.vocab.Vocab(counter, min_freq=5)

def preprocess_imdb(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed

def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'