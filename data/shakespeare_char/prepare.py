"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
1. 主要功能是为字符级语言建模（Character-Level Language Modeling）准备数据集，
2. 对对其进行预处理（字符映射、划分训练集和验证集），并将处理后的数据保存为模型可直接使用的格式。
3. 它完成了从 “原始文本” 到 “模型可直接使用的数字序列” 的转换，并通过划分训练 / 验证集、保存元信息，
4.为后续模型训练和文本生成做好了准备。整个过程的核心是 “字符→数字” 的映射，这是字符级建模的基础（与基于词或子词的建模方式不同）
5.train.bin	训练集文本的整数序列（二进制）	模型训练时的输入数据
val.bin	验证集文本的整数序列（二进制）	模型训练中验证性能，调整超参数
meta.pkl	字符 - 数字映射关系等元信息	训练 / 生成时编码输入、解码模型输出结果
6.
"""
import os
import pdb
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r',encoding='utf-16') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")
'''
 构建字符词汇表（Vocabulary）
提取唯一字符：通过 set(data) 获取数据中所有出现过的唯一字符，再排序得到 chars 列表（示例输出包含大小写字母、标点、空格等，共 65 个字符）。
定义词汇表大小：vocab_size 即唯一字符的数量（示例中为 65）。
'''
# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))

print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
'''
为了让模型能够处理文本（模型只能接受数字输入），需要将字符转换为整数：
stoi：字典，键为字符（如 'A'），值为对应的整数（如 10），用于 “编码”（字符串→数字列表）。
itos：字典，键为整数（如 10），值为对应的字符（如 'A'），用于 “解码”（数字列表→字符串）。
定义 encode 函数：将字符串转换为整数列表（例如，"abc" → [3,4,5]）。
定义 decode 函数：将整数列表转换回字符串（例如，[3,4,5] → "abc"）。
'''
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
# encode both to integers
'''
照 9:1 的比例拆分数据：
train_data：前 90% 的文本，用于模型训练。
val_data：后 10% 的文本，用于验证模型性能（避免过拟合）
'''
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
'''
将整数序列转换为 np.uint16 类型的 numpy 数组（uint16 足够存储 0-65 的整数，节省空间）。
用 tofile 方法将数组保存为二进制文件（train.bin 和 val.bin），这种格式加载速度快，适合模型训练时读取。
保存元信息（meta.pkl）：包含 vocab_size、itos、stoi，方便后续训练或生成时复用编码 / 解码规则。

train.bin	训练集文本的整数序列（二进制）	模型训练时的输入数据
val.bin	验证集文本的整数序列（二进制）	模型训练中验证性能，调整超参数
meta.pkl	字符 - 数字映射关系等元信息	训练 / 生成时编码输入、解码模型输出结果

'''
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
"""
pickle 是 Python 的序列化工具，可以将 Python 对象（如字典、列表等）直接保存为二进制文件，之后可以完整地读取并恢复为原来的对象。
这里用 pickle.dump(meta, f) 将 meta 字典序列化到文件中，确保后续读取时能原样恢复 vocab_size、itos、stoi 这三个关键数据。
"""
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
