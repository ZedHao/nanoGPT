"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'  # 模型训练输出目录，用于保存检查点、日志等文件
eval_interval = 2000  # 每训练2000步进行一次验证，监控模型在验证集上的性能
log_interval = 1  # 每1步打印一次训练日志，实时输出训练进度和损失值
eval_iters = 200  # 验证时运行的迭代次数，用于计算平均损失以评估模型泛化能力
eval_only = False  # 若设为True，脚本在首次验证后直接退出，用于测试或推理模式
always_save_checkpoint = True  # 每次验证后强制保存模型检查点，确保训练进度不丢失
init_from = 'scratch'  # 模型初始化方式：'scratch'（从头训练）、'resume'（恢复训练）或'gpt2*'（加载预训练模型）
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'  # 使用OpenWebText数据集，包含约800万网页文本
gradient_accumulation_steps = 5 * 8  # 梯度累积步数，通过累积多个小批量梯度模拟更大批量训练（等效batch_size *= 此值）
batch_size = 12  # 微批量大小，当gradient_accumulation_steps>1时，实际批量为batch_size * 累积步数
block_size = 1024  # 模型输入的最大上下文长度（token数），决定模型能处理的文本依赖范围
# model
n_layer = 12  # Transformer网络层数，决定模型深度（GPT-2 124M对应12层）
n_head = 12  # 注意力机制的头数，每层将特征分为12个并行子空间处理
n_embd = 768  # 词嵌入维度，每个token的向量表示维度（与n_head关联：768=12*64）
dropout = 0.0  #  dropout率，预训练时通常设为0，微调时可增加至0.1+以防止过拟合
bias = False  # 是否在LayerNorm和Linear层中使用偏置项，False为GPT-2原始设计
# adamw optimizer
learning_rate = 6e-4  # 最大学习率，基于Chinchilla缩放法则设定的初始值
max_iters = 600000  # 总训练迭代次数，对应约400亿token的训练规模
weight_decay = 1e-1  # 权重衰减系数，用于正则化防止过拟合
beta1 = 0.9  # AdamW优化器的一阶矩估计衰减率
beta2 = 0.95  # AdamW优化器的二阶矩估计衰减率
grad_clip = 1.0  # 梯度裁剪阈值，防止梯度爆炸，0.0表示禁用裁剪
# learning rate decay settings
decay_lr = True  # 是否启用学习率衰减，True时采用余弦退火策略
warmup_iters = 2000  # 学习率预热步数，训练初期逐步提升至最大学习率
lr_decay_iters = 600000  # 学习率衰减总步数，通常与max_iters一致
min_lr = 6e-5  # 最小学习率，为最大学习率的1/10（遵循Chinchilla原则）
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 设置数据类型，优先使用MPS支持的类型
print(torch.__version__)  # 应输出2.0.0或更高版本


# 其他配置保持不变...
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# 这段代码是 PyTorch 中实现 ** 分布式数据并行训练（Distributed Data Parallel, DDP）** 的初始化逻辑，主要功能是配置多 GPU / 多节点训练环境
# 这段代码通过环境变量检测是否启动分布式训练，并自动配置：
#
# 进程间通信与 GPU 分配
# 主进程任务调度
# 梯度累积参数调整
# 训练吞吐量计算
# ddp是一个布尔标志，决定是否启用分布式训练模式
# 当ddp=True时，代码会初始化分布式环境并配置多 GPU 训练参数
# 当ddp=False时，默认使用单 GPU 训练
master_process = False
seed_offset = 0
ddp = False
ddp_local_rank = 0
def ddp_set_muti(gradient_accumulation_steps: int) -> (bool, int):
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    return
# 这段代码是深度学习训练脚本的设备配置与环境初始化部分，
# 主要负责：创建输出目录、设置随机种子、检测计算设备、配置数据类型和精度模式，
# 以及准备数据加载路径。以下是详细解析：
def torch_init()-> (str, str,str):
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset) # 功能：设置 PyTorch 的随机种子，保证模型初始化、数据洗牌等操作的可复现性。

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'mps'
    # 优化设置，根据设备类型调整
    if torch.backends.mps.is_available():
        device = 'mps'
        dtype = 'float32'  # M1/M2上float32更稳定
        use_autocast = False  # 禁用自动混合精度
    elif torch.cuda.is_available():
        device = 'cuda'
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        use_autocast = True  # 启用自动混合精度
    else:
        device = 'cpu'
        dtype = 'float32'
        use_autocast = False  # CPU上禁用自动混合精度
    device = 'cpu'
    dtype = 'float32'
    device_type = 'mps'
    return device, dtype,device_type
# 上下文管理器
ctx = nullcontext() #if not use_autocast else torch.amp.autocast( device_type=device, dtype={'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype])
# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split, device_type):
    '''
    代码功能解析：高效数据加载与批量生成
    这段代码是深度学习模型（尤其是语言模型）的数据加载核心逻辑，主要实现了基于内存映射的高效数据读取和批量数据生成。以下是详细解析：
        内存映射：np.memmap将二进制文件直接映射到内存，无需一次性加载全部数据到 RAM，适合处理 GB 级大规模数据集（如语言模型训练数据）。
        避免内存泄漏：每次迭代重新创建memmap对象，解决了长期持有内存映射可能导致的资源释放问题（参考注释中的 Stack Overflow 解决方案）。
    :param split:
    :param device_type:
    :return:
    '''
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    '''
    . 随机索引生成
        ix = torch.randint(...)：生成batch_size个随机索引，范围是[0, len(data)-block_size]。
        目的：从长序列中随机采样block_size长度的片段，用于训练语言模型的上下文预测任务。
        2. 输入 - 目标序列构建
        输入序列 x：从索引 i 开始的block_size个 token。
        目标序列 y：从索引 i+1 开始的block_size个 token（即 x 的下一个 token 序列）。
    '''
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
def get_model_init_from(init_from):
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    return model

model = get_model_init_from(init_from)
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

def foreach_learn_stop(iter_num,running_mfu,best_val_loss,local_iter_num,t0) ->bool:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        return True

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        return True

while True:
    res = foreach_learn_stop(iter_num,running_mfu,best_val_loss,local_iter_num,t0)
    if res:
        break

if __name__ == '__main__':
    destroy_process_group()
