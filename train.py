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
import pdb
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
def ddp_set_muti(gradient_accumulation_steps: int) -> (bool, int):
    '''
    这个函数 ddp_set_muti 是用于配置分布式训练（DDP，Distributed Data Parallel）环境的工具函数，
    主要作用是根据是否启用分布式训练，初始化进程组、设置设备、调整梯度累积参数，并返回主进程标识和种子偏移量。
    核心功能
    在大规模模型训练中，单卡算力可能不足，因此需要用多卡（或多进程）分布式训练（DDP）。这个函数的作用就是：

    判断当前是否处于分布式训练环境；
    若启用 DDP，初始化分布式进程组、配置设备和进程参数；
    调整梯度累积步数以适配分布式场景；
    计算每轮迭代处理的总 token 数，方便监控训练效率。
    :param gradient_accumulation_steps:
    :return:
    '''
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
    return master_process, seed_offset

def torch_init()-> (str, str,str):
    '''
    # 这段代码是深度学习训练脚本的设备配置与环境初始化部分，
    # 主要负责：创建输出目录、设置随机种子、检测计算设备、配置数据类型和精度模式，
    # 以及准备数据加载路径。以下是详细解析：
    1.为什么需要随机种子？
        模型参数的初始值（权重矩阵随机初始化）；
        数据加载时的随机打乱（get_batch 中用 torch.randint 随机采样数据）；
        Dropout 层的随机失活（训练时随机丢弃部分神经元）；
        优化器中的随机梯度更新（如 Adam 中的动量随机估计）。
    2. 为什么要 “固定” 随机种子？
        固定随机种子的本质是让这些随机操作的结果可预测，从而带来两个关键好处：
        实验可复现：同一代码、同一参数，无论何时、何地运行，都能得到完全相同的训练过程和结果（例如，损失曲线、模型精度完全一致）。这对调试代码、对比不同实验（如调整学习率、模型结构）至关重要 —— 如果结果不可复现，就无法判断性能变化是来自参数调整还是随机因素。
        分布式训练一致性：在分布式训练（DDP）中，多个进程需要处理不同的数据分片，但初始化逻辑（如模型参数）必须完全同步。代码中 seed_offset = ddp_rank 确保每个进程的种子不同但固定（1337 + 0、1337 + 1...），既避免了不同进程的数据采样重复，又保证了整体随机性的可控性。
    1. 不设随机种子：第一次可能抽到 5 张猫、5 张狗；第二次可能抽到 8 张猫、2 张狗。数据不一样，模型学出来的 “判断标准” 可能就不一样 —— 第一次可能更擅长认狗，第二次可能更擅长认猫，两次训练结果差异很大，你说不清是模型改得好还是运气好。
    2. 设了随机种子：不管你跑多少次，“随机选图” 的顺序都是固定的 —— 第一次抽哪 10 张，第二次还抽哪 10 张。这样一来，如果你改了模型的某个参数（比如学习率），两次结果的差异就能明确归因为 “参数改得好不好”，而不是 “抽到的数据不一样”
       for i in range(10):
        random.seed(1)
        print(random.randint(0, 10))
        每次都会输出2
    :return:
    '''
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

def get_model_init_from(init_from) :
    """
    1. 这段代码的核心功能是根据不同的初始化方式（从头训练、从断点恢复、基于预训练模型）创建并配置 GPT 模型，
    2. 是深度学习训练中 “模型初始化” 的关键逻辑。我们逐部分解析：
    3. 初始化方式总览
        代码通过 init_from 参数的不同值，决定模型的初始化方式，主要有 3 种：

        init_from == 'scratch'：从头开始训练（全新模型）
        init_from == 'resume'：从之前保存的断点（checkpoint）恢复训练
        init_from.startswith('gpt2')：基于 OpenAI 预训练的 GPT-2 模型（如gpt2-small、gpt2-large）继续训练
    :param init_from:
    :return:
    """
    checkpoint = None
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        print("scratch 从头开始训练（全新模型），选定gpt模型", gptconf)
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
        print("resume 从之前保存的断点（checkpoint）恢复训练, 选定gpt模型", gptconf)

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
        print("startswith('gpt2')：基于 OpenAI 预训练的 GPT-2 模型（如gpt2-small、gpt2-large）继续训练, 选定gpt模型")
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    return model,checkpoint
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(device_type):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,device_type)
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

def foreach_learn_stop(iter_num,optimizer,device_type,master_process,best_val_loss):
    '''
    这段代码是模型训练的 “引擎”，完整实现了：

    数据加载与预处理（get_batch）；
    学习率动态调整；
    模型前向 / 反向传播（含梯度累积、混合精度）；
    参数更新与梯度裁剪；
    定期评估与 Checkpoint 保存；
    分布式训练支持（DDP）；
    训练日志与效率监控。
    这段代码是深度学习模型（如 GPT）的核心训练循环，包含了从数据加载、模型训练、
    参数更新到日志记录、早停判断的完整流程，同时支持分布式训练（DDP）和混合精度训练。我们分模块解析：
    1. 小孩：相当于 “模型”（他的大脑就是一个待训练的 “算法”）。
        题目和答案：相当于 “训练数据”（比如(3+5, 8)、(2+7, 9)）。
        你的要求：相当于 “损失函数”（答对得 100 分，答错扣分数，分数越高低越好）。
        你的教学方法：相当于 “优化器”（比如错了就讲思路，再给类似题练习）。
    2。 核心训练循环（对应代码中的while True循环）
        循环的目的：通过反复练习→纠错→改进，让小孩（模型）逐渐学会正确解题。
        每一轮循环（每一次练习）的步骤如下：
        步骤 1：给小孩一道题（对应get_batch获取数据）
        你随机选一道题：“3+5=？”（相当于代码中从训练集中取一个批次的X），并知道正确答案是 8（相当于Y）。
        步骤 2：让小孩答题（对应 “前向传播”model(X, Y)）
        小孩第一次瞎猜：“等于 6？”（相当于模型根据当前参数计算出的结果logits）。
        步骤 3：判断对错，指出差距（对应 “计算损失”loss）
        你说：“错了，正确答案是 8，差 2 分”（损失loss就是 “猜测结果” 和 “正确答案” 的差距，这里差距是 2）。
        步骤 4：教他怎么改（对应 “反向传播”backward()）
        你告诉他：“3+5 就是从 3 往后数 5 个数：4、5、6、7、8，所以是 8”（相当于模型根据损失计算 “参数应该怎么调整”）。
        步骤 5：让他记住改进（对应 “参数更新”optimizer.step()）
        小孩调整自己的思路（比如记住 “3+5=8”，或者学会 “往后数” 的方法）（相当于模型更新权重参数，让下次计算更接近正确答案）。
        步骤 6：换一道题重复练习（循环的意义）
        你再给一道题 “2+7=？”，小孩根据刚才的经验回答 “9”（对了），或者再错（比如答 8），重复步骤 2-5。


        循环步骤	代码操作	作用总结
        取数据	X, Y = get_batch('train')	给模型喂 “练习题”
        前向传播	logits, loss = model(X, Y)	让模型 “做题”，算 “错题数”
        反向传播	loss.backward()	分析 “为什么错”，找改进方向
        参数更新	optimizer.step()	按改进方向调整模型 “解题思路”
        定期评估	estimate_loss() + 保存 checkpoint	检查学习效果，存档进度
    :param iter_num:
    :param optimizer:
    :param device_type:
    :param master_process:
    :param best_val_loss:
    :return:
    '''
    # 关键变量初始化
    X, Y = get_batch('train', device_type)  # 	1. 给模型喂 “练习题” # 获取第一批训练数据（输入X和目标Y）
    t0 = time.time()  # 记录当前时间，用于计算每步耗时
    local_iter_num = 0  # 本地迭代次数（当前进程内的计数）
    raw_model = model.module if ddp else model  # 解包DDP模型（如果是分布式训练，DDP会包装模型，需获取原始模型）
    running_mfu = -1.0  # 用于跟踪模型的计算效率（MFU，模型FLOPS利用率）
    while True:
        # 根据当前迭代次数计算学习率（如余弦退火+预热） 更新优化器的学习率
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 模型评估与 Checkpoint 保存
        # evaluate the loss on train/val sets and write checkpoints
        # 作用：定期评估模型性能（避免过拟合），并保存最优模型状态（方便中断后恢复训练）。
        # 仅主进程（master_process）执行，避免分布式训练中重复评估 / 保存。
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(device_type)
            print(f"master_process 进行中 step {iter_num}:# 每训练eval_interval{eval_interval}步进行一次验证，监控模型在验证集上的性能  train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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
        #  仅评估模式（不训练）
        if iter_num == 0 and eval_only:
            print("--------单纯测试完成--------")
            return

        # 前向传播 + 反向传播（核心训练步骤）
        # 梯度累积：将多个微批次的梯度合并，模拟大批次训练
        # 梯度累积：当 GPU 显存不足时，用多个小批次（micro_step）的梯度累积，等效于大批次训练（如 8 个微批次，每个 batch=12，等效 batch=96）。
        # 分布式优化：DDP 模式下仅最后一步同步梯度，减少通信开销。
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # 分布式训练：仅最后一个微步骤同步梯度（提高效率）
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:  # 混合精度训练上下文（如float16的自动精度转换）
                # 前向传播 2. 让模型 “做题”，算 “错题数”
                print("2. 让模型 “做题”，算 “错题数")
                logits, loss = model(X, Y)  # 前向传播：计算输出和损失
                loss = loss / gradient_accumulation_steps  # 缩放损失（适应梯度累积）
            # 异步获取下一批数据（利用GPU计算时的空闲时间，加速流程）
            print("1. 给模型喂 “练习题")
            X, Y = get_batch('train', device_type)
            # 反向传播：计算梯度（混合精度训练时用scaler缩放梯度，避免下溢）
            print("3.分析 “为什么错”，找改进方向")
            scaler.scale(loss).backward()
        # clip the gradient
        # 梯度裁剪：防止梯度爆炸（当grad_clip>0时） 梯度裁剪是 Transformer 模型训练的常见操作，避免梯度过大导致参数更新不稳定。
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)  # 取消梯度缩放（用于裁剪）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 裁剪梯度到阈值内

        # 更新参数：用优化器和scaler（混合精度训练时）
        print("按改进方向调整模型 “解题思路")
        scaler.step(optimizer)  # 根据梯度更新参数
        scaler.update()  # 调整scaler的缩放系数（适应下一轮）

        # 清零梯度（释放内存）
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
            print(f"第iter {iter_num}步迭代log_interval 完成{log_interval}步打印一次训练日志，实时输出训练进度和损失值  : loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            return

if __name__ == '__main__':
    ## 参数初始化
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    print("---------参数初始化开始--------")
    out_dir = 'out'  # 模型训练输出目录，用于保存检查点、日志等文件
    eval_interval = 500  # 每训练2000步进行一次验证，监控模型在验证集上的性能
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
    max_iters = 2000  # 总训练迭代次数，对应约400亿token的训练规模
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
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # 设置数据类型，优先使用MPS支持的类型
    print(torch.__version__)  # 应输出2.0.0或更高版本


    # 其他配置保持不变...
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    print("---------参数加载和覆盖configurator.py--------")
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    master_process = False
    seed_offset = 0
    ddp = False
    ddp_local_rank = 0
    master_process, seed_offset = ddp_set_muti(gradient_accumulation_steps)
    print("---------参数初始化结束--------" ,  master_process, seed_offset)
    # 上下文管理器
    '''
    nullcontext()：一个空的上下文管理器，相当于 “不做任何特殊处理”（默认使用单精度训练）。
    注释部分是完整逻辑：如果启用混合精度（use_autocast=True），则使用 torch.amp.autocast 创建上下文，
    自动在计算中混合 float16/bfloat16（低精度）和 float32（高精度），加速训练并减少显存占用；否则使用空上下文（纯 float32 训练）。
    这里当前代码强制使用了 nullcontext()（可能是为了调试或兼容某些设备），实际使用时可根据 use_autocast 动态切换。
    '''
    ctx = nullcontext() #if not use_autocast else torch.amp.autocast( device_type=device, dtype={'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype])
    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    device, dtype,device_type = torch_init()
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    print("---------torch初始化【创建输出目录、设置随机种子、检测计算设备、配置数据类型和精度模式】--------" ,device, dtype,device_type)

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
    print("---------get_model_init_from[根据不同的初始化方式（从头训练、从断点恢复、基于预训练模型）创建并配置 GPT 模型]--------")
    model, checkpoint = get_model_init_from(init_from)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    # 混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    # optimizer
    # 配置 AdamW 优化器
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if (init_from == 'resume' and checkpoint is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])


    # compile the model
    # 通过 PyTorch 2.0 引入的torch.compile对模型进行优化，提升训练速度。
    # 原理：将模型的计算图转换为更高效的机器码（类似 “提前编译”），减少 Python 解释器的开销，优化 CUDA 内核调用等底层操作。
    # 效果：通常能提升 10%-50% 的训练速度（视模型和设备而定），但首次编译需要 1-2 分钟（类似 “预热”）。
    # 备份：unoptimized_model保留原始模型，方便后续可能的调试或对比。
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    # 将模型封装为DistributedDataParallel（DDP），支持多 GPU / 多进程分布式训练。
    # 场景：当使用多卡训练时（通过torchrun启动多个进程），ddp会被设为True。
    # 原理：DDP 会自动将数据拆分到不同 GPU，各自计算梯度后同步更新，实现多卡并行训练（提升算力利用率）。
    # device_ids=[ddp_local_rank]：指定当前进程使用的 GPU 编号（确保每个进程绑定到正确的设备）。
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    print("---------ddp-master_process--------",ddp,master_process)

    # foreach_learn_stop 参数初始化
    # 用：初始化 Weights & Biases（W&B）日志工具，记录训练过程中的指标（损失、学习率等）。
    # wandb_log：控制是否启用 W&B（通常设为True用于实验跟踪）。
    # master_process：确保只有主进程（分布式训练中的rank=0进程）初始化 W&B，避免多进程重复日志。
    # 功能：记录的指标会同步到 W&B 官网，方便可视化训练曲线、对比不同实验（如调整超参数后的效果）。
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    print("---------foreach_learn_stop--------",iter_num,optimizer,device_type)

    foreach_learn_stop(iter_num,optimizer,device_type,master_process,best_val_loss)
    #  # 训练结束后，销毁分布式进程组（释放资源）
    if ddp:
            destroy_process_group()
