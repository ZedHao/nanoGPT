# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
# 训练配置与输出设置
out_dir = 'out-shakespeare-char'  # 训练输出目录，保存模型检查点和日志
eval_interval = 200  # 每训练200步进行一次验证，频繁验证有助于监控过拟合
eval_iters = 200  # 验证时使用的迭代次数，用于计算平均验证损失
log_interval = 10  # 每10步打印一次训练日志，避免输出过于频繁
# we expect to overfit on this small dataset, so only save when val improves
# 由于在小数据集上容易过拟合，仅在验证损失提升时保存检查点
always_save_checkpoint = False

# 是否使用WandB进行实验跟踪（可通过命令行覆盖配置）
wandb_log = False
wandb_project = 'shakespeare-char'  # WandB项目名称
wandb_run_name = 'mini-gpt'  # 当前运行的名称

dataset = 'shakespeare_char'  # 使用莎士比亚文本的字符级数据集
gradient_accumulation_steps = 4  # 梯度累积步数，等效于增大batch_size
batch_size = 8  # 每次训练的样本数（实际批量大小 = batch_size * gradient_accumulation_steps）
block_size = 128  # 上下文长度，模型一次处理的最大字符数（用于自回归）
# baby GPT model :)
n_layer = 4  # Transformer层数，相当于GPT的“深度”
n_head = 4  # 注意力头数，决定并行注意力机制的能力
n_embd = 256  # 嵌入维度，即每个token的向量表示维度
dropout = 0.2  # Dropout率，用于防止过拟合（训练时随机“丢弃”部分神经元）

learning_rate = 1e-3  # 初始学习率，小模型可使用较大值
max_iters = 1000  # 最大训练迭代次数
lr_decay_iters = 1000  # 学习率衰减步数，通常等于max_iters
min_lr = 1e-4  # 最小学习率，衰减的下限（通常为初始学习率的1/10）
beta2 = 0.99  # Adam优化器的beta2参数（控制二阶矩估计的指数衰减率）
warmup_iters = 100  # 学习率预热步数，训练初期逐步提高学习率

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
print("----------train_shakespeace_char---------")