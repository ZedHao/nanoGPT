一、推荐入门级开源项目（帮你理解前向传播、反向传播等基础概念）
这些项目代码量小、逻辑清晰，能帮你直观理解 “模型如何学习”，以及梯度、传播、优化等核心概念：
# 1. PyTorch 官方教程（最推荐）
   官网：PyTorch Tutorials https://docs.pytorch.org/tutorials/


Learning the Basics 系列：从张量操作到简单神经网络（如 MNIST 手写数字识别），手把手教你搭建第一个模型。
Neural Network Basics：用代码实现 “前向传播”“反向传播”“参数更新” 的完整流程，甚至会手动推导梯度（帮你理解 PyTorch 的autograd自动求导原理）。
为什么适合：官方教程最权威，例子极简（比如用 50 行代码实现一个两层神经网络），能帮你搞懂 “输入数据→模型计算→损失→更新参数” 的闭环。
# 2. pytorch-examples（GitHub 开源） https://github.com/pytorch/examples
   地址：pytorch/examples

包含各种经典任务的极简实现：
mnist：用 CNN 识别手写数字（理解卷积层、池化层的前向传播）。
word_language_model：用 RNN 生成文本（理解循环神经网络的反向传播，以及梯度裁剪 ——RNN 很容易出现梯度爆炸，这里会用到torch.nn.utils.clip_grad_norm_，正好对应你困惑的 “梯度裁剪”）。
梯度裁剪的直观理解：在 RNN 例子中，你可以故意不裁剪梯度，观察模型是否发散（损失突然变大），再对比裁剪后的效果，瞬间明白它的作用。
# 3. fastai的入门课程配套代码
   地址：fastai/course-v3 https://github.com/fastai/course-v3

对应《Deep Learning for Coders with fastai and PyTorch》一书，代码注重 “直观理解”，比如：
用动画展示 “梯度下降” 过程（参数如何一步步逼近最优解）。
用简单模型（如线性回归、决策树）对比神经网络，帮你理解 “为什么需要反向传播”。
适合点：fastai 的理念是 “先会用，再深究原理”，代码里会用通俗的注释解释 “梯度累积”（比如 “当 GPU 显存不够时，如何用多个小批次模拟大批次训练”）。
4. tinygrad（极简深度学习框架）
   地址：tinygrad/tinygrad https://github.com/tinygrad/tinygrad

一个极简的深度学习框架（只有几千行代码），手动实现了张量、自动求导、优化器等核心组件。
如果你想搞懂 “PyTorch 的backward()到底在做什么”，可以看它的autograd实现 —— 代码里会用递归遍历计算图，手动传播梯度，比 PyTorch 的底层 C++ 代码更容易看懂。