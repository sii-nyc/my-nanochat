# nanochat 模型架构概览

本文基于当前仓库的 [`nanochat/gpt.py`](../nanochat/gpt.py) 和 [`scripts/base_train.py`](../scripts/base_train.py)，记录第二阶段形成的整体认识。目标是能沿实际代码解释模型的数据流、主要组件和训练入口如何确定模型大小；本阶段不评价各项特殊设计的独立收益，也不设计或启动预训练实验。

## 从 token ID 到训练损失

记批大小为 `B`、序列长度为 `T`、模型宽度为 `C`、真实词表大小为 `V`。训练数据加载器先从每行 `T+1` 个 token 构造错开一位的 `inputs` 和 `targets`：`inputs = row[:-1]`，`targets = row[1:]`，见 [`nanochat/dataloader.py`](../nanochat/dataloader.py)。`GPT.forward(idx, targets)` 本身不移动标签。

```text
idx (B,T)
  → token embedding、RMSNorm、Smear (B,T,C)
  → 多个 Transformer Block (B,T,C)
  → Backout、RMSNorm、独立 lm_head (B,T,padded_vocab_size)
  → 裁掉补齐的词表项、logit softcap (B,T,V)
  → 与 targets (B,T) 计算交叉熵损失
```

不给 `targets` 时，[`GPT.forward()`](../nanochat/gpt.py) 返回每个位置的 logits；给出 `targets` 时，默认返回有效目标位置的平均交叉熵，目标 `-1` 被忽略。位置 `t` 的 logits 用来预测下一 token；因果注意力阻止它读取未来位置。词表补齐到 64 的倍数只是计算实现，输出在损失或生成前裁回真实 `V`。

## Transformer 主干

每个 [`Block`](../nanochat/gpt.py) 依次执行 `x + Attention(norm(x))` 和 `x + MLP(norm(x))`。这是带因果自注意力、逐位置前馈网络和残差连接的 decoder-only Transformer 主干；这里采用 pre-norm。Attention 在允许的过去位置之间交换信息，MLP 则对每个位置独立地做 `C → 4C → C` 变换，中间激活为 `ReLU(x)²`。

Attention 将输入分别投影为 Q、K、V。Q、K 经 RoPE 旋转以让匹配分数包含相对位置信息，再分别做 RMSNorm，并各乘 `1.2`。注意力权重由 Q/K 的匹配分数、因果掩码和该层的窗口共同决定，用于加权汇总 V。输出合并各查询头后，经 `c_proj` 回到 `(B,T,C)`；实际计算使用 FA3，或通过 [`nanochat/flash_attention.py`](../nanochat/flash_attention.py) 回退到 PyTorch SDPA。

[`GPTConfig`](../nanochat/gpt.py) 支持让 K/V 头少于 Q 头的 GQA，但 `base_train.py` 构造配置时令 `n_kv_head=n_head`，所以该训练入口默认没有启用头共享。RoPE 是此模型的位置编码选择；因果掩码仍由注意力计算单独施加。这里的 `norm()` 是无可学习缩放参数的 RMSNorm，只沿张量最后一维归一化，不跨 token 或 batch 计算统计量。

## 本仓库值得辨认的设计

这些机制出现在当前实现中，不应与 decoder-only Transformer 的基本结构混为一谈：

| 机制 | 在数据流中的作用 |
| --- | --- |
| Smear | Block 之前，将前一 token 的归一化 embedding 按当前 token 决定的门控量混入当前位置，提供邻接 token 信息；其总体系数初始化为 0。 |
| Value embedding | 隔层从 token ID 查询独立的表，按每个位置、KV 头的门控量加到 V 上；同一层的 Q/K 不直接加入这张表。 |
| `x0` 与残差系数 | 每层进入 Block 前，用可学习标量混合当前表示与最初经过 Smear 的表示；与 Block 内的两次残差相加是不同路径。 |
| Backout | 保存中途一层的输出，全部 Block 结束后减去它的可学习倍数，再做最终归一化。 |
| 滑动窗口 | 按 `window_pattern` 在层间安排短、长因果注意力窗口；最后一层强制使用长窗口。 |
| 独立输出头与 softcap | `lm_head` 不与输入 embedding 共用权重；输出 logits 转为 fp32 后，经 `15·tanh(logit/15)` 平滑限制幅度。 |

Smear 的目标是让相邻 token 信息在第一层之前融合，以提供类似 bigram 的输入；它曾在 [`modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt) 中出现，Karpathy 在 [`dev/LEADERBOARD.md`](../dev/LEADERBOARD.md) 说明了此处采用它的背景。当前综合改动的运行结果不能直接当作 Smear 单独收益的证据。

初始化集中在 `GPT.init_weights()`：输入 embedding、输出头和内部投影采用不同的初值；Attention 与 MLP 的输出投影初始化为零，因此单独的 Block 初始不改变输入，但层前的 `x0` 混合和最终 Backout 仍会作用。模型先在 `meta` 设备建立形状，再 `to_empty()` 分配存储并调用 `init_weights()`。Smear、Value embedding、Backout、各层标量和 RoPE 缓冲也在这一流程中明确初始化。

生成时，[`GPT.generate()`](../nanochat/gpt.py) 反复对当前整段 token 运行 `forward()`，只取末位置 logits，按 `top_k`、温度和采样或贪心规则选出新 token，并追加到输入。它是易读的无缓存路径；[`Engine.generate()`](../nanochat/engine.py) 则预填充提示，缓存各层 K/V 和 Smear 所需的前一 embedding，之后逐 token 解码。

## 训练入口如何确定模型配置

[`scripts/base_train.py`](../scripts/base_train.py) 的架构参数默认值为 `depth=20`、`aspect_ratio=64`、`head_dim=128`、`max_seq_len=2048`、`window_pattern="SSSL"`。主要计算是：

```text
base_dim = depth × aspect_ratio
model_dim = 向上取整到 head_dim 的倍数(base_dim)
num_heads = model_dim / head_dim
GPTConfig(sequence_len=max_seq_len, vocab_size=所加载 tokenizer 的词表大小,
          n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
          n_embd=model_dim, window_pattern=window_pattern)
```

因此，不覆盖架构参数时得到 `20` 层、`1280` 维、`10` 个 Q 头和 `10` 个 KV 头、每头 `128` 维；例如只指定 `--depth=12` 则是 `12` 层、`768` 维、`6/6` 头。`GPTConfig` 类自身的默认 `n_layer=12` 不决定 `base_train.py` 的默认模型。

词表大小来自 `get_tokenizer()` 读取的 `<base_dir>/tokenizer/tokenizer.pkl`；`base_dir` 默认是本仓库的 `training/`，也可由 `NANOCHAT_BASE_DIR` 指定。训练入口没有选择实验 tokenizer 文件名的参数，因此未来运行前需明确实际加载的词表及其版本。脚本会打印 `model_config`，并把它与命令行参数 `user_config` 一起写入检查点元数据；服务器实验仍应记录代码提交、命令、数据版本和结果。批大小、训练时长和学习率的推导在模型建立之后进行，不改变上述架构字段。

阅读源码时有两处旧注释与实现不一致：默认 `sequence_len=2048` 时，短窗口表达式算出 `512`，而注释写 `768`，训练参数帮助文字还写 `S=half context`；`init_weights()` 说明中的 `wte std=1.0`、`c_fc std=1/√C` 与实际的 `0.8`、`0.4/√C` 不同。本文均以执行表达式为准。

## 当前边界

本阶段完成了架构与配置路径的代码阅读，没有运行本地模型检查，也没有启动正式训练。下一步应继续理解预训练入口中的数据、优化与训练循环，然后固定 tokenizer 和可追溯的服务器运行配置，再设计预训练实验。
