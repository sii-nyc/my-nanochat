# 模型架构：形成整体代码图景

**记录日期**：2026-10-05；**状态**：整体理解已完成。

这一阶段承接 tokenizer 学习，以当前仓库的 `nanochat/gpt.py` 为准，梳理了 token ID 到 logits、训练损失及生成 token 的路径，并追踪 `scripts/base_train.py` 如何构造实际模型配置。目标是建立可以接续预训练入口阅读的整体认识；对仓库的专门设计暂不做独立收益研究。

## 起点与工作脉络

Tokenizer 阶段已解释文本怎样变为 token ID，但尚不清楚这些 ID 在模型中怎样被处理，也不能仅凭 `GPTConfig` 的默认值判断预训练入口实际使用的模型。因此先沿 `GPT.forward()` 分清输入、Block 和输出，再分别讨论 Attention、MLP、RoPE、归一化、残差路径、初始化和生成，最后回到训练脚本的配置构造。

目前形成的主线是：数据加载器提供错开一位的 `idx` 与 `targets`；模型把 ID 变为 embedding，经过归一化、Smear 和多个 decoder-only Block，最后经 Backout、归一化和独立输出头得到各位置的词表 logits，使用下一 token 目标计算交叉熵。Block 的核心仍是因果注意力与逐位置 MLP 各自的 pre-norm 残差更新。RoPE 处理 Q/K 的位置信息，因果与滑动窗口约束可读取的位置。详细形状、路径和代码入口集中在[模型架构专题文档](../../model_architecture.md)。

这次阅读也区分了主干与仓库的具体选择：Smear 提前混合相邻 token，部分层的 value embedding 注入 V，逐层标量把初始表示重新混入残差流，Backout 在输出前减去中途表示；此外还有 QK norm、`ReLU²`、滑动窗口、未绑定的 `lm_head` 和 logit softcap。它们有清晰的代码位置与设计意图，但现阶段不从现有综合实验推断各自的独立效果。

## 阶段判断与接续

`base_train.py` 以 `--depth` 为主要规模入口，按 `depth × aspect_ratio` 并向 `head_dim` 的倍数取整来确定宽度，再确定头数；真实词表大小取自运行环境默认路径下加载的 tokenizer。脚本默认 `depth=20` 对应 `20` 层、`1280` 维、`10` 个 Q 头和 `10` 个 KV 头，而非 `GPTConfig` 类默认的 12 层。该入口令 Q/KV 头数相等，虽有 GQA 支持，默认并未启用。模型配置会打印，模型配置与命令行参数都会写入检查点元数据。

维护者决定跳过本地模型运行检查；本阶段也未启动预训练，因而没有新的损失、性能或质量结果。下一步应阅读 `base_train.py` 的数据、优化和训练循环，明确使用哪个 tokenizer、服务器命令与配置记录方式，再设计预训练实验。源码中短窗口大小及部分初始化参数的注释与表达式不符，后续按实际代码和值核对。

## 资料入口

- [模型架构专题文档](../../model_architecture.md)：数据流、组件分工、专门设计与配置来源。
- 主要代码：[`nanochat/gpt.py`](../../../nanochat/gpt.py)、[`scripts/base_train.py`](../../../scripts/base_train.py)、[`nanochat/dataloader.py`](../../../nanochat/dataloader.py)、[`nanochat/engine.py`](../../../nanochat/engine.py)。阅读依据为提交 [`f0baf17`](https://github.com/sii-nyc/my-nanochat/commit/f0baf176014dfeb1bd90626e22e4049404d0cbdd)。
