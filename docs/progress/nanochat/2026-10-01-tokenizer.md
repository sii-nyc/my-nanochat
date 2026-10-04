# Tokenizer：原理理解与复现实验

**记录日期**：2026-10-01；**更新日期**：2026-10-04

**状态**：已完成。

这一阶段从理解 nanochat 的 byte-level BPE 实现出发，完成了五组 tokenizer 的从头训练、对比评估和训练语料审计。它让我们掌握了文本进入模型之前的处理路径，也为后续选择基础模型词表提供了可复查的依据。

## 起点

要从 nanochat 训练自己的聊天模型，首先需要弄清文本如何变为模型使用的 token ID，以及词表如何影响序列长度和后续训练。因此我们先阅读 tokenizer 的原理与实现，再设计对比实验，分别考察训练文本规模与词表大小的影响。

## 工作脉络与认识

我们沿着实际代码路径梳理了 ClimbMix Parquet 文档的读取与截断、正则预切分、UTF-8 字节基础词表、BPE 合并规则，以及训练结果如何交由 tiktoken 编码和解码；同时确认了字符预算、特殊 token 和 token 字节长度缓存各自的作用。实际训练与编码、解码往返检查验证了这条路径，原理和使用方式整理在 [tokenizer 专题文档](../../tokenizer.md)。

随后在相同数据顺序与评估样本下从头训练五个词表：固定 32K 词表比较 1B、2B、3B 训练字符，固定 2B 字符比较 16K、32K、64K 词表。在留出验证分片的抽样文本上，增加 32K 词表的训练字符时，BPT 仅从 **4.6892** 变为 **4.6901**；固定 2B 字符而将词表从 16K 扩到 64K 时，BPT 从 **4.3946** 增至 **4.8791**。因此，在这次评估范围内，词表大小对 token 数的影响比增加训练字符数更明显。[实验分析](../../experiments/tokenizer/tokenizer-20261003/analysis.md)保留了完整设置、指标、切分样例及评估范围。

观察韩文等文本的切分时，我们意识到需要先确认训练语料的语言构成。补充审计核对了本次实际读取的前 14 个训练分片；在 1B、2B、3B 截止点，被识别为英语主要语言的字符占比分别为 **99.953%、99.952%、99.955%**。这说明本轮主要考察英语语料训练出的 tokenizer；其他语言的切分样例适合作为压力测试。这里的比例来自逐篇主要语言预测，只覆盖本次使用的分片，不能代表整套 ClimbMix 的语言分布，详见[语料审计](../../experiments/tokenizer/tokenizer-20261003/analysis.md#训练语料的语言构成)。

## 阶段结果与接续

Tokenizer 的原理学习、代码理解和本轮复现实验已经完成。2B/32K 可以作为后续基础模型训练的比较基线，64K 在当前样本上的 token 节省也值得结合模型开销检验；我们尚未据此确定最终词表，因为 BPT 无法回答模型质量、训练成本和推理开销的问题。进入模型阶段时，应固定并保存所选 tokenizer，再用下游训练与评测验证选择。

## 资料入口

- [阅读笔记](../../reading_notes.md)与 [tokenizer 专题文档](../../tokenizer.md)：原理、代码路径和使用方式。
- [实验分析](../../experiments/tokenizer/tokenizer-20261003/analysis.md)：实验设置、结果与局限；[切分对比](../../experiments/tokenizer/tokenizer-20261003/comparison.md)、[精确指标](../../experiments/tokenizer/tokenizer-20261003/metrics.json)和[运行记录](../../experiments/tokenizer/tokenizer-20261003/run.json)保存结果、配置与数据分片指纹。运行 ID 为 `tokenizer-20261003`，实验代码提交为 [`72b090b`](https://github.com/sii-nyc/my-nanochat/commit/72b090b1ca6a16bb0d8ac6fe8d98c179086ef477)，语料审计脚本提交为 [`8ee7f17`](https://github.com/sii-nyc/my-nanochat/commit/8ee7f17)。
- 词表、训练日志、离线网页和完整语言审计 JSON 保存在服务器上 Git 忽略的 `training/tokenizer_experiments/tokenizer-20261003/`。
