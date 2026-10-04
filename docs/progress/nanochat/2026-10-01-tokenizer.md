# Tokenizer：原理理解与复现实验

**记录日期**：2026-10-01；**更新日期**：2026-10-04

**状态**：已完成 tokenizer 原理与代码流程的理解，以及一次可追溯的从头训练和对比实验。

## 目标与结论

这一阶段以 nanochat 的 tokenizer 为对象，理解文本如何从 ClimbMix Parquet 文档进入训练流程，经预切分、UTF-8 字节基础词表和 BPE 合并生成词表，再由 tiktoken 完成编码与解码。我们梳理了 `doc-cap`、字符预算、特殊 token 和 token 字节长度缓存的作用，并通过实际训练和往返检查验证了代码路径。完整原理与用法见[专题文档](../../tokenizer.md)。

2026-10-03 从头训练了五个 tokenizer：固定 32K 词表比较 1B、2B、3B 训练字符，固定 2B 字符比较 16K、32K、64K 词表。在同一 ClimbMix 验证样本上，前三者的 BPT 分别为 **4.6892、4.6899、4.6901**，后三者为 **4.3946、4.6899、4.8791**。本次样本中，扩大词表减少 token 数的效果比把 32K 词表的训练文本从 1B 加到 3B 更明显；但 BPT 不能替代模型训练后的质量评估。

随后核对了这次实际读取的前 14 个训练分片，并对逐篇截断后的文本做语言识别：在 1B、2B、3B 截止点，判为英语主要语言的字符占比分别为 **99.953%、99.952%、99.955%**。因此本轮主要是英语 tokenizer 实验；韩文等文本的切分观察属于跨语言压力测试，不能说明模型已经具备多语言能力。语言标签是识别器预测，完整口径与限制见[本次实验分析](../../experiments/tokenizer/tokenizer-20261003/analysis.md)。

## 证据

[阅读笔记](../../reading_notes.md)和[专题文档](../../tokenizer.md)记录 BPE 流程与使用方式；[实验分析](../../experiments/tokenizer/tokenizer-20261003/analysis.md)和同目录的 `comparison.md`、`metrics.json`、`run.json` 保存设置、指标、切分样例及数据分片指纹。训练运行 ID 为 `tokenizer-20261003`，实验代码提交为 [`72b090b`](https://github.com/sii-nyc/my-nanochat/commit/72b090b1ca6a16bb0d8ac6fe8d98c179086ef477)；补充语料审计脚本提交为 [`8ee7f17`](https://github.com/sii-nyc/my-nanochat/commit/8ee7f17)。词表、训练日志和离线网页保存在 Git 忽略的 `training/tokenizer_experiments/tokenizer-20261003/`，语言审计的完整 JSON 保留在服务器该运行的 `public/` 目录。

## 未解决的问题

Tokenizer 阶段的原理学习和本次对比实验已完成。接下来进入模型架构与代码实现的学习；实际训练基础模型前，还需结合模型质量、训练与推理开销确定最终词表。若要研究多语言模型，需要另行检查或构建相应语料，并扩大多语言评估。当前语言比例只描述本次 tokenizer 使用的分片，不能外推到整套 ClimbMix。
