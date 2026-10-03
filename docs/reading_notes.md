# Data

## Pretraining

data source: https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix
NVIDIA 清洗、筛选、混合后的预训练数据集，以 GPT-2 token IDs 保存

nvidia/Nemotron-ClimbMix
    ↓
repackage_data_reference.py
    ├─ GPT-2 tokens 解码为文本
    ├─ seed=42 shuffle
    ├─ 按约 2.5 亿字符分片
    ├─ 写成 Parquet
    └─ 上传到 Hugging Face
    ↓
karpathy/climbmix-400b-shuffle (shard_[00000-06542].parquet)
    ↓
dataset.py
    ├─ 命令行使用：下载数据到本地 python -m nanochat.dataset -n -1 -w 4 (n 表示下载几个 shard，w 是 worker 数量)
    └─ import 使用: parquets_iter_batched(split, start=0, step=1) 返回一个生成器，每次 yield 一个 row group list[str]，start 和 step 参数用于 DDP (start=rank, step=world_size).

# Tokenizer

依赖库:
- [rustbpe](https://github.com/karpathy/rustbpe): a simple, efficient (GPT-style) byte-level BPE training implementation in Rust with Python bindings.
- [tiktoken](https://github.com/openai/tiktoken): a fast BPE tokenizer (inference-only, i.e., encode and decode)

## 训练

python -m scripts.tok_train --max-chars=xxx --vocab-size xxx --doc-cap xxx

1. 准备训练数据，这里使用 pretraining data， 并通过 max-chars 控制训练语料字符数；vocab-size 控制最终词表大小；doc-cap 控制每篇文档最多截取多少字符（防止少量超长文档 dominate）
2. 每篇文档先经过 SPLIT_PATTERN 正则预切分
3. 将所有文档转换为 UTF-8 bytes 序列，Byte-level BPE 的基础词表固定包含 256 个 token (对应 byte 0-255)
4. 不断统计相邻 token pair 频率，并合并最高频 pair 为一个新的 token (e.g., [23, 45] -> 256)
5. 直到词表达到目标大小（即进行 vocab-size - 256 次合并）

## 推理