# Tokenizer

## 简介

nanochat 使用 byte-level BPE（Byte Pair Encoding）将文本转换为模型处理的 token ID。这一流程分为两个阶段：训练阶段从语料中学习词表和合并规则；使用/推理阶段按照这些规则把新文本编码为 ID，并将 ID 解码回文本。

本项目没有从零实现 BPE 算法，而是组合了两个库：

- [rustbpe](https://github.com/karpathy/rustbpe)：用 Rust 实现、提供 Python 接口，负责训练 GPT 风格的 byte-level BPE。
- [tiktoken](https://github.com/openai/tiktoken)：在 nanochat 中负责高效编码与解码。训练完成后，nanochat 将 rustbpe 学到的预切分正则和 token 合并优先级构造成 tiktoken 的 Encoding。

项目的衔接代码在 [nanochat/tokenizer.py](../nanochat/tokenizer.py)。

## 训练阶段

```bash
NANOCHAT_BASE_DIR=training uv run python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 --doc-cap=10000 --tokenizer-file=tokenizer.pkl
```

参数说明：

- **NANOCHAT_BASE_DIR=training**：数据与运行产物的根目录，是环境变量而非训练脚本参数。脚本从 training/base_data_climbmix/ 读取数据，并将结果保存到 training/tokenizer/；若不设置，默认使用 ~/.cache/nanochat/。
- **--max-chars=2000000000**：训练文本的字符预算，按每篇截断后的 Python 字符数累计，不是 UTF-8 字节数。累计数超过预算后才停止，因此可能多读一篇文档；如果数据不足，脚本不会自动下载补齐。
- **--vocab-size=32768**：最终词表大小，包含普通 token 和特殊 token。
- **--doc-cap=10000**：每篇文档最多保留 10000 个字符，避免少数超长文档占据过多训练预算。
- **--tokenizer-file=tokenizer.pkl**：保存的词表文件名，位于根目录的 tokenizer/ 子目录中。文件名可省略 .pkl 后缀；默认缓存名为 token_bytes.pt，若指定 tok_a.pkl，则缓存名为 tok_a.token_bytes.pt。使用不同名称可保留多个实验词表。

执行效果：脚本读取预先放在 training/base_data_climbmix/ 的 Parquet 训练数据，从头训练 BPE 分词器，并进行一次编码与解码的往返检查。完成后在 training/tokenizer/ 生成 tokenizer.pkl（包含词表、预切分正则、合并优先级和特殊 token 定义）与 token_bytes.pt（每个 token 的原始字节长度，供 BPB 评测使用），同时打印实际读取的字符数和训练耗时。两个文件同名已存在时会被覆盖；整个 training/ 目录已被 Git 忽略。

训练的主要过程如下：

1. 从预训练数据的训练分片读取文本；数据以 Parquet 文件保存，按文件名排序后，最后一个分片留作验证，其余用于训练。每篇文本先按 doc-cap 截断，再累计到 max-chars 字符预算附近。
2. rustbpe 使用 [SPLIT_PATTERN](../nanochat/tokenizer.py) 正则对文本预切分。后续 BPE 合并只在各个预切分片段内部进行，不跨越片段边界。
3. 将每个片段编码成 UTF-8 字节。byte-level BPE 从 256 个单字节 token（0–255）开始，因此任意 UTF-8 文本都能表示。
4. 统计相邻 token 对的出现频率，将最高频的一对合并为新 token，然后重新统计、继续合并。例如两个单字节 token 可以合并为一个表示更长字节序列的 token。
5. 普通 token 的数量达到目标后，另行加入 nanochat 的特殊 token（如 <|bos|>）。这里 --vocab-size 指**包含特殊 token 的最终词表大小**。当前代码定义了 9 个特殊 token，因此以 32768 为目标时，普通 token 目标为 32759；从 256 个基础 token 出发，目标合并次数为 32768 − 9 − 256 = 32503。

## 推理阶段

模型使用分词器时，nanochat 从已保存的文件加载 tiktoken Encoding。下面是编码、解码和查看 token 原始字节的最小示例；运行前需要先训练并保存默认词表：

```python
import os
os.environ["NANOCHAT_BASE_DIR"] = "training"  # 与上面的训练命令使用同一目录

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer(filename="tokenizer.pkl")  # 加载上面命令训练的词表
# 如果训练时使用 --tokenizer-file=tok_2b_vocab64k，也可改为：
# tokenizer = get_tokenizer(filename="tok_2b_vocab64k")
text = "Hello, 世界"

ids = tokenizer.encode(text)  # 普通文本编码为 token ID；默认不加特殊 token
restored = tokenizer.decode(ids)  # ID 解码回原文
assert restored == text

pieces = [tokenizer.decode_single_token_bytes(i) for i in ids]
assert b"".join(pieces) == text.encode("utf-8")

ids_with_bos = tokenizer.encode(text, prepend="<|bos|>")
assert ids_with_bos[0] == tokenizer.get_bos_token_id()
```

get_tokenizer 的 filename 可写完整的 .pkl 文件名，也可省略后缀；不传时默认加载 tokenizer.pkl。一个 token 的原始字节片段未必能单独解码为完整 Unicode 字符，因此查看切分时使用 decode_single_token_bytes；所有片段拼接后应还原原文。nanochat 的普通文本编码使用 tiktoken 的 encode_ordinary，只有显式指定 prepend 或 append 时才加入相应的特殊 token。

## Tokenizer 对比实验

### 目的与设计

本次实验从头训练多组 tokenizer，分别考察**训练语料量**和**词表大小**对分词表现的影响，为后续训练聊天模型选择词表提供依据。重点观察相同文本需要多少 token、不同类型文本如何切分，以及训练所需时间；BPT 等分词指标不能单独代表下游模型质量。

使用同一批 ClimbMix Parquet 分片，按文件名排序后将最后一个分片留作验证集；所有实验保持相同的数据顺序、预切分规则和 `doc-cap=10000`。每次只改变下表中的一个变量：

| 实验词表 | 训练字符预算 | 词表大小 | 对比目的 |
| --- | ---: | ---: | --- |
| `tok_1b_chars` | 1B | 32K | 与 2B、3B 比较语料量 |
| `tok_2b_chars` | 2B | 32K | 两组对比的共同基线 |
| `tok_3b_chars` | 3B | 32K | 与 1B、2B 比较语料量 |
| `tok_2b_vocab16k` | 2B | 16K | 与 32K、64K 比较词表大小 |
| `tok_2b_vocab64k` | 2B | 64K | 与 16K、32K 比较词表大小 |

### 实验流程

1. 固定训练数据及评估文本，记录代码提交、数据分片、依赖版本和各组实际读取的字符数。所有 tokenizer 均从头训练，不复用上次实验的词表。
2. 依次训练五个 tokenizer，分别保存词表与训练日志，记录训练耗时；检查编码后能否完整解码。
3. 在相同的 ClimbMix 训练/验证样本，以及新闻、韩文、代码、数学/LaTeX、科学文本上计算 **BPT = UTF-8 字节数 ÷ token 数**。同一文本上 BPT 越高，表示使用的 token 越少。当前评估脚本对 ClimbMix 两组样本各只读取第一个 row group，因此这一指标是抽样结果。
4. 比较固定样例的实际切分，并使用离线网页查看多语言、代码、数字和特殊字符的分词情况。GPT-2、GPT-4 tokenizer 作为外部参照；由于训练语料、词表和规则不同，不把它们当作控制变量实验。
5. 分别分析增加训练语料、扩大词表的收益与训练成本，再结合定性观察选择后续模型训练使用的 tokenizer。若抽样结果不足以支持选择，再扩大验证样本。

实验入口为 `runs/tokenizer_experiment.py`，在服务器准备好 ClimbMix Parquet 数据后运行：

```bash
uv run python runs/tokenizer_experiment.py --data-dir /path/to/base_data_climbmix
```

脚本将词表、日志和离线网页保存在 Git 忽略的 `training/tokenizer_experiments/` 中；将 Markdown 报告、精确的字节数与 token 数及 BPT、运行配置写入 `docs/experiments/tokenizer/`，默认提交并推送到当前分支的上游。运行前需要干净的 Git 工作区和已配置的上游分支。

如果希望本次运行的**全部产物只保存在服务器的 `training/` 下**，在命令末尾加 `--no-upload`。此时报告留在 `training/tokenizer_experiments/<run-id>/public/`，脚本不会在 `docs/` 创建副本，也不会提交或推送结果。

### 实验结果

待本次训练与评估完成后填写。

### 分析

待结合定量结果、实际切分及训练成本填写。
