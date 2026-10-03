# Tokenizer

## 简介

nanochat 使用 byte-level BPE（Byte Pair Encoding）将文本转换为模型处理的 token ID。这一流程分为两个阶段：训练阶段从语料中学习词表和合并规则；使用/推理阶段按照这些规则把新文本编码为 ID，并将 ID 解码回文本。

本项目没有从零实现 BPE 算法，而是组合了两个库：

- [rustbpe](https://github.com/karpathy/rustbpe)：用 Rust 实现、提供 Python 接口，负责训练 GPT 风格的 byte-level BPE。
- [tiktoken](https://github.com/openai/tiktoken)：在 nanochat 中负责高效编码与解码。训练完成后，nanochat 将 rustbpe 学到的预切分正则和 token 合并优先级构造成 tiktoken 的 Encoding。

项目的衔接代码在 [nanochat/tokenizer.py](../nanochat/tokenizer.py)。

## 训练阶段

```bash
uv run python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 --doc-cap=10000 --tokenizer-file=tokenizer.pkl
```

参数说明：

- **数据目录**：默认使用仓库根目录下的 `training/`，从 `training/base_data_climbmix/` 读取数据，并将结果保存到 `training/tokenizer/`。需要改变位置时可设置 `NANOCHAT_BASE_DIR` 环境变量。
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

### 实验结果（运行标识：`tokenizer-20261003`）

服务器使用代码提交 `72b090b1ca6a16bb0d8ac6fe8d98c179086ef477`，以 `--run-id tokenizer-20261003 --no-upload` 从头完成五组训练、评估和离线网页导出。实际运行日期尚未从 `run.json` 核实。训练字符数均略超过目标，是因为脚本在读完使累计量超过预算的那篇文档后停止；三组 2B 实验实际读取的字符数完全相同。脚本还检查了各词表文件、token 字节长度缓存、编码/解码往返、评估结果和验证样本是否存在。

| 实验词表 | 实际训练字符数 | 训练阶段耗时（秒） | ClimbMix 训练 BPT | ClimbMix 验证 BPT | 韩文 BPT | 代码 BPT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tok_1b_chars`（32K） | 1,000,001,077 | 55.84 | 4.7378 | 4.6892 | 1.1612 | 3.1554 |
| `tok_2b_chars`（32K） | 2,000,005,430 | 103.51 | 4.7371 | 4.6899 | 1.1923 | 3.1713 |
| `tok_3b_chars`（32K） | 3,000,005,961 | 156.47 | 4.7364 | 4.6901 | 1.1612 | 3.1713 |
| `tok_2b_vocab16k` | 2,000,005,430 | 102.23 | 4.4441 | 4.3946 | 1.0531 | 2.8103 |
| `tok_2b_vocab64k` | 2,000,005,430 | 101.16 | 4.9215 | 4.8791 | 1.4834 | 3.4212 |

以上数值来自服务器生成的 `public/README.md` 与 `public/comparison.md`。完整报告及精确 token 数保存在服务器 `training/tokenizer_experiments/tokenizer-20261003/public/`；词表、日志和离线网页位于同一运行目录下的 `base/` 与 `logs/`。此次使用 `--no-upload`，报告尚未同步到仓库。

### 分析

**训练语料量。**固定 32K 词表时，ClimbMix 验证样本的 BPT 从 1B 字符的 4.6892 变为 2B 的 4.6899、3B 的 4.6901。1B 到 3B 的相对增幅约为 **0.02%**，而记录的训练阶段耗时从 55.84 秒增至 156.47 秒。新闻样例的 BPT 在三组中均为 4.4914；科学文本则没有单调改善。这说明在当前样本和设置下，继续增加训练字符数的可见收益很小，不能据此推断更多数据对其他文本或下游模型也没有价值。

**词表大小。**固定 2B 字符时，ClimbMix 验证 BPT 随 16K、32K、64K 词表增大依次为 4.3946、4.6899、4.8791。32K 比 16K 高约 **6.72%**；64K 比 32K 高约 **4.03%**，相当于同一验证文本约少用 **3.88%** 的 token。韩文与代码样例也受益于更大的词表，但幅度不同：32K 到 64K 的韩文 BPT 从 1.1923 增至 1.4834，代码从 3.1713 增至 3.4212。本次三个 2B 实验记录的训练阶段耗时接近，不能仅凭单次计时判断词表大小对训练成本没有影响。

**切分观察。**固定样例中，64K 词表能够把部分中文字符（如“用”“文”“本”）、韩文片段和代码中的 `Response` 合并成完整 token；16K 词表则更常拆成单字节或不完整的 UTF-8 片段。即使是 64K，中文和韩文仍有大量这样的字节片段。GPT-2 和 GPT-4 词表提供了参照，但其训练数据和词表不同，不能从横向 BPT 直接得出算法优劣结论。

ClimbMix 的训练与验证评估各只覆盖首个 row group，其他领域各只有一段固定文本；目前没有单独的中文定量样本，也没有训练语言模型后的质量和成本数据。因此这些结果可以说明本批文本的分词压缩率，尚不足以确定后续基础模型应使用 32K 还是 64K 词表。下一步应先保留完整报告，再按模型训练目标评估更广的文本分布及词表增大带来的模型参数和计算开销。
