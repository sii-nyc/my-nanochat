# Tokenizer 用法

## 训练与评估

```bash
python -m scripts.tok_train --max-chars=1000000000 --tokenizer-file=tok_1b_chars
# 指定一个 tokenizer，与 GPT-2、GPT-4 比较
python -m scripts.tok_eval --tokenizer-file=tok_1b_chars
# 自动发现并比较目录中的所有 tokenizer，以及 GPT-2、GPT-4
python -m scripts.tok_eval
# 同时保存 Markdown 报告（.md 可省略）
python -m scripts.tok_eval --output reports/tokenizer_eval
```

| 训练参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--max-chars` | `2000000000` | 文档截断后累计的字符预算，按 Python `len(text)` 计数；**1B = 10⁹ 字符** |
| `--doc-cap` | `10000` | 每篇文档只保留前 N 个字符 |
| `--vocab-size` | `32768` | 最终词表大小，包含特殊 token |
| `--tokenizer-file` | `tokenizer.pkl` | 文件名；自动补全后缀 `.pkl` |

**训练**：运行 `scripts.tok_train` 后，依次执行：

1. **划分数据集**：列出 `NANOCHAT_BASE_DIR` 下 `base_data_climbmix/` 中的 Parquet 文件，按文件名排序，将最后一个作为验证集，其余作为训练集。
2. **读取训练文本**：顺序读取训练文件各 row group 的 `text` 列，每篇只取前 `doc_cap` 个字符。交出该文档后，若累计字符数 **大于** `max_chars` 就停止取样，因此最多超出一个 `doc_cap`（默认 10,000 字符）；若数据提前耗尽，则用已读取的文本训练，不报预算不足错误。
3. **训练 tokenizer**：将文本交给 `rustbpe` 在 CPU 上训练 BPE，学习词表和合并规则，再加入特殊 token，构造用于编码、解码的 tokenizer。
4. **保存与检查**：先保存 tokenizer，再检查样例文本能否编码后完整解码，最后生成字节数缓存。两个文件均保存在 `NANOCHAT_BASE_DIR` 下的 `tokenizer/` 目录。

以 `--tokenizer-file=tok_1b_chars` 为例：

| 文件 | 内容与用途 |
| --- | --- |
| `tok_1b_chars.pkl` | 训练好的 tokenizer，包含词表、合并优先级、预切分正则和特殊 token 定义；加载后可直接编码、解码文本 |
| `tok_1b_chars.token_bytes.pt` | token ID 到原始字节长度的映射张量，特殊 token 记为 0；供语言模型的 bits-per-byte（BPB）评估使用 |

不指定名称时，保存为 `tokenizer.pkl` 和 `token_bytes.pt`；同名文件会被覆盖，改名不影响输入数据。

注：`base_train` 等模型脚本仍默认加载 `tokenizer.pkl`；自定义名称目前只接入 tokenizer 训练与评估，不会自动切换模型训练配置。

**评估**：运行 `scripts.tok_eval` 后，依次执行：

1. **选择 tokenizer**：从 `NANOCHAT_BASE_DIR` 下的 `tokenizer/` 目录加载文件。指定 `--tokenizer-file` 时只选该文件（`.pkl` 可省略）；不指定时，按文件名排序加载目录顶层所有 `.pkl` 文件。两种模式都加入 `gpt2`（GPT-2）、`cl100k_base`（GPT-4）作为参照。
2. **定量评估**：使用内置新闻、韩文、代码、数学/LaTeX、科学文本，再按**当前数据目录**的文件排序与划分规则，读取训练集和验证集各自的**第一个 row group**，分别用换行拼接其中的文档。该划分不追踪 tokenizer 的实际训练数据，需保持数据目录不变才能与训练时一致。所有 tokenizer 在这些相同文本上计算 BPT。
3. **定性比较**：所有 tokenizer 编码下方同一段示例文本，展示每个 token 的 ID 和字节片段，观察中外文、数字、代码及空白的切分。定量、定性编码均不额外添加 BOS 或聊天模板 token，并检查 `decode(encode(text)) == text`；不一致即终止。
4. **输出结果**：先显示一张 BPT 表，每行一个 tokenizer、每列一组测试文本，词表大小作为背景信息；随后显示示例原文和各 tokenizer 的完整分词结果。默认只打印到屏幕；指定 `--output 文件名` 时，同时保存相同内容的 UTF-8 Markdown 报告。

`--output` 支持相对或绝对路径，相对路径以当前工作目录为基准；自动补全 `.md` 后缀并创建父目录，同名报告会被覆盖。可与 `--tokenizer-file` 同时使用。

**定量指标**：只使用 **BPT（bytes/token）= 文本 UTF-8 字节数 / token 数**，显示四位小数；同一文本上越大，每个 token 平均承载的文本越多。它不表示磁盘压缩比，也不能单独推断模型效果；空文本记为 `N/A`。

**定性示例**：这段文本只用于展示切分，不加入定量测试集合。两处 `café` 分别使用预组合字符和组合重音，外观相似但编码不同。

```text
Hello, tokenizer! 今天用 Python 处理文本：你好，世界🙂。
한국어도 테스트합니다. café ≠ café; 2026-09-03, 3.14159.
def parseHTTPResponse(user_id=42):
    return f"user_{user_id}"  # 保留空格与换行
公式：x^2 + y^2 = z^2；LaTeX: \frac{a+b}{2}
```

每项按 `ID:片段` 显示，用 `|` 分隔 token；引号内保留空格，换行显示为 `\n`。能独立解码的片段直接显示文字；不完整的 UTF-8 片段用 `b'\xe4\xbd'` 这类原始字节形式显示，拼接后仍完整还原原文。token ID 只在各自词表内有意义，跨 tokenizer 应比较切分片段。

参考 tokenizer 首次可能下载词表，不运行 GPT 模型；其词表大小、语料和预切分规则不同，差异不能只归因于语料量。当前定量评估仅覆盖上述样本，尚未实现完整验证集评估和逐文档统计。
