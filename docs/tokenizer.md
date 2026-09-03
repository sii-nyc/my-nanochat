# Tokenizer 用法与实测结果

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

## 网页查看分词

在仓库根目录、已设置 `NANOCHAT_BASE_DIR` 的环境中运行：

```bash
python -m scripts.tok_export
# 可选：只导出一个本地 tokenizer，或指定 HTML 路径（.html 可省略）
python -m scripts.tok_export --tokenizer-file=tok_2b_chars --output training/tokenizer
```

默认按文件名排序读取 `NANOCHAT_BASE_DIR/tokenizer/` 顶层所有 `.pkl`，加入 GPT-2 和 GPT-4，生成 `NANOCHAT_BASE_DIR/tokenizer.html`。没有本地 tokenizer 时会明确提示，并仅导出两个 GPT 参照。首次导出可能下载 GPT 词表；不需要 GPU 或前端构建工具。

将生成的 HTML 下载到电脑后直接双击打开，无需服务器或联网。选择一个 tokenizer，左侧显示分词文本、右侧显示 IDs；悬停任一侧会高亮另一侧。完整 token（如 `Response`）整体高亮；一个字跨多个 token 时保留完整字形，并联动多个 ID。页面同时显示 token 数、BPT，支持复制 IDs；悬停 ID 可查看字节。输入上限为 6,000 个 UTF-16 单元。

网页与评估脚本都使用 `encode_ordinary`，不添加 BOS 或聊天模板，特殊 token 名称按普通文字处理。导出保留实际词表、正则和合并优先级；网页加载每个 tokenizer 时校验内置样例的 IDs 与 Python 一致，每次编码也检查字节能否还原原文。浏览器需支持 WebAssembly、Web Worker 和 `Intl.Segmenter`。

HTML 包含导出时的词表快照；新增或重训 tokenizer 后重新导出即可。`--output` 的相对路径以当前工作目录为基准，自动创建父目录并覆盖同名文件。

## 实测结果

以下汇总用户提供的服务器运行结果。训练规模按文件名及前述命令标记；本次未提供 `actual_chars`、训练耗时和数据清单，分析以训练时满足对应预算且数据保持一致为前提。

Tokenizer 目录：`/inspire/hdd/global_user/niuyuchen-253108120111/llm_playground/my-nanochat/training/tokenizer`。

### 定量结果

保留原始报告的四位小数；每列独立计算 BPT，越大越好。`climbmix-train` 和 `climbmix-val` 各仅覆盖一个 row group，训练样本只作诊断；这里不对七列求平均，也不计算跨文本总分。

| Tokenizer | 词表大小 | news | korean | code | math | science | climbmix-train | climbmix-val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | 50257 | 4.5025 | 1.1987 | 2.1858 | 1.9594 | 4.2769 | 4.6709 | 4.6013 |
| GPT-4 | 100277 | 4.7003 | 2.4533 | 4.0744 | 2.2043 | 4.4659 | 4.8213 | 4.7736 |
| `tok_1b_chars.pkl` | 32768 | 4.4914 | 1.1612 | 3.1554 | 2.0110 | 4.5950 | 4.7378 | 4.6649 |
| `tok_2b_chars.pkl` | 32768 | 4.4914 | 1.1923 | 3.1713 | 2.0132 | 4.5020 | 4.7371 | 4.6653 |
| `tok_2b_vocab16k.pkl` | 16384 | 4.2106 | 1.0531 | 2.8103 | 1.7980 | 4.0290 | 4.4441 | 4.3757 |
| `tok_2b_vocab64k.pkl` | 65536 | 4.7493 | 1.4834 | 3.4212 | 2.1501 | 5.1481 | 4.9215 | 4.8526 |
| `tok_3b_chars.pkl` | 32768 | 4.4914 | 1.1612 | 3.1713 | 2.0132 | 4.5203 | 4.7364 | 4.6653 |

### 简单分析

先看 `climbmix-val` 上两组控制变量对比。下列百分比均为 **BPT 相对变化**，由表中四位小数计算：`100 × (新 BPT / 原 BPT − 1)`。

| 固定条件 | 对比 | climbmix-val BPT | BPT 相对变化 |
| --- | --- | --- | ---: |
| 32K 词表 | 1B → 2B 字符 | 4.6649 → 4.6653 | 约 +0.0086% |
| 32K 词表 | 2B → 3B 字符 | 4.6653 → 4.6653 | 四位小数下相同 |
| 2B 字符 | 16K → 32K 词表 | 4.3757 → 4.6653 | 约 +6.62% |
| 2B 字符 | 32K → 64K 词表 | 4.6653 → 4.8526 | 约 +4.01% |

- **固定词表，增加语料量的收益很小且不一致。** 32K 下，验证样本几乎不变；news 三组完全相同，code 和 math 从 1B 到 2B 小幅提高后持平。korean 在 2B 最好，science 则在 1B 最好，因此本次结果未显示 3B 的稳定优势。四位小数相同只说明这些样本上的 BPT 接近，不表示词表或切分完全相同。
- **固定语料，扩大词表的影响更明显。** 2B 下，七类样本均随 16K → 32K → 64K 提高；验证样本从 16K 到 64K 共提高约 **10.90%**。验证样本上第二次翻倍的相对收益减小，但这个趋势不能推广到所有类别：32K → 64K 时，korean 和 science 分别提高约 **24.41%**、**14.35%**。
- **2B / 64K 在本地五个 tokenizer 中七项均最高。** 它也在七项上均超过 GPT-2；相对 GPT-4，在 news、science、climbmix-train、climbmix-val 四项领先，在 korean、code、math 三项落后。其验证样本 BPT 比 GPT-4 高约 **1.65%**，但参考 tokenizer 的词表大小、训练语料和规则不同，这不是相同训练条件下的对照。

这批样本支持的初步结论是：**词表大小带来的 BPT 差异，明显大于固定 32K 时 1B–3B 训练语料量带来的差异**。它尚不能说明 1B 对所有领域都已足够，也不能仅凭 BPT 判断语言模型效果或训练成本。

### 定性观察

以下摘录上方同一示例中的实际 token 片段，保留前导空格、省略 ID。三种 32K tokenizer 在这三处的切分一致，合并展示。

| Tokenizer | ` parseHTTPResponse` | ` café` | `LaTeX` |
| --- | --- | --- | --- |
| GPT-2 | `[" parse", "HT", "T", "PR", "esp", "onse"]` | `[" café"]` | `["La", "TeX"]` |
| GPT-4 | `[" parse", "HTTP", "Response"]` | `[" café"]` | `["La", "TeX"]` |
| 2B / 16K | `[" par", "se", "HT", "TP", "Res", "p", "onse"]` | `[" c", "af", "é"]` | `["L", "a", "Te", "X"]` |
| 1B、2B、3B / 32K | `[" par", "se", "HT", "TP", "Resp", "onse"]` | `[" caf", "é"]` | `["La", "Te", "X"]` |
| 2B / 64K | `[" parse", "HT", "TP", "Response"]` | `[" café"]` | `["La", "TeX"]` |

- **扩大词表确实合并了更多片段。** 表中的 `Response`、` café`、`TeX` 在 64K 下更完整；中文“用”在 2B / 16K 下切成 `b'\xe7'`、`b'\x94'`、`b'\xa8'`，32K 下为 `b'\xe7\x94'`、`b'\xa8'`，64K 下成为一个完整的 `"用"` token。不过，64K 在中韩文中仍有许多字节片段，不能据此认定这些语言的覆盖已充分。
- **增加语料也会改变局部切分。** 1B / 32K 把 `user_id` 的后缀切成 `"_"`、`"id"`，2B 和 3B / 32K 则有完整的 `"_id"`。这说明总体 BPT 接近时，局部词表仍可能不同。
- **部分边界由规则决定。** 五个本地 tokenizer 都把 `2026` 切成 `"20"`、`"26"`，把小数点后的 `14159` 切成 `"14"`、`"15"`、`"9"`，与当前正则最多按两位数字预切分一致；扩大词表不会取消这些预切分边界。字节片段和组合重音的独立切分也不等于编码错误，应以完整文本能否还原为准。
