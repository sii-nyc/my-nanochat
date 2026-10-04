# Tokenizer comparison: tokenizer-20261003

实验结论、训练语料语言构成与解释见[中文分析](analysis.md)。本目录从服务器 `--no-upload` 运行的报告复制而来；下方绝对路径是原服务器路径，本地阅读时以当前仓库中的报告文件为准。

Code commit: 72b090b1ca6a16bb0d8ac6fe8d98c179086ef477
Data directory on server: /inspire/hdd/global_user/niuyuchen-253108120111/llm_playground/my-nanochat/training/base_data_climbmix
Validation shard: shard_06542.parquet
Server-only artifacts: /inspire/hdd/global_user/niuyuchen-253108120111/llm_playground/my-nanochat/training/tokenizer_experiments/tokenizer-20261003/base/tokenizer
Server-only offline viewer: /inspire/hdd/global_user/niuyuchen-253108120111/llm_playground/my-nanochat/training/tokenizer_experiments/tokenizer-20261003/base/tokenizer.html
Server-only logs: /inspire/hdd/global_user/niuyuchen-253108120111/llm_playground/my-nanochat/training/tokenizer_experiments/tokenizer-20261003/logs

All five tokenizers were trained from scratch from the same ordered dataset and split.
For the BPT table and qualitative token splits, see [comparison.md](comparison.md).
For exact byte/token counts and sample fingerprints, see [metrics.json](metrics.json).
Commands, versions, shard manifest, consumed-shard hashes, durations and artifact hashes are in [run.json](run.json).

| Tokenizer | Character target | Actual characters | Vocab size | BPE training seconds |
| --- | ---: | ---: | ---: | ---: |
| tok_1b_chars | 1,000,000,000 | 1,000,001,077 | 32,768 | 55.84 |
| tok_2b_chars | 2,000,000,000 | 2,000,005,430 | 32,768 | 103.51 |
| tok_3b_chars | 3,000,000,000 | 3,000,005,961 | 32,768 | 156.47 |
| tok_2b_vocab16k | 2,000,000,000 | 2,000,005,430 | 16,384 | 102.23 |
| tok_2b_vocab64k | 2,000,000,000 | 2,000,005,430 | 65,536 | 101.16 |

The train and validation BPT samples each cover only the first row group.
BPT alone does not measure downstream language model quality.
