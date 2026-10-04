# 文档索引

```text
docs/
├── index.md                              # 本目录的文档索引
├── reading_notes.md                      # 阅读和理解项目代码时持续整理的个人笔记
├── tokenizer.md                          # 分词器原理、使用及对比实验的持续整理
├── experiments/                          # 各次实验的设置、报告与分析
│   └── tokenizer/                       # 分词器实验的设置、原始报告与分析
│       └── tokenizer-20261003/           # 本次从头训练与评估的可追溯报告
│           ├── README.md                 # 运行信息与报告入口
│           ├── analysis.md               # 实验结果、语料语言构成和结论
│           ├── comparison.md             # 完整 BPT 表与切分样例
│           ├── metrics.json              # 样本指纹、token 数和 BPT
│           └── run.json                  # 命令、环境、分片清单和产物指纹
└── progress/                             # 按阶段组织的项目进展记录
    ├── README.md                         # 当前状态与里程碑索引
    └── nanochat/                         # 理解和复现 nanochat 阶段的记录
        └── 2026-10-01-tokenizer.md       # 分词器原理与正式复现实验的阶段总结
```

新增或移动 `docs/` 下的文档时，同步更新本索引。
