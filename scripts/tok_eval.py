"""Compare tokenizer BPT and example segmentations, optionally saving Markdown."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer, get_tokenizer_paths, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

parser = argparse.ArgumentParser(description='Compare tokenizer BPT and example segmentations')
parser.add_argument('--tokenizer-file', help='Evaluate one filename in the tokenizer directory (.pkl is optional); if omitted, discover all .pkl files. GPT-2 and GPT-4 baselines are always included.')
parser.add_argument('--output', type=Path, help='Also save the report to this Markdown file (.md is appended if omitted). Relative paths use the current directory; parent directories are created. Defaults to stdout only.')
parser.add_argument('--json-output', type=Path, help='Save exact byte/token counts, BPT, and sample fingerprints as JSON for later analysis.')
args = parser.parse_args()
output_path = args.output
if output_path is not None:
    output_path = output_path.expanduser()
    if not output_path.name or output_path.name == '..' or output_path.is_dir():
        parser.error('--output must name a Markdown file, not a directory')
    if output_path.suffix.lower() != '.md':
        output_path = output_path.with_name(output_path.name + '.md')
json_output_path = args.json_output
if json_output_path is not None:
    json_output_path = json_output_path.expanduser()
    if not json_output_path.name or json_output_path.name == '..' or json_output_path.is_dir():
        parser.error('--json-output must name a JSON file, not a directory')
    if json_output_path.suffix.lower() != '.json':
        json_output_path = json_output_path.with_name(json_output_path.name + '.json')
tokenizer_dir = Path(get_base_dir()) / "tokenizer"
if args.tokenizer_file is not None:
    try:
        tokenizer_path, _ = get_tokenizer_paths(tokenizer_dir, args.tokenizer_file)
    except ValueError as e:
        parser.error(str(e))
    tokenizer_files = [Path(tokenizer_path).name]
else:
    tokenizer_files = sorted(path.name for path in tokenizer_dir.glob("*.pkl") if path.is_file())
    if not tokenizer_files:
        parser.error(f"No tokenizer .pkl files found in {tokenizer_dir}. Train one with python -m scripts.tok_train first.")

local_tokenizers = {}
for filename in tokenizer_files:
    try:
        local_tokenizers[filename] = get_tokenizer(filename=filename)
    except Exception as e:
        parser.error(f"Could not load tokenizer {tokenizer_dir / filename}: {e}")

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()

# These labels use the current directory split, not a record of training inputs.
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("climbmix-train", train_text),
]
if val_text:
    all_text.append(("climbmix-val", val_text))

# One shared probe for inspecting boundaries, whitespace, numbers and byte pieces.
# The second spelling of cafe\u0301 uses a combining accent intentionally.
qualitative_text = (
    "Hello, tokenizer! 今天用 Python 处理文本：你好，世界🙂。\n"
    "한국어도 테스트합니다. café ≠ cafe\u0301; 2026-09-03, 3.14159.\n"
    'def parseHTTPResponse(user_id=42):\n'
    '    return f"user_{user_id}"  # 保留空格与换行\n'
    r"公式：x^2 + y^2 = z^2；LaTeX: \frac{a+b}{2}"
)

# Evaluate every tokenizer on the same texts, loading each baseline only once.
tokenizers = {
    "GPT-2": RustBPETokenizer.from_pretrained("gpt2"),
    "GPT-4": RustBPETokenizer.from_pretrained("cl100k_base"),
    **local_tokenizers,
}
tokenizer_results = {}
vocab_sizes = {}
qualitative_results = {}
sample_info = {
    name: {
        'utf8_bytes': len(text.encode('utf-8')),
        'sha256_utf8': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'documents': len(train_docs) if name == 'climbmix-train' else len(val_docs) if name == 'climbmix-val' else None,
        'row_group': 0 if name.startswith('climbmix-') else None,
    }
    for name, text in all_text
}

for tokenizer_name, tokenizer in tokenizers.items():
    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Round-trip failed for {tokenizer_name} on {name}"

        encoded_bytes = text.encode('utf-8')
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'bpt': len(encoded_bytes) / len(encoded) if encoded else None,
        }

    ids = tokenizer.encode(qualitative_text)
    assert tokenizer.decode(ids) == qualitative_text, f"Round-trip failed for {tokenizer_name} on the qualitative sample"
    pieces = [tokenizer.decode_single_token_bytes(token_id) for token_id in ids]
    assert b''.join(pieces) == qualitative_text.encode('utf-8'), f"Token bytes do not reconstruct the qualitative sample for {tokenizer_name}"
    qualitative_results[tokenizer_name] = {'ids': ids, 'pieces': pieces}


def markdown_escape(text):
    """Keep filenames literal inside both Markdown tables and headings."""
    text = str(text).replace('\r', r'\r').replace('\n', r'\n')
    return re.sub(r'([\\`*_\[\]|<>])', r'\\\1', text)


def format_piece(piece):
    """Readable Unicode when complete, lossless byte notation otherwise."""
    try:
        return json.dumps(piece.decode('utf-8'), ensure_ascii=False)
    except UnicodeDecodeError:
        return repr(piece)


def format_segmentation(ids, pieces, width=100):
    """Wrap between tokens so every id:piece item stays intact."""
    lines = []
    line = ''
    for token_id, piece in zip(ids, pieces):
        item = f'{token_id}:{format_piece(piece)}'
        if line and len(line) + len(item) + 3 > width:
            lines.append(line)
            line = ''
        line += (' | ' if line else '') + item
    if line:
        lines.append(line)
    return '\n'.join(lines)


text_names = [name for name, _ in all_text]
headers = ['Tokenizer', 'Vocab size', *text_names]
report_lines = [
    '# Tokenizer evaluation',
    '',
    f'Local tokenizer directory: {markdown_escape(tokenizer_dir)}',
    '',
    '## Quantitative comparison: BPT',
    '',
    'BPT = UTF-8 bytes / tokens. Higher is better on the same text. Vocab size is metadata; N/A means empty text.',
    'All tokenizers encode the same raw texts without extra BOS or chat-template tokens, with exact round-trip checks.',
    '',
    'The five built-in samples are news, Korean, code, math/LaTeX and science. '
    'climbmix-train and climbmix-val each join the first row group of the current train/validation split with newlines. '
    'These labels do not track the data actually used to train each tokenizer.',
    '',
    '| ' + ' | '.join(headers) + ' |',
    '| --- | ' + ' | '.join(['---:'] * (len(headers) - 1)) + ' |',
]
for tokenizer_name, results in tokenizer_results.items():
    cells = [markdown_escape(tokenizer_name), str(vocab_sizes[tokenizer_name])]
    cells.extend(f"{results[name]['bpt']:.4f}" if results[name]['bpt'] is not None else 'N/A' for name in text_names)
    report_lines.append('| ' + ' | '.join(cells) + ' |')

report_lines.extend([
    '',
    '## Qualitative comparison: shared sample',
    '',
    'The sample covers English, Chinese, Korean, numbers, code identifiers/indentation, LaTeX, emoji and composed/combining accents.',
    '',
    '```text',
    qualitative_text,
    '```',
    '',
    r'Each item is token_id:piece; | separates tokens. Quoted strings preserve spaces and escape newlines as \n. '
    r"b'\x..' notation preserves raw bytes when a token is not valid UTF-8 on its own. "
    'Such fragments are normal; their concatenated bytes reconstruct the original text. '
    'Token IDs are specific to each tokenizer and should not be compared across tokenizers.',
])
for tokenizer_name, result in qualitative_results.items():
    report_lines.extend([
        '',
        f'### {markdown_escape(tokenizer_name)}',
        '',
        '```text',
        format_segmentation(result['ids'], result['pieces']),
        '```',
    ])

# The terminal and optional file receive exactly the same Markdown report.
report = '\n'.join(report_lines) + '\n'
print(report, end='')
if output_path is not None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding='utf-8')
    except OSError as e:
        parser.error(f'Could not save report to {output_path}: {e}')
    print(f'\nSaved report to {output_path.resolve()}')
if json_output_path is not None:
    metrics = {
        'metric': 'BPT = UTF-8 bytes / ordinary tokens; no BOS or chat template',
        'samples': sample_info,
        'tokenizers': {
            name: {'vocab_size': vocab_sizes[name], 'samples': tokenizer_results[name]}
            for name in tokenizers
        },
    }
    try:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except OSError as e:
        parser.error(f'Could not save JSON metrics to {json_output_path}: {e}')
    print(f'Saved JSON metrics to {json_output_path.resolve()}')
