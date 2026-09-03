"""Export GPT references and local tokenizers into one offline HTML explorer."""

import argparse
import base64
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

import tiktoken

from nanochat.tokenizer import get_tokenizer_paths


ASSETS = Path(__file__).resolve().parents[1] / "dev" / "tokenizer"
PROBES = [
    "",
    'def parseHTTPResponse(user_id=42):\n    return f"user_{user_id}"  # 保留空格与换行\n',
    "Hello, tokenizer! 今天用 Python 处理文本：你好，世界🙂。\n한국어도 테스트합니다.",
    "café ≠ cafe\u0301; 2026-09-03, 3.14159. 👩‍💻 🇨🇳",
    "I'm I'M we'll WE'LL 1234567890\r\n\t  \u00a0 \u2003 \ufeff\n\n",
    r"LaTeX: \frac{a+b}{2} <|endoftext|> <|bos|> <|user_start|>",
]


def export_encoding(enc, key, name, filename, group):
    """Preserve the encoding's actual pattern, byte ranks and special-token metadata."""
    if not isinstance(enc, tiktoken.Encoding):
        raise ValueError(f"{filename} does not contain a tiktoken.Encoding")
    # tiktoken exposes these via its pickle state, but has no public export API.
    ranks = enc._mergeable_ranks
    return {
        "key": key,
        "name": name,
        "file": filename,
        "group": group,
        "vocab": enc.n_vocab,
        "pattern": enc._pat_str,
        "special_tokens": enc._special_tokens,
        "ranks": "\n".join(
            f"{base64.b64encode(piece).decode('ascii')} {rank}"
            for piece, rank in sorted(ranks.items(), key=lambda item: item[1])
        ),
        "checks": [{"text": text, "ids": enc.encode_ordinary(text)} for text in PROBES],
    }


def collect_models(tokenizer_dir, filename=None):
    if filename is None:
        paths = sorted(path for path in tokenizer_dir.glob("*.pkl") if path.is_file())
    else:
        paths = [Path(get_tokenizer_paths(tokenizer_dir, filename)[0])]
    # Load requested local files before downloading any reference vocabularies.
    local = []
    for index, path in enumerate(paths):
        try:
            with path.open("rb") as stream:
                enc = pickle.load(stream)
            local.append(export_encoding(enc, f"local-{index}", path.name, path.name, "本地训练"))
        except Exception as exc:
            raise ValueError(f"Could not export tokenizer {path}: {exc}") from exc
    references = [
        export_encoding(tiktoken.get_encoding(encoding), key, name, encoding, "GPT 参照")
        for key, name, encoding in (("gpt4", "GPT-4", "cl100k_base"), ("gpt2", "GPT-2", "gpt2"))
    ]
    return references + local


def browser_bindings():
    """Wrap the unchanged, pinned wasm-bindgen module for a classic inline worker."""
    source = (ASSETS / "vendor" / "tiktoken_bg.js").read_text(encoding="utf-8")
    exports = re.findall(r"^export (?:function|class) (\w+)", source, flags=re.MULTILINE)
    source = re.sub(r"^export ", "", source, flags=re.MULTILINE)
    return "const TiktokenBindings = (() => {\n" + source + "\nreturn {" + ",".join(exports) + "};\n})();"


def build_html(models):
    payload = {
        "version": 1,
        "models": models,
        "wasm": base64.b64encode((ASSETS / "vendor" / "tiktoken_bg.wasm").read_bytes()).decode("ascii"),
    }
    # Script elements terminate on </script> even when their type is application/json.
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    worker = "\n".join([
        browser_bindings(),
        (ASSETS / "engine.js").read_text(encoding="utf-8"),
        (ASSETS / "worker.js").read_text(encoding="utf-8"),
    ])
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    license_text = (ASSETS / "vendor" / "LICENSE").read_text(encoding="utf-8")
    scripts = (
        f'<script type="application/json" id="tokenizer-data">{data}</script>\n'
        f'<script type="text/plain" id="tokenizer-worker">{worker}</script>\n'
        f'<script>{app}</script>\n'
        f'<!-- Bundled tiktoken 1.0.22, https://github.com/dqbd/tiktoken\n{license_text}-->'
    )
    return (ASSETS / "page.html").read_text(encoding="utf-8").replace("<!--TOKENIZER_ASSETS-->", scripts)


def write_html(path, html):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR") or "~/.cache/nanochat").expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-file", help="Export one filename (.pkl optional); default: all top-level .pkl files. GPT-2 and GPT-4 are always included.")
    parser.add_argument("--output", type=Path, default=base_dir / "tokenizer.html", help="Output HTML path (.html optional). Default: NANOCHAT_BASE_DIR/tokenizer.html; parent directories are created.")
    args = parser.parse_args(argv)
    path = args.output.expanduser()
    if not path.name or path.name == ".." or path.is_dir():
        parser.error("--output must name an HTML file, not a directory")
    if path.suffix.lower() != ".html":
        path = path.with_name(path.name + ".html")
    try:
        models = collect_models(base_dir / "tokenizer", args.tokenizer_file)
        html = build_html(models)
        write_html(path, html)
    except Exception as exc:
        parser.error(str(exc))
    for model in models:
        print(f"Included {model['name']} ({model['vocab']:,} vocab)")
    if len(models) == 2:
        print(f"No local .pkl files found in {base_dir / 'tokenizer'}; included GPT references only.")
    print(f"Saved {path.resolve()} ({path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
