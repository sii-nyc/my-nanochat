"""Offline export, escaping and actual WASM/Python encoding parity."""

import ast
import json
import os
import pickle
import random
import shutil
import subprocess
from pathlib import Path

import pytest
import tiktoken

from nanochat.tokenizer import RustBPETokenizer
from scripts import tok_export


@pytest.fixture(scope="module")
def trained():
    return RustBPETokenizer.train_from_iterator(tok_export.PROBES * 12, 512).enc


@pytest.fixture
def offline_references(monkeypatch, trained):
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: trained)


def test_export_discovery_and_html(tmp_path, monkeypatch, offline_references, trained, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    for filename in ("z.pkl", "a<>&.pkl"):
        with (directory / filename).open("wb") as stream:
            pickle.dump(trained, stream)
    (directory / "cache.token_bytes.pt").write_text("ignored")
    (directory / "backup.pkl.tmp").write_text("ignored")
    (directory / "directory.pkl").mkdir()
    models = tok_export.collect_models(directory)
    assert [model["name"] for model in models] == ["GPT-4", "GPT-2", "a<>&.pkl", "z.pkl"]
    assert models[2]["pattern"] == trained._pat_str
    assert models[2]["special_tokens"] == trained._special_tokens
    assert models[2]["vocab"] == trained.n_vocab

    # This is data even when it resembles executable HTML.
    models[2]["name"] = '</script><script>alert("x")</script>'
    html = tok_export.build_html(models)
    assert html.count("</script>") == 3
    assert '<script>alert("x")' not in html
    assert "\\u003c/script>" in html
    assert '<script src=' not in html and '<link ' not in html
    assert "WebAssembly.instantiate" in html
    assert "Copyright (c) 2022 OpenAI, Shantanu Jain" in html
    assert "SNAPSHOTS" not in html and "simulate(" not in html

    output = tmp_path / "reports" / "分词.html"
    for filename in (output.with_suffix(""), output):
        tok_export.main(["--tokenizer-file=z", "--output", str(filename)])
        assert output.exists()
        assert "z.pkl" in output.read_text()
        assert "a\\u003c>" not in output.read_text()
        assert not output.with_suffix(".html.html").exists()
    tok_export.main([])
    assert (tmp_path / "tokenizer.html").exists()
    assert "Saved" in capsys.readouterr().out


def test_reference_only_export(tmp_path, monkeypatch, offline_references, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tok_export.main([])
    assert (tmp_path / "tokenizer.html").exists()
    assert "included GPT references only" in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["../escape", ".", "..", "", "missing"])
def test_invalid_local_file_preserves_output(filename, tmp_path, monkeypatch, offline_references):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    output = tmp_path / "tokenizer.html"
    output.write_text("previous export")
    with pytest.raises(SystemExit) as exc:
        tok_export.main(["--tokenizer-file", filename])
    assert exc.value.code == 2
    assert output.read_text() == "previous export"


def test_invalid_pickle_fails_before_download(tmp_path, monkeypatch):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    (directory / "bad.pkl").write_bytes(pickle.dumps(123))
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="does not contain a tiktoken.Encoding"):
        tok_export.collect_models(directory)


@pytest.mark.parametrize("filename", [".", "..", "/"])
def test_invalid_output(filename, monkeypatch, tmp_path):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        tok_export.main(["--output", filename])
    assert exc.value.code == 2


def parity_texts():
    texts = list(tok_export.PROBES)
    # Reuse the evaluation's domain texts without executing its CLI or importing torch.
    tree = ast.parse((tok_export.ASSETS.parents[1] / "scripts" / "tok_eval.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") in
                {"news_text", "korean_text", "code_text", "math_text", "science_text"} for target in node.targets):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                texts.append(ast.literal_eval(node.value.func.value).strip())
    rng = random.Random(42)
    alphabet = "abcXYZ0123456789_'.,!<>|\\ \t\r\n你好世界한국어é́🙂👩‍💻🇨🇳\u00a0\u0085\u2003\u2028\u2029\ufeff"
    texts += ["".join(rng.choices(alphabet, k=rng.randrange(1, 180))) for _ in range(100)]
    texts += ["a" * 2000, "\n" * 400, " " * 256 + "\t", "🙂" * 200, "</script><b>&"]
    return texts


def run_node_parity(tmp_path, encodings):
    node = os.environ.get("NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the browser-engine parity check")
    models = [tok_export.export_encoding(enc, key, name, key, "GPT 参照" if key.startswith("gpt") else "本地训练")
              for key, name, enc in encodings]
    html_path = tmp_path / "tokenizer.html"
    html_path.write_text(tok_export.build_html(models), encoding="utf-8")
    texts = parity_texts()
    fixture = {key: [{"text": text, "ids": enc.encode_ordinary(text)} for text in texts]
               for key, name, enc in encodings}
    fixture_path = tmp_path / "expected.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    result = subprocess.run([node, str(Path(__file__).with_name("tokenizer_web_check.cjs")), str(html_path), str(fixture_path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "frontend checks passed" in result.stdout


def test_local_wasm_parity(tmp_path, trained):
    run_node_parity(tmp_path, [("local", "Trained 512", trained)])


def test_cached_gpt_wasm_parity(tmp_path, monkeypatch):
    # Use cached official vocabularies when present; never download in the test suite.
    import tiktoken.load

    def no_download(*args, **kwargs):
        raise FileNotFoundError("reference vocabulary is not cached")

    monkeypatch.setattr(tiktoken.load, "read_file", no_download)
    try:
        encodings = [("gpt4", "GPT-4", tiktoken.get_encoding("cl100k_base")),
                     ("gpt2", "GPT-2", tiktoken.get_encoding("gpt2"))]
    except FileNotFoundError:
        pytest.skip("Cache GPT vocabularies once with scripts.tok_export to run reference parity")
    run_node_parity(tmp_path, encodings)
