"""Exercise tokenizer experiment files through the training and evaluation CLIs."""

import ast
import runpy
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.tokenizer import RustBPETokenizer, get_tokenizer, get_token_bytes


@pytest.mark.parametrize("filename", ["experiment.v1", "experiment.v1.pkl"])
def test_train_and_eval_named_tokenizer(filename, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    from nanochat import dataset

    data_dir = tmp_path / "base_data_climbmix"
    data_dir.mkdir()
    monkeypatch.setattr(dataset, "DATA_DIR", str(data_dir))
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "hello world, hello tokenizer, hello hello hello",
        "Numbers like 12345 and unicode like naïve café 你好 🙂 should survive.",
        "def f(x):\n    return x + 1\n",
    ] * 8
    train_path = data_dir / "shard_00000.parquet"
    pq.write_table(pa.table({"text": corpus}), train_path)
    pq.write_table(pa.table({"text": corpus[:2]}), data_dir / "shard_00001.parquet")

    monkeypatch.setattr(sys, "argv", ["tok_train", "--max-chars=10000", "--vocab-size=300"])
    runpy.run_module("scripts.tok_train", run_name="__main__")
    tokenizer_dir = tmp_path / "tokenizer"
    default_files = {path.name: path.read_bytes() for path in tokenizer_dir.iterdir()}
    assert set(default_files) == {"tokenizer.pkl", "token_bytes.pt"}

    # A second corpus and vocabulary must not overwrite either default artifact.
    pq.write_table(pa.table({"text": ["A different experiment with more Chinese 中文语料。"] * 16}), train_path)
    monkeypatch.setattr(sys, "argv", [
        "tok_train", "--max-chars=10000", "--vocab-size=280", f"--tokenizer-file={filename}",
    ])
    runpy.run_module("scripts.tok_train", run_name="__main__")
    assert {path.name for path in tokenizer_dir.iterdir()} == {
        "tokenizer.pkl", "token_bytes.pt", "experiment.v1.pkl", "experiment.v1.token_bytes.pt",
    }
    for name, original in default_files.items():
        assert (tokenizer_dir / name).read_bytes() == original

    default_tokenizer = get_tokenizer()
    custom_tokenizer = get_tokenizer(filename=filename)
    probe = "hello 中文 🙂"
    for tokenizer, token_bytes, vocab_size in (
        (default_tokenizer, get_token_bytes(), 300),
        (get_tokenizer(filename="tokenizer"), get_token_bytes(filename="tokenizer"), 300),
        (custom_tokenizer, get_token_bytes(filename=filename), 280),
    ):
        assert tokenizer.get_vocab_size() == vocab_size
        assert token_bytes.numel() == vocab_size
        assert token_bytes[tokenizer.get_bos_token_id()].item() == 0
        ids = tokenizer.encode(probe)
        assert tokenizer.decode(ids) == probe
        assert token_bytes[ids].sum().item() == len(probe.encode("utf-8"))

    # Discovery includes default and custom files, without confusing baseline names
    # or trying to load byte caches, backup files, or directories.
    default_tokenizer.save(tokenizer_dir, filename="GPT-2.pkl")
    (tokenizer_dir / "notes.txt").write_text("not a tokenizer")
    (tokenizer_dir / "backup.pkl.tmp").write_text("partial file")
    (tokenizer_dir / "directory.pkl").mkdir()

    # Keep the test offline; baseline downloads are unrelated to file selection.
    baseline_calls = []
    def load_baseline(cls, name):
        baseline_calls.append(name)
        return default_tokenizer
    monkeypatch.setattr(RustBPETokenizer, "from_pretrained", classmethod(load_baseline))
    monkeypatch.chdir(tmp_path)
    original_paths = set(tmp_path.rglob('*'))
    for arguments, expected_local_sizes in (
        ([], {"GPT-2.pkl": 300, "experiment.v1.pkl": 280, "tokenizer.pkl": 300}),
        (["--tokenizer-file=tokenizer"], {"tokenizer.pkl": 300}),
        (["--tokenizer-file=experiment.v1"], {"experiment.v1.pkl": 280}),
        (["--tokenizer-file=experiment.v1.pkl"], {"experiment.v1.pkl": 280}),
    ):
        baseline_calls.clear()
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["tok_eval", *arguments])
        results = runpy.run_module("scripts.tok_eval", run_name="__main__")
        assert baseline_calls == ["gpt2", "cl100k_base"]
        assert results["vocab_sizes"] == {"GPT-2": 300, "GPT-4": 300, **expected_local_sizes}
        assert results["tokenizer_files"] == list(expected_local_sizes)
        assert list(results["tokenizer_results"]) == ["GPT-2", "GPT-4", *expected_local_sizes]
        output = capsys.readouterr().out
        assert output == results["report"]
        assert "| Tokenizer | Vocab size | news | korean | code | math | science | climbmix-train | climbmix-val |" in output
        assert "vs GPT-2 %" not in output and "vs GPT-4 %" not in output
        assert "Ratio (B/T)" not in output and "| Tokens |" not in output
        assert "## Qualitative comparison" in output
        assert results["qualitative_text"] in output
        assert set(tmp_path.rglob('*')) == original_paths  # No report by default.
        assert list(results["qualitative_results"]) == ["GPT-2", "GPT-4", *expected_local_sizes]
        for selected in expected_local_sizes:
            assert selected in output
            assert "climbmix-val" in results["tokenizer_results"][selected]
        for name, tokenizer in results["tokenizers"].items():
            sample = results["qualitative_results"][name]
            assert tokenizer.decode(sample["ids"]) == results["qualitative_text"]
            assert b''.join(sample["pieces"]) == results["qualitative_text"].encode("utf-8")
            assert f'### {name}' in output

    # Relative paths, automatic .md suffix, UTF-8 output and replacing a report.
    report_path = tmp_path / "reports" / "分词对比.md"
    for report_filename in ("reports/分词对比", "reports/分词对比.md"):
        if report_path.exists():
            report_path.write_text("old report", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["tok_eval", f"--output={report_filename}"])
        results = runpy.run_module("scripts.tok_eval", run_name="__main__")
        output = capsys.readouterr().out
        assert report_path.read_text(encoding="utf-8") == results["report"]
        assert output == results["report"] + f"\nSaved report to {report_path}\n"
        assert not (tmp_path / "reports" / "分词对比.md.md").exists()

    # A bad save target reports an error while keeping the screen report available.
    (tmp_path / "blocked").write_text("not a directory")
    monkeypatch.setattr(sys, "argv", ["tok_eval", "--output=blocked/report.md"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tok_eval", run_name="__main__")
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "## Qualitative comparison" in output.out
    assert "Could not save report" in output.err

    # Unicode, whitespace and partial UTF-8 tokens all have lossless displays.
    for piece in ("你好🙂".encode("utf-8"), b'\xe4\xbd', b'\xa0word', b'\n\t  ', b'"`|\\', b'\\xe4'):
        displayed = results["format_piece"](piece)
        restored = ast.literal_eval(displayed)
        assert (restored.encode("utf-8") if isinstance(restored, str) else restored) == piece
    assert results["format_piece"](b'\xe4\xbd') == r"b'\xe4\xbd'"
    assert results["format_piece"](b'\n  ') == r'"\n  "'
    assert results["format_segmentation"]([1, 2, 3], [b'hello world', b' ', b'\n'], width=20) == '1:"hello world"\n2:" " | 3:"\\n"'
    assert results["markdown_escape"]('a|b`[x]_*.pkl') == r'a\|b\`\[x\]\_\*.pkl'


@pytest.mark.parametrize("filename", ["", ".", "..", "/"])
def test_eval_invalid_output_filename(filename, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tok_eval", f"--output={filename}"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tok_eval", run_name="__main__")
    assert exc.value.code == 2
    assert "--output must name a Markdown file" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["scripts.tok_train", "scripts.tok_eval"])
@pytest.mark.parametrize("filename", ["../escape", "..", ".pkl", ""])
def test_invalid_tokenizer_filename(module, filename, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [module, f"--tokenizer-file={filename}"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(module, run_name="__main__")
    assert exc.value.code == 2
    assert "Tokenizer filename must" in capsys.readouterr().err
    assert not (tmp_path / "tokenizer").exists()


@pytest.mark.parametrize("filename", ["missing", "missing.pkl"])
def test_eval_missing_tokenizer_file(filename, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["tok_eval", f"--tokenizer-file={filename}"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tok_eval", run_name="__main__")
    assert exc.value.code == 2
    assert "missing.pkl" in capsys.readouterr().err


@pytest.mark.parametrize("directory_exists", [False, True])
def test_eval_no_discovered_tokenizers(directory_exists, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    if directory_exists:
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "token_bytes.pt").write_bytes(b"cache only")
        (tokenizer_dir / "directory.pkl").mkdir()
    monkeypatch.setattr(sys, "argv", ["tok_eval"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tok_eval", run_name="__main__")
    assert exc.value.code == 2
    assert "No tokenizer .pkl files found" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[], ["--tokenizer-file=broken"]])
def test_eval_unreadable_tokenizer(arguments, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "broken.pkl").write_bytes(b"not a tokenizer")
    monkeypatch.setattr(sys, "argv", ["tok_eval", *arguments])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tok_eval", run_name="__main__")
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Could not load tokenizer" in error and "broken.pkl" in error
