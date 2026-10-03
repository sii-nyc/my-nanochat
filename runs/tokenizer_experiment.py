"""
Train the tokenizer comparison from scratch, evaluate it, and publish reports.

Run from a clean Git checkout with the project environment installed:
    uv run python runs/tokenizer_experiment.py --data-dir /path/to/base_data_climbmix

The five tokenizer files and the offline HTML stay under the ignored training/
directory. By default, small reports are committed and pushed to the current
upstream branch. With --no-upload, the reports also stay under training/.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOC_CAP = 10_000
VALIDATION_SHARD = "shard_06542.parquet"
EXPERIMENTS = (
    ("tok_1b_chars", 1_000_000_000, 32_768),
    ("tok_2b_chars", 2_000_000_000, 32_768),
    ("tok_3b_chars", 3_000_000_000, 32_768),
    ("tok_2b_vocab16k", 2_000_000_000, 16_384),
    ("tok_2b_vocab64k", 2_000_000_000, 65_536),
)


class ExperimentError(Exception):
    pass


def training_arguments(name, chars, vocab):
    return [
        f"--max-chars={chars}", f"--doc-cap={DOC_CAP}",
        f"--vocab-size={vocab}", f"--tokenizer-file={name}",
    ]


def git(*args, check=True):
    result = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True
    )
    if check and result.returncode:
        raise ExperimentError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def git_ignored(path):
    result = git("check-ignore", "-q", "--no-index", str(path), check=False)
    if result.returncode not in (0, 1):
        raise ExperimentError(f"Could not check Git ignore rule for {path}")
    return result.returncode == 0


def data_manifest(data_dir):
    files = sorted(
        path for path in data_dir.iterdir()
        if path.name.endswith(".parquet") and path.is_file()
    )
    if len(files) < 2:
        raise ExperimentError(
            f"Need at least two Parquet shards in {data_dir}: one train and one validation"
        )
    if files[-1].name != VALIDATION_SHARD:
        raise ExperimentError(
            f"Expected the ClimbMix validation shard {VALIDATION_SHARD} as the last file in {data_dir}; "
            "download it before running this experiment"
        )
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]


def training_shards_for_targets(data_dir, shards, targets):
    """Mirror tok_train's document order and cap to find the needed shard prefix."""
    import pyarrow.parquet as pq

    pending = sorted(set(targets))
    cutoffs = {}
    used = []
    chars = 0
    for item in shards[:-1]:
        path = data_dir / item["name"]
        used.append(item["name"])
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            for doc in parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist():
                chars += len(doc[:DOC_CAP])
                while pending and chars > pending[0]:
                    target = pending.pop(0)
                    cutoffs[target] = {"last_shard": item["name"], "shards_used": len(used)}
                if not pending:
                    return cutoffs, used
    if pending and chars >= pending[-1]:
        # tok_train also accepts an exact target when the data ends there.
        for target in pending:
            cutoffs[target] = {"last_shard": used[-1], "shards_used": len(used)}
        return cutoffs, used
    raise ExperimentError(
        f"Training data reaches only {chars:,} characters after doc-cap={DOC_CAP}; "
        f"at least {pending[-1]:,} characters are needed"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    versions = {}
    for name in ("rustbpe", "tiktoken", "torch", "pyarrow", "requests"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_module(module, arguments, log_path, env):
    command = [sys.executable, "-m", module, *arguments]
    print(f"Running: {' '.join(command)}", flush=True)
    print(f"Log: {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:])
        raise ExperimentError(
            f"{module} exited with status {result.returncode}; full log: {log_path}\n{tail}"
        )
    return command


def training_result(name, chars, vocab, tokenizer_dir, log_path, command):
    log = log_path.read_text(encoding="utf-8")
    chars_match = re.search(r"^actual_chars:\s*([\d,]+)\s*$", log, re.MULTILINE)
    time_match = re.search(r"^Training time:\s*([\d.]+)s\s*$", log, re.MULTILINE)
    if not chars_match or not time_match:
        raise ExperimentError(f"Missing actual_chars or training time in {log_path}")
    actual = int(chars_match.group(1).replace(",", ""))
    if actual < chars:
        raise ExperimentError(
            f"{name} trained on {actual:,} characters, below its {chars:,} target. "
            f"Check the data shards; partial artifacts remain on the server."
        )
    token_path = tokenizer_dir / f"{name}.pkl"
    bytes_path = tokenizer_dir / f"{name}.token_bytes.pt"
    if not token_path.is_file() or not bytes_path.is_file():
        raise ExperimentError(f"Training did not produce both artifacts for {name}")
    return {
        "name": name,
        "max_chars": chars,
        "actual_chars": actual,
        "vocab_size": vocab,
        "doc_cap": DOC_CAP,
        "training_seconds": float(time_match.group(1)),
        "command": command,
        "artifacts": {
            token_path.name: {"bytes": token_path.stat().st_size, "sha256": sha256(token_path)},
            bytes_path.name: {"bytes": bytes_path.stat().st_size, "sha256": sha256(bytes_path)},
        },
    }


def write_results(public_dir, metadata, work_dir):
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Tokenizer comparison: {metadata['run_id']}",
        "",
        f"Code commit: {metadata['code_commit']}",
        f"Data directory on server: {metadata['data']['directory']}",
        f"Validation shard: {metadata['data']['validation_shard']}",
        f"Server-only artifacts: {work_dir / 'base' / 'tokenizer'}",
        f"Server-only offline viewer: {work_dir / 'base' / 'tokenizer.html'}",
        f"Server-only logs: {work_dir / 'logs'}",
        "",
        "All five tokenizers were trained from scratch from the same ordered dataset and split.",
        "For the BPT table and qualitative token splits, see [comparison.md](comparison.md).",
        "For exact byte/token counts and sample fingerprints, see [metrics.json](metrics.json).",
        "Commands, versions, shard manifest, consumed-shard hashes, durations and artifact hashes are in [run.json](run.json).",
        "",
        "| Tokenizer | Character target | Actual characters | Vocab size | BPE training seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in metadata["training"]:
        lines.append(
            f"| {item['name']} | {item['max_chars']:,} | {item['actual_chars']:,} "
            f"| {item['vocab_size']:,} | {item['training_seconds']:.2f} |"
        )
    lines += [
        "",
        "The train and validation BPT samples each cover only the first row group.",
        "BPT alone does not measure downstream language model quality.",
        "",
    ]
    (public_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    default_data_root = Path(os.environ.get("NANOCHAT_BASE_DIR") or REPO / "training")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_root / "base_data_climbmix",
        help="Existing ClimbMix Parquet directory (default: NANOCHAT_BASE_DIR/base_data_climbmix, or training/base_data_climbmix)",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=REPO / "training" / "tokenizer_experiments",
        help="Ignored server-only root for tokenizer files, logs and HTML",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="Repository directory for uploaded reports (default: docs/experiments/tokenizer/RUN_ID; unavailable with --no-upload)",
    )
    parser.add_argument("--run-id", help="Unique run name; default: UTC timestamp plus code commit")
    parser.add_argument("--no-upload", action="store_true", help="Keep reports with the other artifacts under the artifact root; do not commit or push")
    parser.add_argument("--plan", action="store_true", help="Show paths and commands without writing anything")
    args = parser.parse_args()

    data_dir = (REPO / args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"Data directory does not exist: {data_dir}")
    shards = data_manifest(data_dir)
    code_commit = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    if not branch:
        raise ExperimentError("Use a branch, not detached HEAD, for this experiment")
    upstream = git(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", check=False
    )
    if not args.no_upload and not args.plan and upstream.returncode:
        raise ExperimentError("Set an upstream branch before running with GitHub upload")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + code_commit[:7]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        parser.error("--run-id must use 1-80 letters, digits, underscores or hyphens")
    artifact_root = (REPO / args.artifact_root).expanduser().resolve()
    work_dir = artifact_root / run_id
    if args.no_upload:
        if args.results_dir:
            parser.error("--results-dir requires upload; --no-upload keeps reports in the artifact run directory")
        results_dir = work_dir / "public"
    else:
        results_dir = (
            (REPO / args.results_dir).expanduser().resolve()
            if args.results_dir else REPO / "docs" / "experiments" / "tokenizer" / run_id
        )
        if not inside(results_dir, REPO) or inside(results_dir, REPO / ".git"):
            parser.error("--results-dir must be inside the repository and outside .git")
        if inside(results_dir, artifact_root) or inside(work_dir, results_dir):
            parser.error("Results and server-only artifacts must use separate directories")
        if git_ignored(results_dir):
            parser.error(f"Results directory is Git-ignored: {results_dir}")
    if inside(artifact_root, REPO) and not git_ignored(work_dir):
        parser.error(f"Artifact directory must be Git-ignored: {work_dir}")
    if work_dir.exists() or (not args.no_upload and results_dir.exists()):
        parser.error("Run directory already exists; choose a new --run-id or --results-dir")

    plan = {
        "run_id": run_id,
        "data_dir": str(data_dir),
        "train_shards": len(shards) - 1,
        "validation_shard": shards[-1]["name"],
        "artifact_dir": str(work_dir),
        "results_dir": str(results_dir),
        "experiments": [
            {
                "name": name,
                "max_chars": chars,
                "vocab_size": vocab,
                "command": [
                    sys.executable, "-m", "scripts.tok_train",
                    *training_arguments(name, chars, vocab),
                ],
            }
            for name, chars, vocab in EXPERIMENTS
        ],
        "evaluation_command": [
            sys.executable, "-m", "scripts.tok_eval",
            "--output", str(work_dir / "public" / "comparison.md"),
            "--json-output", str(work_dir / "public" / "metrics.json"),
        ],
        "export_command": [
            sys.executable, "-m", "scripts.tok_export",
            "--output", str(work_dir / "base" / "tokenizer.html"),
        ],
        "upload": not args.no_upload,
    }
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    status = git("status", "--porcelain", "--untracked-files=normal").stdout
    if status.strip():
        raise ExperimentError(
            "Start from a clean Git checkout so the code commit identifies this experiment.\n"
            + status
        )
    if not args.no_upload:
        git("var", "GIT_AUTHOR_IDENT")
        git("var", "GIT_COMMITTER_IDENT")
    missing = [
        name for name in ("rustbpe", "tiktoken", "torch", "pyarrow", "requests")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ExperimentError(f"Install project dependencies in this Python environment: {', '.join(missing)}")

    import tiktoken
    try:
        for name in ("gpt2", "cl100k_base"):
            tiktoken.get_encoding(name)
    except Exception as error:
        raise ExperimentError(f"Could not load GPT reference tokenizer {name}: {error}") from error
    vendor_dir = REPO / "dev" / "tokenizer"
    required_assets = (
        "vendor/tiktoken_bg.js", "vendor/tiktoken_bg.wasm", "vendor/LICENSE",
        "engine.js", "worker.js", "app.js", "page.html",
    )
    missing_assets = [name for name in required_assets if not (vendor_dir / name).is_file()]
    if missing_assets:
        raise ExperimentError(f"Offline viewer assets are missing: {', '.join(missing_assets)}")

    print("Scanning training documents up to the 3B-character cutoff...", flush=True)
    cutoffs, used_shards = training_shards_for_targets(
        data_dir, shards, (chars for _, chars, _ in EXPERIMENTS)
    )
    # Hash only the shards that can affect this experiment, even if the data
    # directory contains the entire much larger ClimbMix collection.
    fingerprinted_shards = [*used_shards, shards[-1]["name"]]
    print(f"Hashing {len(fingerprinted_shards)} consumed/validation shards...", flush=True)
    shard_hashes = {name: sha256(data_dir / name) for name in fingerprinted_shards}
    if data_manifest(data_dir) != shards:
        raise ExperimentError("The Parquet shard list, size or mtime changed while hashing")

    started = datetime.now(timezone.utc).isoformat()
    work_dir.mkdir(parents=True)
    base_dir = work_dir / "base"
    base_dir.mkdir()
    (base_dir / "base_data_climbmix").symlink_to(data_dir, target_is_directory=True)
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir()
    log_dir = work_dir / "logs"
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base_dir))
    training = []
    for name, chars, vocab in EXPERIMENTS:
        log_path = log_dir / f"{name}.log"
        command = run_module(
            "scripts.tok_train",
            training_arguments(name, chars, vocab),
            log_path, env,
        )
        item = training_result(name, chars, vocab, tokenizer_dir, log_path, command)
        training.append(item)
        print(f"Completed {name}: {item['actual_chars']:,} characters")
    for target in {item["max_chars"] for item in training}:
        actual_counts = {
            item["actual_chars"] for item in training if item["max_chars"] == target
        }
        if len(actual_counts) != 1:
            raise ExperimentError(
                f"Runs with the same {target:,} character target read different amounts of data"
            )

    public_dir = work_dir / "public"
    public_dir.mkdir()
    comparison = public_dir / "comparison.md"
    metrics_path = public_dir / "metrics.json"
    eval_command = run_module(
        "scripts.tok_eval", ["--output", str(comparison), "--json-output", str(metrics_path)],
        log_dir / "eval.log", env
    )
    html_path = base_dir / "tokenizer.html"
    export_command = run_module(
        "scripts.tok_export", ["--output", str(html_path)], log_dir / "export.log", env
    )
    if not comparison.is_file() or not metrics_path.is_file() or not html_path.is_file():
        raise ExperimentError("Evaluation report, JSON metrics or offline HTML was not created")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_tokenizers = {"GPT-2", "GPT-4", *(f"{name}.pkl" for name, _, _ in EXPERIMENTS)}
    if set(metrics["tokenizers"]) != expected_tokenizers:
        raise ExperimentError("JSON metrics do not contain all five trained tokenizers and both GPT references")
    if "climbmix-val" not in metrics["samples"] or not metrics["samples"]["climbmix-val"]["utf8_bytes"]:
        raise ExperimentError("The validation sample is empty or missing from JSON metrics")
    if any(not item["samples"]["climbmix-val"]["tokens"] for item in metrics["tokenizers"].values()):
        raise ExperimentError("A tokenizer produced no tokens for the validation sample")
    if data_manifest(data_dir) != shards:
        raise ExperimentError("The Parquet shard list, size or mtime changed during the run")

    metadata = {
        "run_id": run_id,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "branch": branch,
        "upstream": upstream.stdout.strip() if upstream.returncode == 0 else None,
        "python": sys.version,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "package_versions": package_versions(),
        "data": {
            "directory": str(data_dir),
            "training_shards": [item["name"] for item in shards[:-1]],
            "validation_shard": shards[-1]["name"],
            "training_shard_cutoffs": cutoffs,
            "fingerprinted_shards": fingerprinted_shards,
            "files": [
                {**item, **({"sha256": shard_hashes[item["name"]]} if item["name"] in shard_hashes else {})}
                for item in shards
            ],
        },
        "training": training,
        "evaluation_command": eval_command,
        "export_command": export_command,
        "server_log_directory": str(log_dir),
        "offline_html_bytes": html_path.stat().st_size,
    }
    write_results(public_dir, metadata, work_dir)
    print(f"Server-only tokenizers and HTML: {base_dir}")
    if args.no_upload:
        print(f"Reports ready: {public_dir}")
        print("Upload skipped; all outputs remain in the ignored artifact run directory.")
        return
    shutil.copytree(public_dir, results_dir)
    print(f"Reports ready: {results_dir}")
    relative_results = str(results_dir.relative_to(REPO))
    git("add", "--", relative_results)
    git("commit", "-m", f"Record tokenizer comparison {run_id}")
    git("push")
    print(f"Uploaded reports to {upstream.stdout.strip()}")


if __name__ == "__main__":
    try:
        main()
    except ExperimentError as error:
        print(f"Experiment stopped: {error}", file=sys.stderr)
        sys.exit(1)
