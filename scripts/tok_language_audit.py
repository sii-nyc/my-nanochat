"""Estimate the dominant language of documents consumed by a tokenizer run.

The run manifest supplies the exact shard order, document cap, and character
cutoffs used by scripts.tok_train. Language labels are fastText predictions,
not ground-truth annotations; code and mixed-language documents need care.

Example (run on the machine that holds the Parquet data):
    uv run --with fasttext-wheel==0.9.2 python -m scripts.tok_language_audit \
      --run-json training/tokenizer_experiments/tokenizer-20261003/public/run.json \
      --model training/lid.176.ftz
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_training_documents(data_dir, shard_names):
    import pyarrow.parquet as pq

    for name in shard_names:
        parquet = pq.ParquetFile(data_dir / name)
        for row_group in range(parquet.num_row_groups):
            for doc in parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist():
                yield name, doc


def counts_snapshot(char_counts, doc_counts, total_chars, total_docs):
    labels = sorted(char_counts | doc_counts, key=lambda label: (-char_counts[label], label))
    return {
        "total_chars": total_chars,
        "total_docs": total_docs,
        "languages": {
            label: {
                "chars": char_counts[label],
                "chars_pct": 100 * char_counts[label] / total_chars if total_chars else 0,
                "docs": doc_counts[label],
                "docs_pct": 100 * doc_counts[label] / total_docs if total_docs else 0,
            }
            for label in labels
        },
    }


def audit(documents, targets, doc_cap, model, min_confidence):
    raw_chars, raw_docs = Counter(), Counter()
    confident_chars, confident_docs = Counter(), Counter()
    total_chars = total_docs = 0
    cutoffs = {}
    pending = sorted(targets)
    for shard_name, full_text in documents:
        text = full_text[:doc_cap]
        length = len(text)
        # fastText requires a single line; this does not change the char budget.
        prediction = text.replace("\n", " ").replace("\r", " ").replace("\x00", " ")
        if prediction.strip():
            labels, scores = model.predict(prediction, k=1)
            label, score = labels[0].removeprefix("__label__"), float(scores[0])
            confident_label = label if score >= min_confidence else "undetermined"
        else:
            label = confident_label = "undetermined"

        raw_chars[label] += length
        raw_docs[label] += 1
        confident_chars[confident_label] += length
        confident_docs[confident_label] += 1
        total_chars += length
        total_docs += 1
        if total_docs % 100_000 == 0:
            print(f"Scanned {total_docs:,} documents / {total_chars:,} characters", flush=True)

        # tok_train yields the whole document before checking whether it passed
        # the limit, so the cutoff has a small, reproducible overshoot.
        while pending and total_chars > pending[0]:
            target = pending.pop(0)
            cutoffs[target] = {
                "last_shard": shard_name,
                "raw": counts_snapshot(raw_chars, raw_docs, total_chars, total_docs),
                "confident": counts_snapshot(
                    confident_chars, confident_docs, total_chars, total_docs
                ),
            }
            if not pending:
                return cutoffs
    raise ValueError(f"Training text ended before the {pending[0]:,}-character cutoff")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True, help="run.json from the tokenizer experiment")
    parser.add_argument("--model", type=Path, required=True, help="fastText lid.176.ftz or lid.176.bin")
    parser.add_argument("--data-dir", type=Path, help="Override the dataset path recorded in run.json")
    parser.add_argument("--output", type=Path, help="Default: language_audit.json next to run.json")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")

    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    data_dir = args.data_dir or Path(run["data"]["directory"])
    training = run["training"]
    doc_caps = {item["doc_cap"] for item in training}
    if len(doc_caps) != 1:
        parser.error("This run contains different document caps")
    doc_cap = doc_caps.pop()
    targets = {item["max_chars"]: item["actual_chars"] for item in training}
    shard_names = run["data"]["training_shards"]
    if not data_dir.is_dir() or not args.model.is_file():
        parser.error("The dataset directory and fastText model must exist on this machine")
    try:
        import fasttext
    except ImportError:
        parser.error("Install the classifier for this run with: uv run --with fasttext-wheel==0.9.2 python -m scripts.tok_language_audit ...")
    model = fasttext.load_model(str(args.model))

    # The consumed shard hashes in run.json tie this audit to the exact data
    # from the tokenizer experiment, beyond a matching character count.
    last_shard = run["data"]["training_shard_cutoffs"][str(max(targets))]["last_shard"]
    used_shards = shard_names[:shard_names.index(last_shard) + 1]
    manifest = {item["name"]: item for item in run["data"]["files"]}
    print(f"Verifying {len(used_shards)} consumed shards against run.json...", flush=True)
    for name in used_shards:
        path = data_dir / name
        expected = manifest[name].get("sha256")
        if not path.is_file() or not expected or sha256(path) != expected:
            parser.error(f"Missing or changed training shard: {path}")
    print(f"Verified {len(used_shards)} consumed shards against run.json", flush=True)
    cutoffs = audit(
        iter_training_documents(data_dir, shard_names), targets, doc_cap,
        model, args.min_confidence,
    )
    for target, cutoff in cutoffs.items():
        actual = cutoff["raw"]["total_chars"]
        if actual != targets[target]:
            raise ValueError(
                f"At {target:,} chars the corpus reached {actual:,}, "
                f"but run.json records {targets[target]:,}; check shard contents and order"
            )
        expected_shard = run["data"]["training_shard_cutoffs"][str(target)]["last_shard"]
        if cutoff["last_shard"] != expected_shard:
            raise ValueError(f"At {target:,} chars the last shard differs from run.json")

    result = {
        "run_id": run["run_id"],
        "code_commit_of_training": run["code_commit"],
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "document_cap_chars": doc_cap,
        "classifier": {
            "model": args.model.name,
            "model_sha256": sha256(args.model),
            "analysis_script_sha256": sha256(Path(__file__)),
            "min_confidence": args.min_confidence,
            "method": "One fastText dominant-language prediction per capped document; character-weighted shares use Python string length.",
        },
        "cutoffs": {str(target): cutoffs[target] for target in sorted(cutoffs)},
    }
    output = args.output or args.run_json.with_name("language_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    for target in sorted(cutoffs):
        cutoff = cutoffs[target]
        print(f"\n{target:,} target / {cutoff['raw']['total_chars']:,} actual chars / {cutoff['raw']['total_docs']:,} docs")
        print("Raw predictions (share of consumed characters):")
        for label, values in list(cutoff["raw"]["languages"].items())[:15]:
            print(f"  {label:16s} {values['chars_pct']:8.3f}%  {values['docs']:,} docs")
        uncertain = cutoff["confident"]["languages"].get("undetermined", {}).get("chars_pct", 0)
        print(f"Below {args.min_confidence:.0%} confidence or empty: {uncertain:.3f}% of characters")


if __name__ == "__main__":
    main()
