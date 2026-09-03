# tiktoken browser runtime

Vendored from [`tiktoken@1.0.22`](https://registry.npmjs.org/tiktoken/-/tiktoken-1.0.22.tgz), the [dqbd/tiktoken](https://github.com/dqbd/tiktoken) JS/WASM bindings for OpenAI tiktoken. Both files are unchanged from the package's `lite/` directory. The MIT license is in `LICENSE` and is also embedded in exported HTML.

| File | SHA-256 |
| --- | --- |
| `tiktoken_bg.js` | `eebed1618f6cd2a9c1be475c698eebcac369c8bcff52206804b98ca67068b9c5` |
| `tiktoken_bg.wasm` | `870fa4e0d8fe30a02b4703b2c4e24e7a8d99dcf6a1e8416c14b239f52a2eed55` |

Package SHA-512 (base64): `PKvy1rVF1RibfF3JlXBSP0Jrcw2uq3yXdgcEXtKTYn3QJ/cBRBHDnrJ5jHky+MENZ6DIPwNUGWpkVx+7joCpNA==`.

`scripts.tok_export` wraps the JS exports in a closure and embeds the WASM as base64. Runtime initialization uses `WebAssembly.instantiate`, without fetching files or importing modules. All vocabularies come from the installed Python tiktoken package and selected local `.pkl` files, not the npm package. `encode_ordinary` preserves the evaluation script's treatment of literal special-token strings.

To update, replace both vendor files together, update these hashes, and run `tests/test_tok_export.py`, including its Node parity checks. No frontend package manager or bundler is required for export.
