/* The same engine runs in the inline browser worker and in the Node parity tests. */
globalThis.TokenizerEngine = class TokenizerEngine {
  async init(wasmBase64, models) {
    const bytes = Uint8Array.from(atob(wasmBase64), c => c.charCodeAt(0));
    const result = await WebAssembly.instantiate(bytes, {'./tiktoken_bg.js': TiktokenBindings});
    TiktokenBindings.__wbg_set_wasm(result.instance.exports);
    this.models = new Map(models.map(model => [model.key, model]));
    this.currentKey = null;
    this.encoding = null;
  }

  select(key) {
    if (this.currentKey === key) return this.encoding;
    const model = this.models.get(key);
    if (!model) throw new Error('找不到 tokenizer：' + key);
    this.encoding?.free();
    this.encoding = null;
    this.currentKey = null;
    const encoding = new TiktokenBindings.Tiktoken(model.ranks, model.special_tokens, model.pattern);
    try {
      for (const check of model.checks) {
        const actual = encoding.encode_ordinary(check.text);
        if (actual.length !== check.ids.length || actual.some((id, i) => id !== check.ids[i])) {
          throw new Error(model.name + ' 与 Python 编码结果不一致，请检查 tokenizer 和分词引擎版本。');
        }
      }
    } catch (error) {
      encoding.free();
      throw error;
    }
    this.encoding = encoding;
    this.currentKey = key;
    return encoding;
  }

  encode(key, text) {
    const encoding = this.select(key);
    const ids = encoding.encode_ordinary(text);
    const source = new TextEncoder().encode(text);
    let offset = 0;
    const pieces = new Map();
    const tokens = Array.from(ids, id => {
      if (!pieces.has(id)) pieces.set(id, encoding.decode_single_token_bytes(id));
      const bytes = pieces.get(id);
      const start = offset;
      for (const byte of bytes) {
        if (source[offset++] !== byte) throw new Error('分词结果无法还原原文。');
      }
      return {id, bytes, start, end:offset};
    });
    if (offset !== source.length) throw new Error('分词结果的字节数与原文不一致。');
    return tokens;
  }
};
