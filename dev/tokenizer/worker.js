const engine = new TokenizerEngine();
let ready;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      ready = engine.init(data.wasm, data.models);
      await ready;
      self.postMessage({type:'ready'});
      return;
    }
    await ready;
    const tokens = engine.encode(data.model, data.text);
    self.postMessage({type:'result', id:data.id, model:data.model, text:data.text, tokens});
  } catch (error) {
    self.postMessage({type:'error', id:data.id, message:error.message || String(error)});
  }
};
