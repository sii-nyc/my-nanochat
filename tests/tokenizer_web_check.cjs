// Runs the exported worker unchanged; DOM stubs test UI state, not browser layout.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const expected = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const payload = JSON.parse(html.match(/id="tokenizer-data">([\s\S]*?)<\/script>/)[1]);
const workerSource = html.match(/id="tokenizer-worker">([\s\S]*?)<\/script>/)[1];
const appSource = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const messages = [];
const self = {postMessage: message => messages.push(message)};
vm.runInNewContext(workerSource, {TextEncoder, TextDecoder, WebAssembly, atob, self});

async function send(data) {
  await self.onmessage({data});
  return messages.pop();
}

async function main() {
  assert.equal((await send({type:'init', ...payload})).type, 'ready');
  for (const model of payload.models) {
    for (const [id, check] of expected[model.key].entries()) {
      const result = await send({type:'encode', id, model:model.key, text:check.text});
      assert.equal(result.type, 'result', JSON.stringify(result));
      assert.deepEqual(Array.from(result.tokens, token => token.id), check.ids, model.name + ': ' + JSON.stringify(check.text));
      const bytes = Array.from(result.tokens).flatMap(token => Array.from(token.bytes));
      assert.deepEqual(bytes, Array.from(new TextEncoder().encode(check.text)));
    }
    console.log(model.name + ': Python/WASM parity passed');
  }

  // The actual async controller: stale replies must never overwrite newer input.
  const nodes = new Map(), outbound = [], timers = new Map();
  let timerId = 0, client;
  class El {
    constructor() {
      this.children = []; this.listeners = {}; this.dataset = {}; this.className = '';
      this.style = {setProperty(){}};
      this.classList = {
        toggle: (name, on) => {const s = new Set(this.className.split(' ').filter(Boolean)); on ? s.add(name) : s.delete(name); this.className = [...s].join(' ');},
        contains: name => this.className.split(' ').includes(name),
      };
    }
    set id(value) {this._id = value; nodes.set(value, this);} get id() {return this._id;}
    append(...items) {this.children.push(...items);} replaceChildren(...items) {this.children = items;}
    setAttribute() {} addEventListener(name, fn) {(this.listeners[name] ||= []).push(fn);}
    emit(name) {for (const fn of this.listeners[name] || []) fn({target:this, pointerType:'mouse'});}
    getBoundingClientRect() {return {top:0, bottom:100};}
  }
  for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {const el = new El(); el.id = id;}
  nodes.get('tokenizer-data').textContent = JSON.stringify(payload);
  nodes.get('tokenizer-worker').textContent = workerSource;
  const document = {createElement:()=>new El(), createElementNS:()=>new El(), getElementById:id=>nodes.get(id), querySelectorAll:()=>[]};
  const context = vm.createContext({
    document, TextEncoder, TextDecoder, Intl, assert,
    Blob:class {}, URL:{createObjectURL:()=>'', revokeObjectURL(){}},
    Worker:class {constructor(){client = this;} postMessage(data){outbound.push(data);}},
    setTimeout:fn=>{timers.set(++timerId, fn); return timerId;}, clearTimeout:id=>timers.delete(id),
  });
  vm.runInContext(appSource, context);
  client.onmessage({data:{type:'ready'}});
  function flush() {for (const [id, fn] of [...timers]) {timers.delete(id); fn();}}
  flush();
  const initial = outbound.pop();
  const initialResult = await send(initial);
  client.onmessage({data:initialResult});
  vm.runInContext(`
    assert.equal(state.pending, false);
    assert.equal(state.groups.map(g=>g.text).join(''), CODE);
    assert.equal(textButtons.length, state.groups.length);
    const full = state.groups.findIndex(g=>g.indices.length===1 && g.text.trim().length>1);
    assert.ok(full >= 0);
    assert.equal(textButtons[full].children.length, 0);
    textButtons[full].emit('pointerenter');
    const lit = buttons=>buttons.flatMap((b,i)=>b.classList.contains('is-highlighted')?[i]:[]);
    assert.deepEqual(lit(idButtons), state.groups[full].indices);
    textButtons[full].emit('pointerleave');
    assert.deepEqual(lit(idButtons), []);
    input.value = CODE + '🧪'; queueEncoding();
  `, context);
  flush();
  client.onmessage({data:await send(outbound.pop())});
  vm.runInContext(`
    const partial = state.groups.findIndex(g=>g.indices.length>1);
    assert.ok(partial >= 0);
    textButtons[partial].emit('pointerenter');
    assert.deepEqual(lit(idButtons), state.groups[partial].indices);
    textButtons[partial].emit('pointerleave');
    idButtons[state.groups[partial].indices[0]].emit('pointerenter');
    assert.ok(lit(textButtons).includes(partial));
    idButtons[state.groups[partial].indices[0]].emit('pointerleave');
    input.value = 'new text'; queueEncoding();
  `, context);
  // A reply to the previous input arrives while the new request is pending.
  client.onmessage({data:initialResult});
  vm.runInContext("assert.equal(state.pending,true); assert.equal(state.tokens.length,0)", context);
  flush();
  client.onmessage({data:await send(outbound.pop())});
  vm.runInContext("assert.equal(state.text,'new text'); assert.equal(state.pending,false); input.value=''; queueEncoding()", context);
  flush();
  client.onmessage({data:await send(outbound.pop())});
  vm.runInContext("assert.equal(state.tokens.length,0); assert.equal(state.groups.length,0); assert.equal(state.pending,false)", context);
  const failed = await send({type:'encode', id:-1, model:'missing', text:'x'});
  assert.equal(failed.type, 'error');
  console.log('frontend checks passed');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
