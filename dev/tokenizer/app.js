'use strict';
const encoder = new TextEncoder();
const payload = JSON.parse(document.getElementById('tokenizer-data').textContent);
const MODELS = payload.models.map(({key,name,file,group,vocab}) => ({key,name,file,group,vocab}));
const CODE = 'def parseHTTPResponse(user_id=42):\n    return f"user_{user_id}"  # 保留空格与换行\n';
const EXAMPLES = {code:CODE, mixed:'Hello, tokenizer! 今天用 Python 处理文本：你好，世界🙂。\n한국어도 테스트합니다. café ≠ cafe\u0301;', numbers:'2026-09-03 · 3.14159 · 1234567890\n公式：x^2 + y^2 = z^2\nLaTeX: \\frac{a+b}{2}'};
const COLORS = [['#dfece2','#486252'],['#e8e1f0','#6d5d83'],['#f4e6d5','#8c6c49'],['#dceaf1','#4d7182'],['#f1e0e6','#906678'],['#e8edce','#737e48']];
const input = document.getElementById('text-input');
const state = {model:MODELS[0].key, whitespace:true, tokens:[], graphemes:[], groups:[], text:'', ready:false, pending:true, error:null};
let updateTimer, toastTimer;
let textButtons = [], idButtons = [], activeHighlight = null, pointerHighlight = null, focusHighlight = null;
function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function hex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, '0').toUpperCase()).join(' '); }
function selectOptions(select, value) {
  for (const group of ['GPT 参照', '本地训练']) {
    const optgroup = element('optgroup');
    optgroup.label = group;
    for (const model of MODELS.filter(m => m.group === group)) {
      const option = element('option', '', model.name);
      option.value = model.key;
      optgroup.append(option);
    }
    select.append(optgroup);
  }
  select.value = value;
}
// Use complete graphemes for display and UTF-8 byte ranges for token membership.
function segmentText(text) {
  let offset = 0;
  return Array.from(new Intl.Segmenter('zh', {granularity:'grapheme'}).segment(text), part => {
    const start = offset;
    offset += encoder.encode(part.segment).length;
    return {text:part.segment, start, end:offset};
  });
}
function overlaps(a, b) { return a.start < b.end && a.end > b.start; }
function mapGlyphs(graphemes, tokens) {
  let cursor = 0;
  return graphemes.map(glyph => {
    while (cursor < tokens.length && tokens[cursor].end <= glyph.start) cursor++;
    const indices = [];
    for (let i = cursor; i < tokens.length && tokens[i].start < glyph.end; i++) indices.push(i);
    return {...glyph, indices};
  });
}
function textGroups(glyphs) {
  const groups = [];
  for (const glyph of glyphs) {
    const previous = groups[groups.length - 1];
    // One hover target per token; retain complete graphemes when a character spans tokens.
    if (glyph.indices.length === 1 && previous?.indices.length === 1 &&
        previous.indices[0] === glyph.indices[0] && !previous.text.endsWith('\n')) {
      previous.text += glyph.text;
      previous.end = glyph.end;
    } else groups.push({...glyph});
  }
  return groups;
}
function visibleText(text) {
  if (state.whitespace) return text.replace(/ /g, '·').replace(/\t/g, '⇥').replace(/\r/g, '␍').replace(/\n/g, '↵');
  return text.replace(/\r?\n/g, '') || ' ';
}
function paintToken(el, index) {
  el.style.setProperty('--token-bg', COLORS[index % COLORS.length][0]);
  el.style.setProperty('--token-ink', COLORS[index % COLORS.length][1]);
}
function highlightMatches(selection, groups) {
  if (!selection) return {groupIndices:[], tokenIndices:[]};
  const tokenIndices = selection.kind === 'group' ? groups[selection.index].indices : [selection.index];
  // A split character remains one readable target. A single token highlights its whole text,
  // including any pieces displayed on separate lines.
  const groupIndices = selection.kind === 'group' && tokenIndices.length > 1
    ? [selection.index]
    : groups.flatMap((group, i) => group.indices.some(index => tokenIndices.includes(index)) ? [i] : []);
  return {groupIndices, tokenIndices};
}
function updateHighlight() {
  const selection = pointerHighlight || focusHighlight;
  const matches = highlightMatches(selection, state.groups);
  const highlightedGroups = new Set(matches.groupIndices), highlightedIds = new Set(matches.tokenIndices);
  textButtons.forEach((button, i) => button.classList.toggle('is-highlighted', highlightedGroups.has(i)));
  idButtons.forEach((button, i) => button.classList.toggle('is-highlighted', highlightedIds.has(i)));
  document.getElementById('paired-output')?.classList.toggle('has-highlight', !!selection);
  const changed = selection && (activeHighlight?.kind !== selection.kind || activeHighlight?.index !== selection.index);
  if (changed) {
    const target = selection.kind === 'group' ? idButtons[matches.tokenIndices[0]] : textButtons[matches.groupIndices[0]];
    const scroller = document.getElementById(selection.kind === 'group' ? 'id-tokens' : 'text-tokens');
    if (target && scroller) {
      const rect = target.getBoundingClientRect(), viewport = scroller.getBoundingClientRect();
      // Scroll only the opposite pane; do not move the page underneath the pointer.
      if (rect.top < viewport.top) scroller.scrollTop -= viewport.top - rect.top + 12;
      else if (rect.bottom > viewport.bottom) scroller.scrollTop += rect.bottom - viewport.bottom + 12;
    }
  }
  activeHighlight = selection;
}
function bindHighlight(button, selection) {
  button.addEventListener('pointerenter', event => {
    if (event.pointerType !== 'touch') { pointerHighlight = selection; updateHighlight(); }
  });
  button.addEventListener('pointerleave', () => {
    if (pointerHighlight === selection) { pointerHighlight = null; updateHighlight(); }
  });
  button.addEventListener('focus', () => { focusHighlight = selection; updateHighlight(); });
  button.addEventListener('blur', () => {
    if (focusHighlight === selection) { focusHighlight = null; updateHighlight(); }
  });
  // Tapping a text piece also exposes its IDs on devices without hover.
  button.addEventListener('click', event => { if (event.detail) button.focus(); });
}
function renderTextGroup(group, index) {
  const button = element('button', 'text-piece' + (group.indices.length > 1 ? ' multi-token' : ''), visibleText(group.text));
  paintToken(button, group.indices[0]);
  button.type = 'button';
  button.dataset.groupIndex = index;
  const ids = group.indices.map(i => state.tokens[i].id).join(', ');
  button.setAttribute('aria-label', JSON.stringify(group.text) + ('，对应 token ID：' + ids));
  if (group.indices.length > 1) {
    const count = element('sup', 'token-count', group.indices.length);
    count.setAttribute('aria-hidden', 'true');
    const bar = element('span', 'token-parts');
    bar.setAttribute('aria-hidden', 'true');
    for (const i of group.indices) {
      const part = element('i');
      part.style.background = COLORS[i % COLORS.length][1];
      part.style.flex = Math.min(state.tokens[i].end, group.end) - Math.max(state.tokens[i].start, group.start);
      bar.append(part);
    }
    button.append(count, bar);
  }
  bindHighlight(button, {kind:'group', index});
  textButtons.push(button);
  return button;
}
function render() {
  const model = MODELS.find(m => m.key === state.model);
  state.graphemes = state.tokens.length ? mapGlyphs(segmentText(state.text), state.tokens) : [];
  state.groups = textGroups(state.graphemes);
  textButtons = []; idButtons = [];
  activeHighlight = null; pointerHighlight = null; focusHighlight = null;
  document.getElementById('char-count').textContent = Array.from(input.value).length;
  document.getElementById('byte-count').textContent = encoder.encode(input.value).length;
  const error = document.getElementById('error-message');
  error.hidden = !state.error;
  error.textContent = state.error || '';
  document.querySelectorAll('[data-sample]').forEach(button => {
    const active = input.value === EXAMPLES[button.dataset.sample];
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active);
  });
  document.getElementById('output-card').replaceChildren(renderCard(model));
}
function renderCard(model) {
  const tokens = state.tokens;
  const card = element('article', 'card tokenizer-card');
  const top = element('div', 'tokenizer-top');
  const modelControl = element('div', 'model-control');
  const label = element('label', 'tokenizer-label', 'TOKENIZER');
  label.htmlFor = 'model-select';
  const wrap = element('div', 'model-select-wrap');
  const select = element('select');
  select.id = 'model-select';
  selectOptions(select, model.key);
  select.addEventListener('change', () => { state.model = select.value; queueEncoding(); });
  wrap.append(select);
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  arrow.setAttribute('width', '12'); arrow.setAttribute('height', '12'); arrow.setAttribute('viewBox', '0 0 16 16');
  arrow.innerHTML = '<path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.4"/>';
  wrap.append(arrow);
  const subtitle = element('div', 'model-subtitle', model.file + ' · ' + model.vocab.toLocaleString() + ' vocab');
  subtitle.title = subtitle.textContent;
  modelControl.append(label, wrap, subtitle);
  const metrics = element('div', 'metrics');
  const count = element('div');
  count.append(element('div', 'metric-value', state.pending ? '…' : state.error ? '—' : tokens.length), element('div', 'metric-label', 'TOKENS'));
  const bpt = element('div');
  bpt.append(element('div', 'metric-value', !state.pending && tokens.length ? (encoder.encode(state.text).length / tokens.length).toFixed(2) : '—'),
    element('div', 'metric-label', 'BYTES / TOKEN'));
  bpt.title = 'UTF-8 字节数 ÷ token 数';
  metrics.append(count, element('span', 'metric-divider'), bpt);
  top.append(modelControl, metrics);
  card.append(top);
  const paired = element('div', 'paired-output');
  paired.id = 'paired-output';
  const textPane = element('section', 'result-pane');
  textPane.setAttribute('aria-label', '文本');
  const textHeading = element('div', 'pane-heading');
  textHeading.append(element('span', '', '文本'));
  const textArea = element('div', 'token-area text-area');
  textArea.id = 'text-tokens';
  state.groups.forEach((group, index) => {
    textArea.append(renderTextGroup(group, index));
    if (group.text.endsWith('\n')) textArea.append(element('span', 'token-linebreak'));
  });
  const idPane = element('section', 'result-pane');
  idPane.setAttribute('aria-label', 'Token IDs');
  const idHeading = element('div', 'pane-heading');
  idHeading.append(element('span', '', 'Token IDs'));
  const copy = element('button', 'copy-button', '⧉ 复制 IDs');
  copy.disabled = state.pending || !tokens.length;
  copy.addEventListener('click', () => copyText(JSON.stringify(tokens.map(t => t.id))));
  idHeading.append(copy);
  const idArea = element('div', 'token-area id-area');
  idArea.id = 'id-tokens';
  tokens.forEach((token, index) => {
    const button = element('button', 'id-token', token.id);
    button.type = 'button';
    button.dataset.tokenIndex = index;
    paintToken(button, index);
    const text = state.graphemes.filter(g => g.indices.includes(index)).map(g => g.text).join('');
    button.title = ('ID ' + token.id) + ' · 对应文字 ' + JSON.stringify(text) +
      '\nUTF-8: ' + hex(token.bytes) + ' · [' + token.start + ', ' + token.end + ')';
    button.setAttribute('aria-label', ('Token ID ' + token.id) + '，对应 ' + JSON.stringify(text));
    bindHighlight(button, {kind:'token', index});
    idButtons.push(button);
    idArea.append(button);
  });
  if (!tokens.length) {
    textArea.append(element('div', 'empty-state', state.pending ? '正在编码…' : state.error ? '编码未完成' : '输入文本后显示分词'));
    idArea.append(element('div', 'empty-state', state.pending ? '正在编码…' : state.error ? '—' : '对应的 ID 会显示在这里'));
  }
  textPane.append(textHeading, textArea);
  idPane.append(idHeading, idArea);
  paired.append(textPane, idPane);
  card.append(paired);
  return card;
}
function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 2400);
}
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const temp = element('textarea');
      temp.value = text;
      temp.style.position = 'fixed'; temp.style.opacity = '0';
      document.body.append(temp); temp.select();
      const copied = document.execCommand('copy');
      temp.remove();
      if (!copied) throw Error('copy unavailable');
    }
    toast('Token IDs 已复制');
  } catch { toast('浏览器未允许复制，可直接选择右侧 IDs 复制'); }
}
let worker, workerURL, requestId = 0;
function queueEncoding(delay = 0) {
  const id = ++requestId;
  clearTimeout(updateTimer);
  if (!state.ready && state.error) { render(); return; }
  state.pending = true;
  state.error = null;
  state.tokens = [];
  state.text = input.value;
  render();
  if (!state.ready) return;
  updateTimer = setTimeout(() => worker.postMessage({type:'encode', id, model:state.model, text:input.value}), delay);
}
function showError(message) {
  state.pending = false;
  state.error = message;
  state.tokens = [];
  render();
}
input.value = CODE;
input.addEventListener('input', () => queueEncoding(120));
document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', () => {
  input.value = EXAMPLES[button.dataset.sample]; queueEncoding();
}));
document.getElementById('clear-button').addEventListener('click', () => {
  input.value = ''; queueEncoding(); input.focus();
});
document.getElementById('whitespace').addEventListener('change', event => { state.whitespace = event.target.checked; render(); });
document.getElementById('about-button').addEventListener('click', () => document.getElementById('about-dialog').showModal());
document.getElementById('close-dialog').addEventListener('click', () => document.getElementById('about-dialog').close());
render();
try {
  workerURL = URL.createObjectURL(new Blob([document.getElementById('tokenizer-worker').textContent], {type:'text/javascript'}));
  worker = new Worker(workerURL);
  worker.onmessage = ({data}) => {
    if (data.type === 'ready') {
      URL.revokeObjectURL(workerURL);
      state.ready = true;
      queueEncoding();
    } else if (data.type === 'error' && (data.id === undefined || data.id === requestId)) {
      showError(data.message);
    } else if (data.type === 'result' && data.id === requestId) {
      state.pending = false;
      state.text = data.text;
      state.tokens = data.tokens;
      render();
    }
  };
  worker.onerror = event => {
    URL.revokeObjectURL(workerURL);
    state.ready = false;
    showError('分词引擎无法运行：' + (event.message || '请使用支持 WebAssembly 的现代浏览器。'));
  };
  worker.postMessage({type:'init', wasm:payload.wasm, models:payload.models});
} catch (error) {
  if (workerURL) URL.revokeObjectURL(workerURL);
  showError('分词引擎无法启动：' + error.message);
}
