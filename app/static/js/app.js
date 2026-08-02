/* ============================================
   Argo Frontend — Modern Chat UI Logic
   ============================================ */

(() => {
  'use strict';

  // ===== State =====
  const state = {
    chats: [],
    currentChatId: null,
    messages: [],
    mode: 'chat', // 'chat' or 'agent'
    model: '',
    workspace: '',
    abortController: null,
    streaming: false,
    pendingFiles: [], // files queued for the next message (uploaded on send)
    settings: {
      base_url: '',
      has_api_key: false,
      model: '',
      temperature: 0.2,
      max_tokens: 8192,
      max_steps: 40,
      auto_approve: false,
      default_workspace: '',
    },
  };

  // ===== DOM helpers =====
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else if (k.startsWith('on') && typeof v === 'function') {
        e.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (k === 'html') e.innerHTML = v;
      else if (v !== null && v !== undefined) e.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      if (typeof c === 'string' || typeof c === 'number') {
        e.appendChild(document.createTextNode(c));
      } else {
        e.appendChild(c);
      }
    }
    return e;
  }

  function escapeHTML(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ===== Markdown (very small, no deps) =====
  function renderMarkdown(text) {
    if (!text) return '';
    let s = escapeHTML(text);

    // Code blocks
    s = s.replace(/```(\w*)\n([\s\S]*?)```/g, (m, lang, code) => {
      return `<pre><button class="copy-btn" onclick="navigator.clipboard.writeText(this.nextElementSibling.textContent)">کپی</button><code class="language-${lang}">${code}</code></pre>`;
    });
    // Inline code
    s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');

    // Italic
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/(^|[^_])_([^_\n]+)_/g, '$1<em>$2</em>');

    // Links
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Headers
    s = s.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    s = s.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    s = s.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Blockquote
    s = s.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Lists
    s = s.replace(/(^|\n)((?:- .+\n?)+)/g, (m, prefix, list) => {
      const items = list.trim().split('\n').map(l => `<li>${l.replace(/^- /, '')}</li>`).join('');
      return `${prefix}<ul>${items}</ul>`;
    });
    s = s.replace(/(^|\n)((?:\d+\. .+\n?)+)/g, (m, prefix, list) => {
      const items = list.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
      return `${prefix}<ol>${items}</ol>`;
    });

    // Horizontal rule
    s = s.replace(/^---+$/gm, '<hr>');

    // Line breaks
    s = s.replace(/\n\n/g, '</p><p>');
    s = s.replace(/\n/g, '<br>');
    s = '<p>' + s + '</p>';
    s = s.replace(/<p>(<h[1-3]>|<ul>|<ol>|<pre>|<blockquote>|<hr>)/g, '$1');
    s = s.replace(/(<\/h[1-3]>|<\/ul>|<\/ol>|<\/pre>|<\/blockquote>|<\/hr>)<\/p>/g, '$1');

    return s;
  }

  // ===== API =====
  async function api(method, path, body) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const r = await fetch(path, opts);
    if (!r.ok) {
      const t = await r.text();
      try {
        const j = JSON.parse(t);
        throw new Error(j.detail || j.message || t);
      } catch {
        throw new Error(t || `HTTP ${r.status}`);
      }
    }
    return r.json();
  }

  // ===== Chat management =====
  async function loadChats() {
    try {
      const r = await api('GET', '/api/chats');
      state.chats = r.chats || [];
      renderChatList();
    } catch (e) {
      console.error('Failed to load chats', e);
    }
  }

  function renderChatList() {
    const list = $('#chatList');
    list.innerHTML = '';
    if (state.chats.length === 0) {
      list.appendChild(el('div', { class: 'muted', style: { padding: '20px 12px', fontSize: '13px', textAlign: 'center' } },
        'چتی وجود ندارد. روی «چت جدید» بزن.'));
      return;
    }
    for (const c of state.chats) {
      const item = el('div', {
        class: 'chat-item' + (c.id === state.currentChatId ? ' active' : ''),
        onclick: (e) => {
          if (e.target.closest('.chat-action-btn')) return;
          openChat(c.id);
        },
      },
        el('div', { class: 'chat-item-title' }, c.title || 'بدون عنوان'),
        el('div', { class: 'chat-item-actions' },
          el('button', {
            class: 'chat-action-btn danger',
            title: 'حذف',
            onclick: (e) => { e.stopPropagation(); deleteChat(c.id); },
          },
            (() => {
              const s = el('span');
              s.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>';
              return s;
            })()
          )
        )
      );
      list.appendChild(item);
    }
  }

  async function newChat() {
    try {
      const chat = await api('POST', '/api/chats', {
        title: 'چت جدید',
        mode: state.mode,
        model: state.model || null,
        workspace: state.workspace || null,
      });
      state.currentChatId = chat.id;
      state.messages = [];
      await loadChats();
      renderMessages();
      $('#chatTitle').textContent = chat.title;
    } catch (e) {
      setStatus('error', e.message);
    }
  }

  async function openChat(id) {
    try {
      const chat = await api('GET', `/api/chats/${id}`);
      state.currentChatId = id;
      state.messages = chat.messages || [];
      state.mode = chat.mode || 'chat';
      state.model = chat.model || '';
      state.workspace = chat.workspace || '';
      renderChatList();
      renderMessages();
      $('#chatTitle').textContent = chat.title;
      updateModeUI();
    } catch (e) {
      setStatus('error', e.message);
    }
  }

  async function deleteChat(id) {
    if (!confirm('این چت حذف شود؟')) return;
    try {
      await api('DELETE', `/api/chats/${id}`);
      if (state.currentChatId === id) {
        state.currentChatId = null;
        state.messages = [];
        renderMessages();
        $('#chatTitle').textContent = 'چت جدید';
      }
      await loadChats();
    } catch (e) {
      setStatus('error', e.message);
    }
  }

  // ===== Messages rendering =====
  function renderMessages() {
    const wrap = $('#messages');
    wrap.innerHTML = '';

    if (state.messages.length === 0) {
      const empty = $('#emptyState');
      if (empty) {
        const clone = empty.cloneNode(true);
        wrap.appendChild(clone);
        // re-bind tip clicks
        wrap.querySelectorAll('.tip').forEach(t => {
          t.addEventListener('click', () => {
            $('#input').value = t.dataset.prompt;
            $('#input').focus();
            autoResize();
          });
        });
      }
      return;
    }

    for (const m of state.messages) {
      wrap.appendChild(renderMessage(m));
    }
    scrollToBottom();
  }

  function renderMessage(m) {
    const isUser = m.role === 'user';
    const wrapper = el('div', { class: 'message ' + (isUser ? 'user' : 'assistant') });

    const avatar = el('div', { class: 'avatar ' + (isUser ? 'user-avatar' : 'bot-avatar') },
      isUser ? 'شما' : 'A'
    );

    const body = el('div', { class: 'message-body' });
    body.appendChild(el('div', { class: 'message-role' }, isUser ? 'شما' : 'Argo'));

    const bubble = el('div', { class: 'bubble' });

    // Render tool calls as cards
    if (m.tool_calls && m.tool_calls.length > 0) {
      for (const tc of m.tool_calls) {
        bubble.appendChild(renderToolCard(tc, null, false));
      }
    }

    // Render text
    if (m.content) {
      const textDiv = el('div', { html: renderMarkdown(m.content) });
      bubble.appendChild(textDiv);
    }

    body.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    return wrapper;
  }

  function renderToolCard(toolCall, toolResult, isStreaming = false) {
    const name = toolCall.name;
    const args = toolCall.arguments || {};
    const argsStr = JSON.stringify(args, null, 2);

    const toolNames = {
      bash: '⌨️',
      read_file: '📖',
      write_file: '✏️',
      edit_file: '🔧',
      list_dir: '📁',
      search_files: '🔍',
    };

    const summaryParts = [];
    if (args.cmd) summaryParts.push(args.cmd);
    else if (args.path) summaryParts.push(args.path);
    else if (args.pattern) summaryParts.push(`"${args.pattern}"`);

    const card = el('div', { class: 'tool-card open' });

    const header = el('div', { class: 'tool-header' },
      el('div', { class: 'tool-icon' }, toolNames[name] || '🔧'),
      el('div', { class: 'tool-name' }, name),
      el('div', { class: 'tool-summary' }, summaryParts.join(' ') || ''),
      el('div', { class: 'tool-toggle' }, '▶')
    );
    header.addEventListener('click', () => card.classList.toggle('open'));
    card.appendChild(header);

    const body = el('div', { class: 'tool-body' });
    const argSection = el('div', { class: 'tool-section' },
      el('div', { class: 'tool-section-label' }, 'ورودی'),
      el('pre', {}, argsStr)
    );
    body.appendChild(argSection);

    if (toolResult !== null && toolResult !== undefined) {
      const resultSection = el('div', { class: 'tool-section tool-result' + (toolResult.error ? ' error' : '') },
        el('div', { class: 'tool-section-label' }, toolResult.error ? '⚠ خطا' : '✓ خروجی'),
        el('pre', {}, toolResult.output || '(خالی)')
      );
      body.appendChild(resultSection);
    } else if (isStreaming) {
      const pendingSection = el('div', { class: 'tool-section' },
        el('div', { class: 'tool-section-label muted' }, '⏳ در حال اجرا…'),
      );
      body.appendChild(pendingSection);
    }

    card.appendChild(body);
    return card;
  }

  // ===== Streaming =====
  async function sendMessage() {
    if (state.streaming) return;
    const input = $('#input');
    const text = input.value.trim();
    if (!text && state.pendingFiles.length === 0) return;

    if (!state.currentChatId) {
      await newChat();
    }

    // Upload any pending files first
    if (state.pendingFiles.length > 0) {
      try {
        setStatus('streaming', 'در حال آپلود فایل…');
        await uploadPendingFiles();
      } catch (e) {
        return; // stop, error already shown
      }
    }

    input.value = '';
    autoResize();

    // If we have uploaded files but no text, add a placeholder message
    let content = text;
    if (!content && state.pendingFiles.some(f => f.status === 'uploaded')) {
      // Show a placeholder listing the files
      const names = state.pendingFiles.filter(f => f.status === 'uploaded').map(f => f.name);
      content = names.length === 1
        ? `فایل ${names[0]} رو ببین و توضیح بده.`
        : `این فایل‌ها رو ببین:\n${names.map(n => `- ${n}`).join('\n')}`;
    }

    // Optimistic: add user message to view
    const userMsg = { role: 'user', content: content, created_at: Date.now() / 1000 };
    state.messages.push(userMsg);
    renderMessages();

    // Prepare assistant placeholder
    const assistantMsg = {
      role: 'assistant',
      content: '',
      tool_calls: [],
      _streaming: true,
      _element: null,
      _cardsByCall: {},
    };
    state.messages.push(assistantMsg);

    // Add the assistant element to the DOM (empty, will fill in)
    const msgEl = createStreamingMessageEl(assistantMsg);
    $('#messages').appendChild(msgEl);
    assistantMsg._element = msgEl;
    scrollToBottom();

    // Send to server
    state.abortController = new AbortController();
    state.streaming = true;
    setStreamingUI(true);
    setStatus('streaming', 'در حال ارسال…');

    let bodyText = '';
    let currentCard = null;

    try {
      const resp = await fetch(`/api/chats/${state.currentChatId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: text,
          mode: state.mode,
          model: state.model || null,
          workspace: state.workspace || null,
        }),
        signal: state.abortController.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(errText || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events
        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const eventText = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (!eventText.trim()) continue;

          for (const line of eventText.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6);
            if (!dataStr.trim()) continue;
            let data;
            try {
              data = JSON.parse(dataStr);
            } catch {
              continue;
            }
            handleStreamEvent(data, assistantMsg);
          }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        setStatus('error', 'توقف شد');
      } else {
        setStatus('error', e.message);
        appendErrorToAssistant(assistantMsg, e.message);
      }
    } finally {
      state.streaming = false;
      state.abortController = null;
      setStreamingUI(false);
      // Reload chat list to update titles/order
      await loadChats();
    }
  }

  function handleStreamEvent(data, msg) {
    switch (data.type) {
      case 'user_msg':
        // already handled
        break;

      case 'text': {
        msg.content += data.content;
        updateStreamingBubble(msg);
        scrollToBottom();
        break;
      }

      case 'reasoning': {
        if (!msg._reasoning) {
          msg._reasoning = '';
          const r = el('div', { class: 'reasoning-block' });
          r.appendChild(el('span', { class: 'reasoning-icon' }, '💭'));
          r.appendChild(el('div', { class: 'reasoning-content' }));
          msg._reasoningEl = r;
          msg._bodyEl.querySelector('.bubble').appendChild(r);
        }
        msg._reasoning += data.content;
        msg._reasoningEl.querySelector('.reasoning-content').textContent = msg._reasoning;
        scrollToBottom();
        break;
      }

      case 'step': {
        setStatus('thinking', `گام ${data.step}/${data.max}…`);
        break;
      }

      case 'tool_call': {
        // Add a tool card to the message
        const tc = { name: data.name, arguments: data.arguments || {} };
        msg.tool_calls = msg.tool_calls || [];
        msg.tool_calls.push(tc);
        const card = renderToolCard(tc, null, true);
        msg._bodyEl.querySelector('.bubble').appendChild(card);
        msg._currentCard = card;
        msg._currentCall = tc;
        scrollToBottom();
        break;
      }

      case 'tool_result': {
        if (msg._currentCard) {
          // Replace the streaming card with a complete one
          const tc = msg._currentCall;
          const newCard = renderToolCard(tc, {
            output: data.output,
            error: data.error,
          }, false);
          msg._currentCard.replaceWith(newCard);
          msg._currentCard = newCard;
        }
        scrollToBottom();
        break;
      }

      case 'done': {
        setStatus('streaming', `پایان (${data.elapsed}s, ${data.tokens || 0} tokens)`);
        if (data.message_id) {
          msg.id = data.message_id;
        }
        // Reload this chat to get the saved messages
        setTimeout(() => openChat(state.currentChatId), 100);
        break;
      }

      case 'error': {
        setStatus('error', data.message || 'خطا');
        appendErrorToAssistant(msg, data.message);
        break;
      }
    }
  }

  function createStreamingMessageEl(msg) {
    const wrapper = el('div', { class: 'message assistant' });
    const avatar = el('div', { class: 'avatar bot-avatar' }, 'A');
    const body = el('div', { class: 'message-body' });
    body.appendChild(el('div', { class: 'message-role' }, 'Argo'));
    const bubble = el('div', { class: 'bubble' });
    const typing = el('div', { class: 'typing-dots' },
      el('span'), el('span'), el('span')
    );
    bubble.appendChild(typing);
    body.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    msg._bodyEl = body;
    msg._bubbleEl = bubble;
    msg._typingEl = typing;
    return wrapper;
  }

  function updateStreamingBubble(msg) {
    if (msg._typingEl) {
      msg._typingEl.remove();
      msg._typingEl = null;
    }
    if (!msg._textEl) {
      msg._textEl = el('div', { class: 'message-text' });
      msg._bubbleEl.appendChild(msg._textEl);
    }
    msg._textEl.innerHTML = renderMarkdown(msg.content);
  }

  function appendErrorToAssistant(msg, message) {
    if (!msg._textEl) {
      msg._textEl = el('div', { class: 'message-text' });
      if (msg._bubbleEl) msg._bubbleEl.appendChild(msg._textEl);
    }
    if (msg.content) msg._textEl.innerHTML = renderMarkdown(msg.content);
    const err = el('div', { class: 'error-text', style: { marginTop: '8px', fontSize: '13px' } },
      `⚠ ${message || 'خطا'}`);
    msg._bubbleEl.appendChild(err);
  }

  function stopStreaming() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  // ===== UI helpers =====
  function setStatus(level, text) {
    const el = $('#status');
    el.className = 'status ' + level;
    el.textContent = text;
  }

  function setStreamingUI(streaming) {
    $('#sendBtn').classList.toggle('hidden', streaming);
    $('#stopBtn').classList.toggle('hidden', !streaming);
    $('#input').disabled = false; // Keep input enabled so user can prepare next msg
  }

  function scrollToBottom() {
    const m = $('#messages');
    requestAnimationFrame(() => {
      m.scrollTop = m.scrollHeight;
    });
  }

  function autoResize() {
    const i = $('#input');
    i.style.height = 'auto';
    i.style.height = Math.min(i.scrollHeight, 240) + 'px';
  }

  // ===== File upload =====
  function formatSize(bytes) {
    if (bytes < 1024) return bytes + 'B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB';
    return (bytes / 1024 / 1024).toFixed(1) + 'MB';
  }

  function renderAttachments() {
    const wrap = $('#attachments');
    wrap.innerHTML = '';
    if (state.pendingFiles.length === 0) {
      wrap.classList.add('hidden');
      return;
    }
    wrap.classList.remove('hidden');
    for (const f of state.pendingFiles) {
      const chip = el('div', { class: 'attachment-chip' + (f.status ? ' ' + f.status : '') },
        el('span', { class: 'chip-name' }, f.name),
        el('span', { class: 'chip-size' }, formatSize(f.size)),
        el('span', { class: 'chip-remove', title: 'حذف' },
          (() => {
            const s = el('span');
            s.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
            return s;
          })()
        )
      );
      chip.querySelector('.chip-remove').addEventListener('click', () => {
        state.pendingFiles = state.pendingFiles.filter(x => x !== f);
        renderAttachments();
      });
      wrap.appendChild(chip);
    }
  }

  async function uploadPendingFiles() {
    if (!state.currentChatId || state.pendingFiles.length === 0) return;
    const fd = new FormData();
    for (const f of state.pendingFiles) {
      fd.append('files', f.file);
    }
    try {
      const r = await fetch(`/api/chats/${state.currentChatId}/files`, {
        method: 'POST',
        body: fd,
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || `HTTP ${r.status}`);
      }
      const data = await r.json();
      // Mark uploaded
      for (const f of state.pendingFiles) f.status = 'uploaded';
      renderAttachments();
      // Clear after 2s
      setTimeout(() => {
        state.pendingFiles = [];
        renderAttachments();
      }, 1500);
      return data;
    } catch (e) {
      setStatus('error', 'آپلود ناموفق: ' + e.message);
      throw e;
    }
  }

  function addFiles(filesList) {
    for (const file of filesList) {
      // Avoid duplicates
      if (state.pendingFiles.some(f => f.name === file.name && f.size === file.size)) continue;
      state.pendingFiles.push({
        name: file.name,
        size: file.size,
        file: file,
        status: '',
      });
    }
    renderAttachments();
  }

  function updateModeUI() {
    $$('.mode-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === state.mode);
    });
    $('#agentBanner').classList.toggle('hidden', state.mode !== 'agent');
  }

  // ===== Settings =====
  async function loadSettings() {
    try {
      const s = await api('GET', '/api/settings');
      Object.assign(state.settings, s);
      state.model = s.model;
      $('#modelBadge').textContent = s.model;
      fillSettingsForm();
    } catch (e) {
      console.error('Failed to load settings', e);
    }
  }

  function fillSettingsForm() {
    $('#setApiKey').value = '';
    $('#setApiKey').placeholder = state.settings.has_api_key ? '•••••••• (ست شده، برای تغییر پر کن)' : 'sk-or-...';
    $('#setBaseUrl').value = state.settings.base_url || '';
    $('#setModel').value = state.settings.model || '';
    $('#setTemp').value = state.settings.temperature ?? 0.2;
    $('#setMaxTokens').value = state.settings.max_tokens ?? 8192;
    $('#setMaxSteps').value = state.settings.max_steps ?? 40;
    $('#setAutoApprove').checked = !!state.settings.auto_approve;
    $('#setWorkspace').value = state.settings.default_workspace || '';
  }

  async function saveSettings() {
    const body = {
      api_key: $('#setApiKey').value || undefined,
      base_url: $('#setBaseUrl').value || undefined,
      model: $('#setModel').value || undefined,
      temperature: parseFloat($('#setTemp').value) || undefined,
      max_tokens: parseInt($('#setMaxTokens').value) || undefined,
      max_steps: parseInt($('#setMaxSteps').value) || undefined,
      auto_approve: $('#setAutoApprove').checked,
    };
    try {
      await api('POST', '/api/settings', body);
      await loadSettings();
      closeSettingsModal();
      setStatus('streaming', 'تنظیمات ذخیره شد');
      setTimeout(() => setStatus('streaming', 'آماده'), 2000);
    } catch (e) {
      setStatus('error', e.message);
    }
  }

  function openSettingsModal() {
    loadSettings();
    $('#settingsModal').classList.remove('hidden');
  }

  function closeSettingsModal() {
    $('#settingsModal').classList.add('hidden');
  }

  // ===== Event wiring =====
  function bindEvents() {
    // New chat
    $('#newChatBtn').addEventListener('click', newChat);

    // Sidebar toggle
    $('#toggleSidebarBtn').addEventListener('click', () => {
      $('#sidebar').classList.add('collapsed');
    });
    $('#openSidebarBtn').addEventListener('click', () => {
      $('#sidebar').classList.remove('collapsed');
    });

    // Mode toggle
    $$('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        state.mode = btn.dataset.mode;
        updateModeUI();
      });
    });

    // Input
    const input = $('#input');
    input.addEventListener('input', autoResize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Send/Stop
    $('#sendBtn').addEventListener('click', sendMessage);
    $('#stopBtn').addEventListener('click', stopStreaming);

    // File attach
    $('#attachBtn').addEventListener('click', () => $('#fileInput').click());
    $('#fileInput').addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        addFiles(e.target.files);
        e.target.value = ''; // reset so same file can be added again
      }
    });

    // Drag & drop on the input area
    const dropZone = $('#dropZone');
    ['dragenter', 'dragover'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (ev === 'dragleave' && e.target !== dropZone && dropZone.contains(e.relatedTarget)) return;
        dropZone.classList.remove('dragover');
      });
    });
    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        addFiles(files);
      }
    });
    // Prevent the page from navigating when files are dropped outside the drop zone
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());

    // Settings
    $('#settingsBtn').addEventListener('click', openSettingsModal);
    $$('[data-close-modal]').forEach(b => b.addEventListener('click', closeSettingsModal));
    $('#saveSettingsBtn').addEventListener('click', saveSettings);
    $('#reloadSettingsBtn').addEventListener('click', loadSettings);

    // Empty state tips (delegated)
    document.addEventListener('click', (e) => {
      const tip = e.target.closest('.tip');
      if (tip) {
        $('#input').value = tip.dataset.prompt;
        $('#input').focus();
        autoResize();
      }
    });

    // Escape to close modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeSettingsModal();
      }
    });
  }

  // ===== Boot =====
  async function boot() {
    bindEvents();
    autoResize();
    await loadSettings();
    await loadChats();
    updateModeUI();
    setStatus('streaming', 'آماده');
    setTimeout(() => setStatus('streaming', ''), 2000);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
