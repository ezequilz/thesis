// Shared per-run palette for the results grid and details page.
(() => {
  const palette = {
    '': ['None', '#89949b', '#171d21'],
    red: ['Red', '#dd7770', '#302226'],
    orange: ['Orange', '#e3a063', '#302820'],
    yellow: ['Yellow', '#d7bf69', '#2d2b21'],
    green: ['Green', '#85b883', '#212c26'],
    teal: ['Teal', '#6bbab0', '#1e2c2d'],
    blue: ['Blue', '#7aa9de', '#202938'],
    purple: ['Purple', '#ad91d3', '#292537'],
    pink: ['Pink', '#d591b5', '#302532'],
  };
  const state = window.runColorState = new Map();
  const channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('splatfix-run-colors') : null;
  function apply(data) {
    if (!data || !(data.color in palette) || typeof data.run_id !== 'string') return;
    state.set(data.run_id, data);
    document.querySelectorAll('run-color').forEach(picker => {
      if (picker.getAttribute('run-id') === data.run_id) picker.setAttribute('color', data.color);
    });
  }
  if (channel) channel.onmessage = event => apply(event.data);
  class RunColor extends HTMLElement {
    static observedAttributes = ['color', 'run-id'];
    constructor() {
      super();
      this.attachShadow({mode: 'open'});
    }
    connectedCallback() { this.render(); }
    attributeChangedCallback() { if (this.isConnected) this.render(); }
    render() {
      const run = this.getAttribute('run-id');
      const requested = state.get(run)?.color ?? this.getAttribute('color');
      const color = requested in palette ? requested : '';
      const [label, swatch, background] = palette[color];
      const tile = this.closest('article');
      if (tile) tile.style.backgroundColor = background;
      if (this.busy) return;
      this.shadowRoot.innerHTML = `<style>
        :host{display:inline-block;position:relative;flex-shrink:0;font:12px/1.4 system-ui;color:#e4e9ec}
        summary{list-style:none;cursor:pointer;border:1px solid #52616a;border-radius:5px;padding:5px;display:flex;align-items:center;gap:4px}
        summary::-webkit-details-marker{display:none}summary:focus-visible,button:focus-visible{outline:2px solid #b4d8ea;outline-offset:2px}
        i{display:inline-block;width:14px;height:14px;border-radius:3px;background:var(--swatch);border:1px solid #ffffff30}
        .menu{position:absolute;right:0;top:34px;z-index:20;width:168px;padding:6px;background:#20272c;border:1px solid #53616b;border-radius:6px;box-shadow:0 5px 18px #0008;display:grid;grid-template-columns:1fr 1fr;gap:3px}
        :host(#runColor) .menu{left:0;right:auto}
        button{font:inherit;color:inherit;background:transparent;border:1px solid transparent;border-radius:4px;padding:6px;display:flex;align-items:center;gap:6px;cursor:pointer;text-align:left}
        button:hover,button[aria-pressed=true]{background:#344149;border-color:#71808b}button:disabled{opacity:.5;cursor:wait}
        .status{display:block;max-width:180px;color:#e3b693}.status:empty{display:none}
      </style><details><summary title="Run color: ${label}" aria-label="Run color: ${label}"><i style="--swatch:${swatch}"></i><span aria-hidden="true">⌄</span></summary><div class="menu" aria-label="Choose run color">${Object.entries(palette).map(([key, value]) => `<button type="button" data-color="${key}" aria-pressed="${key === color}"><i style="--swatch:${value[1]}"></i>${value[0]}</button>`).join('')}</div></details><span class="status" role="status"></span>`;
      this.shadowRoot.onkeydown = event => {
        if (event.key === 'Escape') { this.shadowRoot.querySelector('details').open = false; this.shadowRoot.querySelector('summary').focus(); }
      };
      this.shadowRoot.querySelectorAll('button').forEach(button => button.onclick = async () => {
        if (!run || this.busy) return;
        this.busy = true;
        this.shadowRoot.querySelectorAll('button').forEach(item => item.disabled = true);
        this.shadowRoot.querySelector('.status').textContent = 'Saving…';
        const selected = button.dataset.color;
        try {
          const response = await fetch('/api/splatfix/results/color', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({run_id: run, color: selected})});
          const data = await response.json();
          if (!response.ok || !data.ok) throw Error(data.error || 'Could not save color');
          this.busy = false;
          apply(data);
          channel?.postMessage(data);
          this.render();
          this.shadowRoot.querySelector('summary').focus();
        } catch (error) {
          this.busy = false;
          this.shadowRoot.querySelectorAll('button').forEach(item => item.disabled = false);
          this.shadowRoot.querySelector('.status').textContent = error.message;
        }
      });
    }
  }
  customElements.define('run-color', RunColor);
})();
