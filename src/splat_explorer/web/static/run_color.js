// Shared per-run palette for the results grid and details page.
(() => {
  const palette = {
    '': ['None', '#89949b', '#171d21'],
    red: ['Red', '#dd7770', '#342327'],
    orange: ['Orange', '#e3a063', '#342a20'],
    yellow: ['Yellow', '#d7bf69', '#302d21'],
    green: ['Green', '#85b883', '#232e27'],
    teal: ['Teal', '#6bbab0', '#1f2e2f'],
    blue: ['Blue', '#7aa9de', '#212b3b'],
    purple: ['Purple', '#ad91d3', '#2c263a'],
    pink: ['Pink', '#d591b5', '#342635'],
  };
  const state = window.runColorState = new Map();
  const channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('splatfix-run-colors') : null;
  function apply(data) {
    if (!data || !(data.color in palette) || typeof data.run_id !== 'string') return;
    state.set(data.run_id, data);
    document.querySelectorAll('run-color').forEach(picker => {
      if (picker.getAttribute('run-id') === data.run_id) picker.setAttribute('color', data.color);
    });
    document.getElementById('colorFilter')?.dispatchEvent(new Event('change'));
  }
  if (channel) channel.onmessage = event => apply(event.data);
  class RunColor extends HTMLElement {
    static observedAttributes = ['color', 'run-id'];
    constructor() {
      super();
      this.attachShadow({mode: 'open'});
      this.closeOnOutsideClick = event => {
        if (!event.composedPath().includes(this)) {
          const popup = this.shadowRoot.querySelector('details');
          if (popup) popup.open = false;
        }
      };
    }
    connectedCallback() {
      this.render();
      if (this.hasAttribute('filter')) document.addEventListener('click', this.closeOnOutsideClick, true);
    }
    disconnectedCallback() { document.removeEventListener('click', this.closeOnOutsideClick, true); }
    attributeChangedCallback() { if (this.isConnected) this.render(); }
    render() {
      const filter = this.hasAttribute('filter');
      const options = filter ? {'*': ['All colors', 'conic-gradient(#dd7770, #d7bf69, #85b883, #7aa9de, #ad91d3, #dd7770)'], ...palette} : palette;
      const run = this.getAttribute('run-id');
      const requested = filter ? (this.getAttribute('color') ?? '*') : (state.get(run)?.color ?? this.getAttribute('color'));
      const color = requested in options ? requested : '';
      const [label, swatch, background] = options[color];
      const title = filter ? 'Filter by color' : 'Run color';
      const tile = this.closest('article');
      if (tile) tile.style.backgroundColor = background;
      if (this.busy) return;
      this.shadowRoot.innerHTML = `<style>
        :host{display:inline-block;position:relative;flex-shrink:0;font:12px/1.4 system-ui;color:#e4e9ec}
        summary{list-style:none;cursor:pointer;border:0;border-radius:5px;padding:6px;display:flex;align-items:center;gap:4px}
        summary::-webkit-details-marker{display:none}summary:focus-visible,button:focus-visible{outline:2px solid #b4d8ea;outline-offset:2px}
        i{display:inline-block;width:14px;height:14px;border-radius:3px;background:var(--swatch);border:1px solid #ffffff30}
        .menu{position:absolute;right:0;top:34px;z-index:20;width:168px;padding:6px;background:#20272c;border:1px solid #53616b;border-radius:6px;box-shadow:0 5px 18px #0008;display:grid;grid-template-columns:1fr 1fr;gap:3px}
        :host([filter]) summary i{width:20px;height:20px}
        :host([filter]) .menu,:host(#runColor) .menu{left:0;right:auto}
        button{font:inherit;color:inherit;background:transparent;border:1px solid transparent;border-radius:4px;padding:6px;display:flex;align-items:center;gap:6px;cursor:pointer;text-align:left}
        button:hover,button[aria-pressed=true]{background:#344149;border-color:#71808b}button:disabled{opacity:.5;cursor:wait}
        .status{display:block;max-width:180px;color:#e3b693}.status:empty{display:none}
      </style><details><summary title="${title}: ${label}" aria-label="${title}: ${label}"><i style="--swatch:${swatch}"></i></summary><div class="menu" aria-label="${title}">${Object.entries(options).map(([key, value]) => `<button type="button" data-color="${key}" aria-pressed="${key === color}"><i style="--swatch:${value[1]}"></i>${value[0]}</button>`).join('')}</div></details><span class="status" role="status"></span>`;
      this.shadowRoot.onkeydown = event => {
        if (event.key === 'Escape') { this.shadowRoot.querySelector('details').open = false; this.shadowRoot.querySelector('summary').focus(); }
      };
      this.shadowRoot.querySelectorAll('button').forEach(button => button.onclick = async () => {
        if (filter) {
          this.setAttribute('color', button.dataset.color);
          this.dispatchEvent(new Event('change', {bubbles: true}));
          this.shadowRoot.querySelector('summary').focus();
          return;
        }
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
