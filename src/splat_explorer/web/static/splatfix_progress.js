/* Browser requests are local. The queue manager owns the shared remote cadence. */
(() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const large = options => ['14b', '14b-teacher'].includes(options?.inference_model) || options?.teacher_inference === true;
  function html(progress, enabled=false) {
    if (!enabled && !progress?.stages) return '';
    const stages = progress?.stages || {
      inference: {label:'14B inference',status:'pending',detail:'Waiting for a progress snapshot'},
      reconstruction: {label:'3D reconstruction',status:'pending',detail:'Waiting for inference'}
    };
    const bars = ['inference','reconstruction','plus'].filter(key => stages[key] && (key !== 'plus' || stages[key].status !== 'pending'));
    const updated = progress?.observed_at ? new Date(progress.observed_at * 1000).toLocaleTimeString() : null;
    return `<div class="splatfix-progress">${bars.map(key => {
      const s=stages[key], known=Number.isFinite(s.percent), percent=known?Math.max(0,Math.min(100,s.percent)):0;
      const value=known || s.status==='pending' ? ` value="${percent}"` : '';
      const frames=Number.isFinite(s.expected_frames)?` · ${s.completed_frames||0} / ${s.expected_frames} frames saved`:'';
      return `<div class="splatfix-progress-stage"><div class="splatfix-progress-label"><strong>${escape(s.label)}</strong><span>${known?percent.toFixed(1)+'%':s.status==='pending'?'Pending':'In progress'}</span></div><progress max="100"${value} aria-label="${escape(s.label)}" aria-valuetext="${escape(s.detail)}"></progress><div class="splatfix-progress-note">${escape(s.detail)}${escape(frames)}${['error','stopped'].includes(s.status)?' · '+escape(s.status):''}</div></div>`;
    }).join('')}<div class="splatfix-progress-note">Remote updates at most once per minute while visible${updated?' · Last checked '+escape(updated):''}</div></div>`;
  }
  let sending=false;
  async function heartbeat() {
    if (document.hidden || sending) return;
    sending=true;
    try { await fetch('/api/splatfix/activity', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); }
    catch (_) { /* Local service may be restarting. Keep the last progress. */ }
    finally { sending=false; }
  }
  window.SplatfixProgress={html,large,heartbeat};
  const style=document.createElement('style');
  style.textContent='.splatfix-progress{display:grid;gap:14px;margin:16px 0;max-width:900px}.splatfix-progress-label{display:flex;justify-content:space-between;gap:12px;font-size:13px}.splatfix-progress progress{display:block;width:100%;height:10px;margin:7px 0;accent-color:#c6e69b}.splatfix-progress-note{font-size:12px;color:var(--muted,#a1afa5)}';
  document.head.append(style);
  document.addEventListener('visibilitychange', heartbeat);
  heartbeat();
  setInterval(heartbeat, 20000);
})();
