// Shared preference for GPU provisioning and new extended runs on this origin.
window.ArtiFixerModel = (() => {
  const key = "splat-explorer.artifixer-model";
  const valid = value => value === "14b" || value === "1.3b";
  let current = "14b";
  try { const saved = localStorage.getItem(key); if (valid(saved)) current = saved; } catch (_) {}
  const listeners = new Set();
  function publish(value) {
    current = valid(value) ? value : "14b";
    listeners.forEach(listener => listener(current));
  }
  window.addEventListener("storage", event => {
    if (event.key === key || event.key === null) publish(event.newValue);
  });
  return {
    get: () => current,
    set(value) {
      if (!valid(value)) throw new Error("Choose ArtiFixer 14B or 1.3B");
      try { localStorage.setItem(key, value); } catch (_) {}
      publish(value);
    },
    subscribe(listener) { listeners.add(listener); },
    prepared(detail) {
      const id = detail?.artifixer_runtime?.model_id;
      if (id === "Wan-AI/Wan2.1-T2V-14B-Diffusers") return "14b";
      if (id === "Wan-AI/Wan2.1-T2V-1.3B-Diffusers") return "1.3b";
      return null;
    },
  };
})();
