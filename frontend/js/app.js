// app.js — router, toast, shared utilities

// ── ROUTER ────────────────────────────────────────────────────────────────
const Router = (() => {
  let _state = {};
  const listeners = [];

  function navigate(page, state = {}) {
    _state = state;
    window.location.hash = page;
    window.scrollTo({ top: 0, behavior: "instant" });
    listeners.forEach(fn => fn(page, state));
  }

  function getState() { return _state; }

  function init(defaultPage = "home") {
    window.addEventListener("hashchange", () => {
      const page = window.location.hash.slice(1) || defaultPage;
      listeners.forEach(fn => fn(page, _state));
    });
    const initial = window.location.hash.slice(1) || defaultPage;
    listeners.forEach(fn => fn(initial, _state));
  }

  function onChange(fn) { listeners.push(fn); }

  return { navigate, getState, init, onChange };
})();

// ── TOAST ─────────────────────────────────────────────────────────────────
const Toast = (() => {
  let el = null;
  let timer = null;

  function show({ type = "success", title, sub }) {
    if (!el) {
      el = document.createElement("div");
      el.className = "toast";
      document.body.appendChild(el);
    }
    el.className = `toast toast--${type}`;
    el.innerHTML = `
      <div class="toast__icon">${type === "success" ? "✓" : "✕"}</div>
      <div>
        <div class="toast__title">${title}</div>
        ${sub ? `<div class="toast__sub">${sub}</div>` : ""}
      </div>
    `;
    el.style.display = "flex";
    clearTimeout(timer);
    timer = setTimeout(hide, 3800);
  }

  function hide() {
    if (el) el.style.display = "none";
  }

  return { show, hide };
})();

// ── HELPERS ────────────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }
function qs(sel, root = document) { return root.querySelector(sel); }
function on(el, ev, fn) { if (el) el.addEventListener(ev, fn); }

function fmt(date) {
  return new Date(date).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

function fmtDT(date) {
  return new Date(date).toLocaleDateString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function riskColor(r) {
  return r === "low" ? "var(--success)" : r === "medium" ? "var(--warn)" : "#B91C1C";
}

const EVENT_LABELS = {
  rain:  { icon: "🌧", name: "Heavy Rainfall",   unit: "mm" },
  heat:  { icon: "🌡", name: "Extreme Heat",      unit: "°C" },
  aqi:   { icon: "🏭", name: "Severe AQI",        unit: "" },
  flood: { icon: "🌊", name: "Flood Alert",       unit: "" },
};

// Expose globally
window.Router = Router;
window.Toast  = Toast;
window.el     = el;
window.qs     = qs;
window.on     = on;
window.fmt    = fmt;
window.fmtDT  = fmtDT;
window.riskColor = riskColor;
window.EVENT_LABELS = EVENT_LABELS;
