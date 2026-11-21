// Shared top navigation for every Watchtower page. Include with:
//   <link rel="stylesheet" href="/_ui/static/ui.css"><script src="/_ui/static/nav.js" defer></script>
(function () {
  const links = [
    ["/", "Dashboard"],
    ["/_ui/tenders/view", "Tenders"],
    ["/_ui/crossref/view", "Several sources"],
    ["/_ui/heal", "Heal theater"],
    ["/_ui/verifier", "Verifier"],
  ];
  const here = location.pathname;
  const active = href => href === "/" ? here === "/" : here.startsWith(href.replace(/\/view$/, "")) ||
    (href === "/_ui/verifier" && here.startsWith("/_ui/audit/"));
  // Brand mark: a tower with a lit lamp room, one solid colour.
  const mark = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 18 8.2 9h3.6L13 18M5 18h10M6.5 9h7M7.5 9V5.5L10 3l2.5 2.5V9"/><path d="M1.5 4.5 4 5.5M18.5 4.5 16 5.5"/></svg>`;
  const nav = document.createElement("nav");
  nav.className = "wt-nav";
  nav.setAttribute("aria-label", "Main");
  nav.innerHTML = `<a class="wt-brand" href="/">${mark}Watchtower</a><div class="wt-links">` +
    links.map(([href, label]) =>
      `<a href="${href}"${active(href) ? ' aria-current="page"' : ""}>${label}</a>`).join("") + `</div>`;
  document.body.prepend(nav);

  // Tooltips for long text: any element with data-tip (body) and optional data-tip-title (plain first line).
  // Shown on hover and on keyboard focus (give the element tabindex="0"), kept inside the viewport, Esc closes.
  const tip = document.createElement("div");
  tip.className = "wt-tip";
  tip.id = "wt-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.append(tip);
  let owner = null;
  const hide = () => {
    if (!owner) return;
    owner.removeAttribute("aria-describedby");
    owner = null;
    tip.hidden = true;
  };
  const show = el => {
    if (el === owner) return;
    hide();
    const body = el.dataset.tip, title = el.dataset.tipTitle;
    if (!body && !title) return;
    tip.textContent = "";
    if (title) { const b = document.createElement("strong"); b.textContent = title; tip.append(b); }
    if (body) { const p = document.createElement("span"); p.textContent = body; tip.append(p); }
    tip.hidden = false;
    owner = el;
    el.setAttribute("aria-describedby", "wt-tip");
    const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect(), gap = 8, edge = 8;
    const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    const left = Math.max(edge, Math.min(r.left + r.width / 2 - t.width / 2, vw - t.width - edge));
    const below = r.bottom + gap + t.height <= vh - edge || r.top - gap - t.height < edge;
    tip.style.left = left + "px";
    tip.style.top = Math.max(edge, below ? r.bottom + gap : r.top - gap - t.height) + "px";
  };
  const target = e => e.target instanceof Element ? e.target.closest("[data-tip], [data-tip-title]") : null;
  document.addEventListener("mouseover", e => { const el = target(e); if (el) show(el); else if (!tip.contains(e.target)) hide(); });
  document.addEventListener("focusin", e => { const el = target(e); if (el) show(el); else hide(); });
  document.addEventListener("focusout", hide);
  document.addEventListener("keydown", e => { if (e.key === "Escape") hide(); });
  // scrolling moves the element away from the tip; a focused element keeps its tip (focusing scrolls it into view)
  addEventListener("scroll", () => { if (owner && document.activeElement !== owner) hide(); }, {passive: true, capture: true});
})();
