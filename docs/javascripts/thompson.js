/* Interactive Thompson sampler: Beta posteriors of four arms, draws, feedback. */
(function () {
  "use strict";
  const root = document.getElementById("ap-ts");
  if (!root) return;
  const gsap = window.gsap;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  // --- math
  function lgamma(z) { // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1; let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  const lbeta = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);
  const pdf = (x, a, b) => Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function gamma(k) { // Marsaglia and Tsang, k >= 1
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = gauss(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  const sample = (a, b) => { const x = gamma(a), y = gamma(b); return x / (x + y); };

  // --- state
  const ARMS = [
    { name: "RF", color: "#6366f1", truth: 0.93, a0: 9, b0: 2 },
    { name: "NB", color: "#f59e0b", truth: 0.78, a0: 7, b0: 3 },
    { name: "LR", color: "#10b981", truth: 0.88, a0: 6, b0: 2 },
    { name: "MLP", color: "#ec4899", truth: 0.90, a0: 4, b0: 2 },
  ];
  let wins, last;
  const reset = () => { ARMS.forEach((a) => { a.a = a.a0; a.b = a.b0; }); wins = ARMS.map(() => 0); last = -1; };
  reset();

  // --- svg
  const W = 640, H = 304, L = 40, R = 610, T = 18, B = 206, N = 140;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Beta posteriors of four classifiers" }, root.querySelector(".ap-ts__plot"));
  el("line", { x1: L, x2: R, y1: B, y2: B, class: "axis" }, svg);
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const x = L + t * (R - L);
    el("line", { x1: x, x2: x, y1: B, y2: B + 4, class: "axis" }, svg);
    el("text", { x, y: B + 17, "text-anchor": "middle" }, svg).textContent = t.toFixed(2);
  });
  el("text", { x: R, y: B + 34, "text-anchor": "end" }, svg).textContent = "probability of answering correctly →";
  const xs = Array.from({ length: N }, (_, i) => 0.002 + (0.996 * i) / (N - 1));
  const sx = (x) => L + x * (R - L);
  const shapes = ARMS.map((a) => ({
    area: el("path", { class: "area", fill: a.color }, svg),
    curve: el("path", { class: "curve", stroke: a.color }, svg),
    draw: el("line", { class: "draw", stroke: a.color, y1: T, y2: B, opacity: 0 }, svg),
    dot: el("circle", { r: 6, fill: a.color, stroke: "#fff", "stroke-width": 2, opacity: 0 }, svg),
  }));
  // legend + mix
  const legend = el("g", {}, svg);
  ARMS.forEach((a, i) => {
    const x = L + i * 120;
    el("rect", { x, y: 250, width: 12, height: 12, rx: 3, fill: a.color }, legend);
    a.label = el("text", { x: x + 18, y: 260 }, legend);
  });
  const mix = el("g", { class: "mix" }, svg);
  const mixBars = ARMS.map((a) => el("rect", { y: 276, height: 10, fill: a.color, x: L, width: 0 }, mix));
  el("text", { x: L, y: 298 }, svg).textContent = "share of requests answered";

  function paths() {
    const ys = ARMS.map((a) => xs.map((x) => pdf(x, a.a, a.b)));
    const max = Math.max(...ys.flat());
    return ys.map((v) => {
      const pts = v.map((y, i) => `${sx(xs[i]).toFixed(1)},${(B - (y / max) * (B - T)).toFixed(1)}`);
      return { curve: "M" + pts.join(" L"), area: `M${sx(xs[0]).toFixed(1)},${B} L` + pts.join(" L") + ` L${sx(xs[N - 1]).toFixed(1)},${B} Z` };
    });
  }
  function render(animate) {
    const p = paths();
    shapes.forEach((s, i) => {
      if (gsap && animate && !reduce) {
        gsap.to(s.curve, { attr: { d: p[i].curve }, duration: 0.5, ease: "power2.out" });
        gsap.to(s.area, { attr: { d: p[i].area }, duration: 0.5, ease: "power2.out" });
      } else { s.curve.setAttribute("d", p[i].curve); s.area.setAttribute("d", p[i].area); }
    });
    ARMS.forEach((a) => { a.label.textContent = `${a.name}  Beta(${a.a}, ${a.b})`; });
    const total = wins.reduce((s, w) => s + w, 0);
    let x = L;
    mixBars.forEach((r, i) => {
      const w = total ? (wins[i] / total) * (R - L) : 0;
      if (gsap && animate && !reduce) gsap.to(r, { attr: { x, width: w }, duration: 0.4 });
      else { r.setAttribute("x", x); r.setAttribute("width", w); }
      x += w;
    });
  }

  const msg = root.querySelector(".ap-ts__msg");
  function drawOnce(show) {
    const th = ARMS.map((a) => sample(a.a, a.b));
    const k = th.indexOf(Math.max(...th));
    wins[k] += 1; last = k;
    if (show) {
      shapes.forEach((s, i) => {
        const x = sx(th[i]);
        s.draw.setAttribute("x1", x); s.draw.setAttribute("x2", x);
        s.dot.setAttribute("cx", x); s.dot.setAttribute("cy", T + 6);
        const op = i === k ? 1 : 0.35;
        if (gsap && !reduce) {
          gsap.fromTo(s.draw, { opacity: 0 }, { opacity: op, duration: 0.3, delay: i * 0.08 });
          gsap.fromTo(s.dot, { opacity: 0, attr: { cy: B } }, { opacity: op, attr: { cy: T + 6 }, duration: 0.45, delay: i * 0.08, ease: "back.out(2)" });
        } else { s.draw.setAttribute("opacity", op); s.dot.setAttribute("opacity", op); }
      });
      msg.textContent = `Draws: ${ARMS.map((a, i) => `${a.name} ${th[i].toFixed(2)}`).join(", ")}. ${ARMS[k].name} answers.`;
    }
    return k;
  }
  function feedback(correct) {
    if (last < 0) { msg.textContent = "Draw a request first."; return; }
    const a = ARMS[last];
    if (correct) a.a += 1; else a.b += 1;
    msg.textContent = `${a.name} was ${correct ? "right" : "wrong"}: its posterior becomes Beta(${a.a}, ${a.b}).`;
    last = -1;
    render(true);
  }

  root.querySelector('[data-act="draw"]').addEventListener("click", () => { drawOnce(true); render(true); });
  root.querySelector('[data-act="ok"]').addEventListener("click", () => feedback(true));
  root.querySelector('[data-act="ko"]').addEventListener("click", () => feedback(false));
  root.querySelector('[data-act="reset"]').addEventListener("click", () => {
    reset(); shapes.forEach((s) => { s.draw.setAttribute("opacity", 0); s.dot.setAttribute("opacity", 0); });
    msg.textContent = "Posteriors reset."; render(true);
  });
  root.querySelector('[data-act="run"]').addEventListener("click", (ev) => {
    const btn = ev.currentTarget; btn.disabled = true;
    let n = 0;
    const step = () => {
      for (let j = 0; j < 10; j++) {
        const k = drawOnce(false);
        if (Math.random() < ARMS[k].truth) ARMS[k].a += 1; else ARMS[k].b += 1;
      }
      n += 10; render(true);
      msg.textContent = `${n} requests with analyst feedback (true accuracies: ${ARMS.map((a) => `${a.name} ${a.truth}`).join(", ")}).`;
      if (n < 200) setTimeout(step, reduce ? 0 : 120); else btn.disabled = false;
    };
    step();
  });
  render(false);
})();
