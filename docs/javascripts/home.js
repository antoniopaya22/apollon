/* Apollon landing page animations (GSAP + ScrollTrigger). */
(function () {
  "use strict";
  if (!window.gsap || !document.getElementById("lp")) return;
  const gsap = window.gsap;
  if (window.ScrollTrigger) gsap.registerPlugin(window.ScrollTrigger);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  // ------------------------------------------------------------ Beta sampling
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
  const beta = (a, b) => { const x = gamma(a), y = gamma(b); return x / (x + y); };
  window.apollonBeta = beta;

  // ------------------------------------------------------------------- hero
  // An attacker sends the same disguised attack (crafted against logistic regression)
  // again and again. Each attempt is answered by the model Thompson sampling draws.
  (function hero() {
    const svg = document.getElementById("hs-svg");
    if (!svg) return;
    const M = [
      { name: "Random Forest", a: 8, b: 2 }, { name: "Decision Tree", a: 7, b: 2 },
      { name: "Naive Bayes", a: 6, b: 2.5 }, { name: "Logistic Regression", a: 6, b: 2.5, fooled: true },
      { name: "Multi-Layer Perceptron", a: 7, b: 2 },
    ];
    const mg = document.getElementById("hs-models");
    const rowY = (i) => 76 + i * 34;
    const pills = M.map((m, i) => {
      const g = el("g", { class: "hs-model" + (m.fooled ? " is-weak" : "") }, mg);
      el("rect", { x: 162, y: rowY(i) - 13, width: 260, height: 26, rx: 13 }, g);
      el("text", { x: 178, y: rowY(i) + 5 }, g).textContent = m.name;
      if (m.fooled) el("text", { x: 410, y: rowY(i) + 5, "text-anchor": "end", class: "hs-weak" }, g).textContent = "fooled by x′";
      return g;
    });
    const packet = document.getElementById("hs-packet");
    const wireIn = document.getElementById("hs-wire-in");
    const wireOut = document.getElementById("hs-wire-out");
    const verdict = document.getElementById("hs-verdict");
    const icon = document.getElementById("hs-icon");
    const vtext = document.getElementById("hs-vtext");
    const cellsA = document.getElementById("hs-a"), cellsS = document.getElementById("hs-s");
    const nA = document.getElementById("hs-a-n"), nS = document.getElementById("hs-s-n");
    const N = 10;
    let n = 0, det = 0;

    const reset = () => {
      n = 0; det = 0;
      [cellsA, cellsS].forEach((c) => { c.innerHTML = ""; for (let i = 0; i < N; i++) c.appendChild(document.createElement("i")); });
      nA.textContent = "0 / 0 detected"; nS.textContent = "0 / 0 detected";
    };
    const draw = (path, d) => {
      path.setAttribute("d", d);
      const len = path.getTotalLength();
      gsap.set(path, { strokeDasharray: len, strokeDashoffset: len, opacity: 1 });
      return gsap.to(path, { strokeDashoffset: 0, duration: 0.35, ease: "power2.out" });
    };

    function attempt() {
      if (n === N) { reset(); }
      const th = M.map((m) => beta(m.a, m.b));
      const k = th.indexOf(Math.max(...th));
      const caught = !M[k].fooled;
      const y = rowY(k);
      const tl = gsap.timeline({ onComplete: () => gsap.delayedCall(n === N ? 2.6 : 0.5, attempt) });
      tl.set(packet, { x: 52, y: 130, opacity: 0, scale: 0.6 })
        .set([wireIn, wireOut], { opacity: 0 })
        .call(() => { pills.forEach((p) => p.classList.remove("is-on")); verdict.setAttribute("class", "hs-verdict"); icon.textContent = "?"; vtext.textContent = ""; })
        .to(packet, { opacity: 1, scale: 1, duration: 0.25 })
        .to(packet, { x: 120, duration: 0.45, ease: "power2.in" })
        .call(() => pills[k].classList.add("is-on"))
        .add(draw(wireIn, `M144,130 C160,130 150,${y} 162,${y}`))
        .to(packet, { x: 292, y, duration: 0.5, ease: "power2.inOut" }, "<")
        .to({}, { duration: 0.35 })
        .add(draw(wireOut, `M422,${y} C470,${y} 460,130 494,130`))
        .to(packet, { x: 530, y: 130, duration: 0.45, ease: "power2.in" }, "<")
        .to(packet, { opacity: 0, scale: 1.8, duration: 0.25 })
        .call(() => {
          n += 1; if (caught) det += 1;
          verdict.setAttribute("class", "hs-verdict " + (caught ? "is-caught" : "is-missed"));
          icon.textContent = caught ? "✓" : "✗";
          vtext.textContent = caught ? "detected" : "missed";
          cellsA.children[n - 1].className = caught ? "is-caught" : "is-missed";
          cellsS.children[n - 1].className = "is-missed";
          nA.textContent = `${det} / ${n} detected`;
          nS.textContent = `0 / ${n} detected`;
        }, null, "<")
        .fromTo(verdict, { scale: 0.85, svgOrigin: "530 130" }, { scale: 1, svgOrigin: "530 130", duration: 0.35, ease: "back.out(3)" }, "<");
    }

    reset();
    if (reduce) { pills[0].classList.add("is-on"); gsap.set(packet, { opacity: 0 }); return; }
    attempt();
  })();

  // ----------------------------------------------- the problem: binary search
  // A decision-based attacker binary-searches the perturbation δ (0 = original attack,
  // 1 = benign-looking) that flips the verdict, then replays the evasion ten times.
  // Each query is one row: the remaining interval and the probe in its middle.
  (function problem() {
    const box = document.getElementById("ps");
    if (!box) return;
    const X0 = 130, X1 = 960, TOP = 34, ROW = 22, xs = (d) => X0 + d * (X1 - X0);
    const MODELS = [
      { name: "RF", d: 0.42, a: 8, b: 2 }, { name: "DT", d: 0.58, a: 7, b: 2 },
      { name: "NB", d: 0.30, a: 6, b: 2.5 }, { name: "LR", d: 0.17, a: 6, b: 2.5 },
      { name: "MLP", d: 0.66, a: 7, b: 2 },
    ];
    const PROBES = 7, REPLAYS = 10, BOTTOM = TOP + (PROBES + 1) * ROW;

    function setup(row) {
      const apollon = row.dataset.mode === "apollon";
      const svg = row.querySelector(".ps-track");
      svg.setAttribute("viewBox", `0 0 1000 ${BOTTOM + 30}`);
      svg.innerHTML = "";
      el("line", { x1: X0, x2: X1, y1: TOP, y2: TOP, class: "ps-axis" }, svg);
      el("text", { x: X0, y: TOP - 12, class: "ps-end" }, svg).textContent = "δ = 0 · original attack";
      el("text", { x: X1, y: TOP - 12, class: "ps-end", "text-anchor": "end" }, svg).textContent = "δ = 1 · looks benign";
      const models = apollon ? MODELS : [MODELS[0]];
      const ticks = models.map((m) => {
        const t = el("g", { class: "ps-bound" + (apollon ? "" : " is-on") }, svg);
        el("line", { x1: xs(m.d), x2: xs(m.d), y1: TOP + 6, y2: BOTTOM }, t);
        el("text", { x: xs(m.d), y: BOTTOM + 18, "text-anchor": "middle" }, t).textContent = apollon ? m.name : "RF boundary";
        return t;
      });
      const rows = el("g", {}, svg);
      return { apollon, ticks, rows, models,
        phase: row.querySelector(".ps-phase"), cells: row.querySelector(".ps-cells"), result: row.querySelector(".ps-result") };
    }

    const pick = (st) => {
      if (!st.apollon) return 0;
      const th = st.models.map((m) => beta(m.a, m.b));
      return th.indexOf(Math.max(...th));
    };

    function run(row) {
      const st = setup(row);
      st.cells.innerHTML = "";
      for (let i = 0; i < REPLAYS; i++) st.cells.appendChild(document.createElement("i"));
      st.result.textContent = "–";
      st.result.className = "ps-result";
      let lo = 0, hi = 1;
      const tl = gsap.timeline();
      tl.call(() => { st.phase.textContent = "1 · binary search for the boundary"; });
      for (let q = 1; q <= PROBES; q++) {
        tl.call(() => {
          const y = TOP + q * ROW;
          const mid = (lo + hi) / 2;
          const k = pick(st);
          const m = st.models[k];
          const benign = mid > m.d;
          st.ticks.forEach((t, i) => t.classList.toggle("is-on", i === k));
          const g = el("g", { class: "ps-probe " + (benign ? "is-ben" : "is-atk") }, st.rows);
          const bar = el("rect", { class: "ps-int", x: xs(lo), y: y - 5, width: xs(hi) - xs(lo), height: 10, rx: 5 }, g);
          el("circle", { cx: xs(mid), cy: y, r: 7 }, g);
          el("text", { x: X0 - 14, y: y + 4, "text-anchor": "end" }, g).textContent = st.apollon ? `query ${q} · ${m.name}` : `query ${q}`;
          gsap.from(bar, { attr: { width: 0 }, duration: 0.35, ease: "power2.out" });
          gsap.from(g, { opacity: 0, duration: 0.25 });
          if (benign) hi = mid; else lo = mid;
        });
        tl.to({}, { duration: 0.7 });
      }
      tl.call(() => {
        st.ticks.forEach((t) => t.classList.toggle("is-on", !st.apollon));
        st.phase.textContent = `2 · evasion found at δ = ${hi.toFixed(2)} · 3 · replaying it`;
      });
      let evaded = 0;
      for (let r = 0; r < REPLAYS; r++) {
        tl.to({}, { duration: 0.18 });
        tl.call(() => {
          const m = st.models[pick(st)];
          const ok = hi > m.d;
          if (ok) evaded += 1;
          const c = st.cells.children[r];
          c.className = ok ? "is-ben" : "is-atk";
          c.title = `${m.name}: ${ok ? "evaded" : "detected"}`;
        });
      }
      tl.call(() => {
        st.result.textContent = `${evaded}/${REPLAYS} evade`;
        st.result.className = "ps-result " + (evaded >= 8 ? "is-bad" : "is-good");
        st.phase.textContent = st.apollon
          ? "it learned a mix of boundaries: the evasion only works on some answers"
          : "it learned the boundary every request meets: the evasion always works";
      });
      tl.to({}, { duration: 3.4 });
      return tl;
    }

    const rows = Array.from(box.querySelectorAll(".ps-row"));
    function loop() { const tls = rows.map(run); tls[0].eventCallback("onComplete", loop); }
    if (reduce) { rows.forEach((r) => setup(r)); return; }
    if (window.ScrollTrigger) {
      window.ScrollTrigger.create({ trigger: box, start: "top 80%", once: true, onEnter: loop });
    } else loop();
  })();

  // --------------------------------------------------------- scroll reveals
  if (!reduce && window.ScrollTrigger) {
    const ST = window.ScrollTrigger;
    gsap.from(".lp-hero__grid > div:first-child > *", { opacity: 0, y: 22, duration: 0.8, stagger: 0.08, ease: "power3.out" });
    gsap.from(".lp-hero__viz", { opacity: 0, y: 30, duration: 1, ease: "power3.out", delay: 0.25 });
    gsap.from(".lp-strip > div", { opacity: 0, y: 14, duration: 0.6, stagger: 0.08, delay: 0.6 });
    const groups = [".lp-split", ".lp-cols > div", ".ps", ".lp-ts", ".lp-code > *",
      ".lp-features > a", ".lp-table__row", ".lp-faq .lp-h2", ".lp-faq__list details"];
    groups.forEach((sel) => {
      gsap.set(sel, { opacity: 0, y: 26 });
      ST.batch(sel, { start: "top 90%", once: true,
        onEnter: (els) => gsap.to(els, { opacity: 1, y: 0, duration: 0.7, stagger: 0.08, ease: "power3.out", overwrite: true }) });
    });
    gsap.from(".lp-watermark", { yPercent: 40, opacity: 0, duration: 1.2, ease: "power3.out",
      scrollTrigger: { trigger: ".lp-cta", start: "top 70%", once: true } });
  }

  // ------------------------------------------------------ nav and side dots
  const nav = document.getElementById("lp-nav");
  const onScroll = () => nav && nav.classList.toggle("is-solid", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const dots = document.querySelectorAll(".lp-dots a");
  if ("IntersectionObserver" in window && dots.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        dots.forEach((d) => d.classList.toggle("is-on", d.dataset.dot === e.target.id));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll(".lp-sec[id]").forEach((sec) => io.observe(sec));
  }

  // --------------------------------------------------------------- terminal
  (function terminal() {
    const pre = document.getElementById("ap-term");
    if (!pre) return;
    const cmd = "apollon demo";
    const out = [
      "synthetic: 20 features, 3500 train (20.2% attacks), 1500 test (20.2% attacks)",
      "",
      "Apollon on synthetic traffic",
      "----------------------------",
      "accuracy             0.9587",
      "detection_rate       0.8086",
      "f1                   0.8877",
      "false_positive_rate  0.0033",
      "roc_auc              0.9026",
      "",
      "Arm selection: RF 79%, DT 1%, NB 0%, LR 19%, MLP 0%",
    ];
    const prompt = '<span class="p">$</span> ';
    if (reduce || !window.ScrollTrigger) { pre.innerHTML = prompt + cmd + "\n" + out.join("\n"); return; }
    pre.innerHTML = prompt + '<span class="cur">▋</span>';
    const o = { n: 0 };
    const tl = gsap.timeline({ paused: true });
    tl.to(o, { n: cmd.length, duration: 0.9, ease: "none", onUpdate: () => {
      pre.innerHTML = prompt + cmd.slice(0, Math.round(o.n)) + '<span class="cur">▋</span>';
    } });
    out.forEach((line, i) => tl.call(() => {
      pre.innerHTML = prompt + cmd + "\n" + out.slice(0, i + 1).join("\n") + '\n<span class="cur">▋</span>';
    }, null, i === 0 ? "+=0.6" : "+=0.12"));
    window.ScrollTrigger.create({ trigger: pre, start: "top 85%", once: true, onEnter: () => tl.play() });
  })();

  // ------------------------------------------------------------- copy button
  const copy = document.querySelector(".ap-install__copy");
  if (copy) copy.addEventListener("click", () => {
    const text = document.getElementById("ap-install-cmd").textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(
      () => { copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy"; }, 1500); },
      () => { copy.textContent = "Select"; });
  });
})();
