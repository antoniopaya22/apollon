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
  // One adversarial flow (crafted against LR) queried again and again. Every query is
  // answered by a model drawn with Thompson sampling; its decision boundary is shown.
  (function hero() {
    const plot = document.getElementById("hv-plot");
    if (!plot) return;
    const M = [
      { name: "RF", a: 8, b: 2, attack: true,
        edge: "M0,200 L80,200 L80,170 L150,170 L150,120 L230,120 L230,100 L330,100 L330,70 L420,70 L420,60 L560,60",
        region: "M0,300 L0,200 L80,200 L80,170 L150,170 L150,120 L230,120 L230,100 L330,100 L330,70 L420,70 L420,60 L560,60 L560,300 Z" },
      { name: "DT", a: 7, b: 2, attack: true,
        edge: "M0,180 L180,180 L180,90 L380,90 L380,30 L560,30",
        region: "M0,300 L0,180 L180,180 L180,90 L380,90 L380,30 L560,30 L560,300 Z" },
      { name: "NB", a: 6, b: 2.5, attack: true,
        edge: "M160,170 a170,110 0 1,0 340,0 a170,110 0 1,0 -340,0",
        region: "M160,170 a170,110 0 1,0 340,0 a170,110 0 1,0 -340,0 Z" },
      { name: "LR", a: 6, b: 2.5, attack: false,
        edge: "M0,40 L560,250",
        region: "M0,40 L560,250 L560,300 L0,300 Z" },
      { name: "MLP", a: 7, b: 2, attack: true,
        edge: "M0,210 C120,60 220,220 330,90 C420,0 480,120 560,80",
        region: "M0,300 L0,210 C120,60 220,220 330,90 C420,0 480,120 560,80 L560,300 Z" },
    ];
    const FULL = { RF: "Random Forest", DT: "Decision Tree", NB: "Naive Bayes", LR: "Logistic Regression", MLP: "Multi-Layer Perceptron" };
    const chipsBox = document.getElementById("hv-chips");
    const chips = M.map((m) => {
      const c = document.createElement("div");
      c.className = "hv-chip";
      c.innerHTML = `<span>${m.name}</span><i><b></b></i>`;
      chipsBox.appendChild(c);
      return c;
    });
    const region = document.getElementById("hv-region");
    const edge = document.getElementById("hv-edge");
    const ring = document.getElementById("hv-ring");
    const label = document.getElementById("hv-model");
    const log = document.getElementById("hv-log");
    const score = document.getElementById("hv-score");
    const wins = M.map(() => 0);
    let q = 0, detected = 0;

    function show(k, animate) {
      const m = M[k];
      chips.forEach((c, i) => c.classList.toggle("is-on", i === k));
      label.textContent = `boundary of ${FULL[m.name]}`;
      region.setAttribute("d", m.region);
      edge.setAttribute("d", m.edge);
      ring.setAttribute("class", "hv-ring " + (m.attack ? "is-attack" : "is-benign"));
      if (!animate) return;
      const len = edge.getTotalLength();
      gsap.fromTo(region, { opacity: 0 }, { opacity: 1, duration: 0.35 });
      gsap.fromTo(edge, { strokeDasharray: len, strokeDashoffset: len }, { strokeDashoffset: 0, duration: 0.6, ease: "power2.out" });
      gsap.fromTo(ring, { attr: { r: 10 }, opacity: 1 }, { attr: { r: 30 }, opacity: 0, duration: 0.9, ease: "power2.out", delay: 0.35 });
    }

    function step() {
      q += 1;
      const th = M.map((m) => beta(m.a, m.b));
      const k = th.indexOf(Math.max(...th));
      const m = M[k];
      wins[k] += 1;
      if (m.attack) detected += 1;
      show(k, true);
      const total = wins.reduce((s, w) => s + w, 0);
      chips.forEach((c, i) => { c.querySelector("b").style.width = `${(100 * wins[i]) / total}%`; });
      const li = document.createElement("li");
      li.innerHTML = `<span>query ${q}</span><span>${m.name}</span><span class="${m.attack ? "atk" : "ben"}">${m.attack ? "attack ✓" : "benign ✗"}</span>`;
      log.prepend(li);
      while (log.children.length > 3) log.lastChild.remove();
      gsap.from(li, { opacity: 0, y: -8, duration: 0.3 });
      score.textContent = `${detected} / ${q}`;
    }

    if (reduce) {
      show(0, false);
      score.textContent = "—";
      return;
    }
    step();
    let timer = setInterval(step, 1700);
    document.addEventListener("visibilitychange", () => {
      clearInterval(timer);
      if (!document.hidden) timer = setInterval(step, 1700);
    });
  })();

  // ------------------------------------------------------------- the duel
  (function duel() {
    const ARMS = ["RF", "DT", "NB", "LR", "MLP"];
    const yMid = (s) => (s[0] + 3 * s[1] + 3 * s[2] + s[3]) / 8; // boundary height at x = 150
    const pathB = (s) => `M0,${s[0]} C100,${s[1]} 200,${s[2]} 300,${s[3]}`;
    const pathZ = (s) => `M0,0 L300,0 L300,${s[3]} C200,${s[2]} 100,${s[1]} 0,${s[0]} Z`;
    const FIXED = [120, 60, 120, 95];
    const rand = (a, b) => a + Math.random() * (b - a);

    function setup(id, moving) {
      const svg = document.getElementById(id);
      if (!svg) return null;
      const boundary = svg.querySelector(".ap-d-boundary");
      const zone = svg.querySelector(".ap-d-zone-attack");
      const probes = svg.querySelector(".ap-d-probes");
      const attacker = svg.querySelector(".ap-d-attacker");
      const armText = svg.querySelector(".ap-d-arm");
      const status = document.querySelector(`.ap-d-status[data-for="${moving ? "apollon" : "single"}"]`);
      let shape = FIXED.slice();

      function run() {
        probes.innerHTML = "";
        shape = FIXED.slice();
        gsap.set(boundary, { attr: { d: pathB(shape) } });
        gsap.set(zone, { attr: { d: pathZ(shape) } });
        gsap.set(attacker, { attr: { cy: 34, class: "ap-d-attacker" } });
        status.textContent = "probing…";
        status.className = "ap-d-status";
        let y = 34, detected = 0, queries = 0;
        const tl = gsap.timeline({ onComplete: () => gsap.delayedCall(1.6, run) });
        for (let q = 0; q < 10; q++) {
          tl.add(() => {
            if (moving) {
              shape = [rand(80, 150), rand(40, 170), rand(40, 170), rand(70, 150)];
              while (Math.abs(yMid(shape) - y) < 6) shape[1] += 12;
              gsap.to(boundary, { attr: { d: pathB(shape) }, duration: 0.35, ease: "power2.inOut" });
              gsap.to(zone, { attr: { d: pathZ(shape) }, duration: 0.35, ease: "power2.inOut" });
              if (armText) armText.textContent = ARMS[Math.floor(Math.random() * ARMS.length)];
            }
          });
          tl.to({}, { duration: moving ? 0.38 : 0.12 });
          tl.add(() => {
            queries += 1;
            const isAttack = y < yMid(shape);
            if (isAttack) detected += 1;
            el("circle", { cx: 150 + rand(-26, 26), cy: y + rand(-8, 8), r: 3.2,
              class: "ap-d-probe " + (isAttack ? "is-attack" : "is-benign") }, probes);
            const evaded = !moving && !isAttack;
            if (evaded) {
              status.textContent = `evaded after ${queries} queries`;
              status.className = "ap-d-status is-bad";
              attacker.setAttribute("class", "ap-d-attacker is-evaded");
              tl.progress(1);
              return;
            }
            y = isAttack ? y + 13 : Math.max(30, y - 9);
            gsap.to(attacker, { attr: { cy: y }, duration: 0.3, ease: "power1.out" });
            status.textContent = `detected ${detected} of ${queries} queries`;
            if (moving) status.className = "ap-d-status is-good";
          });
          tl.to({}, { duration: 0.32 });
        }
        return tl;
      }
      return run;
    }

    const runs = [setup("ap-duel-single", false), setup("ap-duel-apollon", true)].filter(Boolean);
    if (!runs.length || reduce) return;
    if (window.ScrollTrigger) {
      window.ScrollTrigger.create({ trigger: ".lp-duel", start: "top 80%", once: true, onEnter: () => runs.forEach((r) => r()) });
    } else runs.forEach((r) => r());
  })();

  // --------------------------------------------------------- scroll reveals
  if (!reduce && window.ScrollTrigger) {
    const ST = window.ScrollTrigger;
    gsap.from(".lp-hero__grid > div:first-child > *", { opacity: 0, y: 22, duration: 0.8, stagger: 0.08, ease: "power3.out" });
    gsap.from(".lp-hero__viz", { opacity: 0, y: 30, duration: 1, ease: "power3.out", delay: 0.25 });
    gsap.from(".lp-strip > div", { opacity: 0, y: 14, duration: 0.6, stagger: 0.08, delay: 0.6 });
    const groups = [".lp-split", ".lp-cols > div", ".lp-duel > *", ".lp-ts", ".lp-code > *",
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
