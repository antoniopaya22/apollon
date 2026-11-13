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
  (function hero() {
    const svg = document.getElementById("ap-bandit");
    if (!svg) return;
    const ARMS = ["RF", "DT", "NB", "LR", "MLP"];
    // Posterior (alpha, beta) per cluster: different arms lead on different traffic.
    const POST = [
      [[46, 3], [40, 6], [30, 10], [41, 5], [38, 6]],
      [[36, 6], [41, 5], [26, 12], [44, 3], [40, 5]],
    ];
    const CL = [{ x: 200, y: 130 }, { x: 200, y: 270 }];
    const X0 = 318, W = 168, rowY = (i) => 78 + i * 62;
    const g = document.getElementById("ap-arms");
    const rows = ARMS.map((name, i) => {
      const y = rowY(i);
      const grp = el("g", { class: "ap-v-arm" }, g);
      el("text", { x: 300, y: y + 5, class: "ap-v-arm-name", "text-anchor": "end" }, grp).textContent = name;
      el("rect", { x: X0, y: y - 9, width: W, height: 18, rx: 9, class: "ap-v-track" }, grp);
      const bar = el("rect", { x: X0, y: y - 9, width: 0, height: 18, rx: 9, class: "ap-v-bar" }, grp);
      const val = el("text", { x: X0 + W + 8, y: y + 5, class: "ap-v-val" }, grp);
      const wins = el("text", { x: 300, y: y + 21, class: "ap-v-wins", "text-anchor": "end" }, grp);
      return { grp, bar, val, wins, n: 0, y };
    });
    const packet = svg.querySelector("#ap-packet circle");
    const linkIn = document.getElementById("ap-link-in");
    const linkOut = document.getElementById("ap-link-out");
    const verdict = document.getElementById("ap-verdict");
    const vtext = document.getElementById("ap-verdict-text");
    const caption = document.getElementById("ap-caption");
    const clusters = [document.getElementById("ap-c0"), document.getElementById("ap-c1")];
    let round = 0;

    const draw = (path, d) => {
      path.setAttribute("d", d);
      const len = path.getTotalLength();
      gsap.set(path, { strokeDasharray: len, strokeDashoffset: len, opacity: 1 });
      return gsap.to(path, { strokeDashoffset: 0, duration: 0.45, ease: "power2.out" });
    };

    function play() {
      round += 1;
      const c = Math.random() < 0.5 ? 0 : 1;
      const theta = POST[c].map(([a, b]) => beta(a, b));
      const win = theta.indexOf(Math.max(...theta));
      const attack = Math.random() < 0.45;
      const r = rows[win];
      const tl = gsap.timeline({ onComplete: play });
      tl.set(packet, { attr: { cx: 50, cy: 200 }, opacity: 0, scale: 0.4, transformOrigin: "50% 50%" })
        .set([linkIn, linkOut], { opacity: 0 })
        .set(verdict, { attr: { class: "ap-v-verdict" } })
        .call(() => { vtext.textContent = "·"; caption.textContent = `request #${round}`; })
        .to(packet, { opacity: 1, scale: 1, duration: 0.3 })
        .to(packet, { attr: { cx: CL[c].x, cy: CL[c].y }, duration: 0.6, ease: "power2.inOut" })
        .fromTo(clusters[c], { scale: 1 }, { scale: 1.12, duration: 0.18, yoyo: true, repeat: 1, transformOrigin: "200px " + CL[c].y + "px" }, "<0.45")
        .call(() => { caption.textContent = `request #${round} → cluster c${c === 0 ? "₀" : "₁"}: every arm draws θ`; });
      rows.forEach((row, i) => {
        tl.to(row.bar, { attr: { width: W * theta[i] }, duration: 0.55, ease: "power3.out" }, i === 0 ? ">" : "<0.06");
        tl.call(() => { row.val.textContent = theta[i].toFixed(2); }, null, "<");
      });
      tl.call(() => {
        rows.forEach((row, i) => row.grp.classList.toggle("is-win", i === win));
        r.n += 1;
        rows.forEach((row) => { row.wins.textContent = row.n ? `answered ×${row.n}` : ""; });
        caption.textContent = `highest draw: ${ARMS[win]} answers request #${round}`;
      }, null, "+=0.15")
        .add(draw(linkIn, `M${CL[c].x + 34},${CL[c].y} C255,${CL[c].y} 250,${r.y} 300,${r.y}`))
        .to(packet, { attr: { cx: X0 + W * theta[win], cy: r.y }, duration: 0.55, ease: "power2.inOut" }, "<")
        .add(draw(linkOut, `M${X0 + W + 30},${r.y} C540,${r.y} 520,200 535,200`))
        .to(packet, { attr: { cx: 581, cy: 200 }, duration: 0.5, ease: "power2.inOut" }, "<")
        .call(() => {
          verdict.setAttribute("class", "ap-v-verdict " + (attack ? "is-attack" : "is-benign"));
          vtext.textContent = attack ? "attack" : "benign";
        })
        .to(packet, { opacity: 0, scale: 2.2, duration: 0.4 }, "<")
        .to({}, { duration: 1.1 })
        .call(() => rows.forEach((row) => row.grp.classList.remove("is-win")))
        .to([linkIn, linkOut], { opacity: 0, duration: 0.3 });
      return tl;
    }

    if (reduce) {
      const theta = POST[0].map(([a, b]) => a / (a + b));
      rows.forEach((row, i) => { row.bar.setAttribute("width", W * theta[i]); row.val.textContent = theta[i].toFixed(2); });
      rows[0].grp.classList.add("is-win");
      gsap.set(packet, { opacity: 0 });
      caption.textContent = "the highest draw answers each request";
      return;
    }
    play();
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
