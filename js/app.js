/* meutipo — clone client-only do mytype.fyi.
   Tudo roda no navegador: nenhum dado sai da maquina. */
"use strict";

const LS_KEY = "meutipo.v1";
const EMBED_DIM = 48;
const CROWD_SIZE = 12;

const $ = (s) => document.querySelector(s);
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

/* ---------------- estado ---------------- */
const S = {
  manifest: [],          // todos os rostos (ordem = linhas do embed.bin)
  emb: null,             // Float32Array N*48 normalizado por linha
  embById: new Map(),
  pool: "f",
  poolFaces: [],
  shown: new Map(),      // id -> {liked:bool, crowd:int}
  taps: 0,
  crowds: 0,
  grid: [],              // ids visiveis agora
  candidateQueue: [],
};

/* ---------------- persistencia ---------------- */
function save() {
  const shown = {};
  for (const [id, v] of S.shown) shown[id] = v.liked ? 1 : 0;
  localStorage.setItem(LS_KEY, JSON.stringify({
    pool: S.pool, shown, taps: S.taps, crowds: S.crowds,
  }));
}
function load() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)); } catch { return null; }
}

/* ---------------- carregamento de dados ---------------- */
async function loadData() {
  const [faces, embBuf] = await Promise.all([
    fetch("faces.json").then((r) => r.json()),
    fetch("embed.bin").then((r) => r.arrayBuffer()).catch(() => null),
  ]);
  S.manifest = faces;
  if (embBuf) {
    const dv = new DataView(embBuf);
    const n = dv.getUint32(0, true), dim = dv.getUint32(4, true);
    const raw = new Int8Array(embBuf, 8, n * dim);
    const emb = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      let norm = 0;
      for (let j = 0; j < dim; j++) { const v = raw[i * dim + j] / 127; emb[i * dim + j] = v; norm += v * v; }
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < dim; j++) emb[i * dim + j] /= norm;
    }
    S.emb = emb; S.embDim = dim;
    faces.forEach((f, i) => S.embById.set(f.id, i));
  }
  return faces;
}

function setPool(pool) {
  S.pool = pool;
  S.poolFaces = S.manifest.filter((f) =>
    pool === "all" ? true : (pool === "f" ? f.g === 0 : f.g === 1));
}

/* ---------------- modelo ---------------- */
function traitStats() {
  const st = {};
  for (const f of S.poolFaces) {
    const s = S.shown.get(f.id);
    if (!s) continue;
    for (const t of f.traits) {
      (st[t] ??= { shown: 0, liked: 0 }).shown++;
      if (s.liked) st[t].liked++;
    }
  }
  return st;
}

function embCentroid(filterFn) {
  if (!S.emb) return null;
  const c = new Float32Array(S.embDim); let n = 0;
  for (const [id, s] of S.shown) {
    if (!filterFn(s)) continue;
    const i = S.embById.get(id); if (i == null) continue;
    for (let j = 0; j < S.embDim; j++) c[j] += S.emb[i * S.embDim + j];
    n++;
  }
  if (!n) return null;
  let norm = 0;
  for (let j = 0; j < S.embDim; j++) { c[j] /= n; norm += c[j] * c[j]; }
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < S.embDim; j++) c[j] /= norm;
  return c;
}

const cos = (i, c) => {
  if (!c) return 0.5;
  let d = 0;
  for (let j = 0; j < S.embDim; j++) d += S.emb[i * S.embDim + j] * c[j];
  return d; // ja normalizado -> -1..1
};

function predict(f, ctx) {
  const { stats, cPos, cNeg, base } = ctx;
  let tr = 0;
  for (const t of f.traits) {
    const s = stats[t];
    tr += s ? (s.liked + 0.5) / (s.shown + 1) : base;
  }
  tr /= f.traits.length;
  let sim = 0.5;
  const i = S.embById.get(f.id);
  if (i != null && (cPos || cNeg)) {
    sim = 0.5 + 0.5 * Math.tanh(4 * (cos(i, cPos) - (cNeg ? cos(i, cNeg) : 0)));
  }
  const personal = 0.55 * tr + 0.45 * sim;
  const conf = S.taps / (S.taps + 15);
  return (1 - conf) * f.pop + conf * (0.78 * personal + 0.22 * f.pop);
}

function ctx() {
  const shownArr = [...S.shown.values()];
  return {
    stats: traitStats(),
    cPos: embCentroid((s) => s.liked),
    cNeg: embCentroid((s) => !s.liked),
    base: Math.min(0.6, Math.max(0.04, S.taps / Math.max(shownArr.length, 1))),
  };
}

/* ---------------- multidao ---------------- */
function unseenFaces() {
  return S.poolFaces.filter((f) => !S.shown.has(f.id));
}

function buildCrowd() {
  const c = ctx();
  const un = unseenFaces();
  const scored = un.map((f) => ({ f, p: predict(f, c) }));
  scored.sort((a, b) => b.p - a.p);

  const pick = [];
  const used = new Set();
  // exploracao: topo previsto com temperatura
  const top = scored.slice(0, Math.min(60, scored.length));
  for (const cand of shuffle([...top])) {
    if (pick.length >= 8) break;
    used.add(cand.f.id); pick.push(cand.f);
  }
  // rostos populares nao vistos (calibracao vs "multidao")
  const pops = shuffle(un.filter((f) => f.pop > 0.7 && !used.has(f.id)));
  for (const f of pops.slice(0, 2)) { used.add(f.id); pick.push(f); }
  // exploracao aleatoria (testa os limites)
  const rest = shuffle(un.filter((f) => !used.has(f.id)));
  for (const f of rest) {
    if (pick.length >= CROWD_SIZE) break;
    used.add(f.id); pick.push(f);
  }
  S.grid = shuffle(pick).map((f) => f.id);
  S.crowds++;
  renderCrowd();
  save();
}

function nextCandidate() {
  // proxima face para repor um slot: explora o topo previsto entre nao vistos
  const un = unseenFaces();
  if (!un.length) return null;
  const c = ctx();
  const scored = un.map((f) => ({ f, p: predict(f, c) }));
  scored.sort((a, b) => b.p - a.p);
  const band = scored.slice(0, Math.min(30, scored.length));
  return Math.random() < 0.75 ? band[Math.floor(Math.random() * band.length)].f
                              : un[Math.floor(Math.random() * un.length)];
}

/* ---------------- render ---------------- */
const faceById = (id) => S.manifest.find((f) => f.id === id);

function faceEl(f, mini = false, tag = "") {
  const el = document.createElement("div");
  el.className = "face new";
  el.dataset.id = f.id;
  el.innerHTML = `<img loading="lazy" src="faces/${f.f}" alt="rosto gerado por IA">${tag ? `<span class="tag">${tag}</span>` : ""}`;
  if (!mini) el.addEventListener("click", () => onTap(el, f));
  return el;
}

function renderCrowd() {
  const grid = $("#grid");
  grid.innerHTML = "";
  for (const id of S.grid) grid.appendChild(faceEl(faceById(id)));
  $("#crowd-count").textContent = `Multidão ${S.crowds}` + (S.crowds < 20 ? ` · meta ~20` : "");
  updateCounts();
}

function updateCounts() {
  $("#tap-count").textContent = `${S.taps} ${S.taps === 1 ? "toque" : "toques"}`;
  const acc = estimateAcc();
  const chip = $("#acc-chip");
  if (acc != null && S.taps >= 15) {
    chip.classList.remove("hidden");
    chip.textContent = `precisão ~${Math.round(acc * 100)}%`;
  } else chip.classList.add("hidden");
}

function onTap(el, f) {
  if (S.shown.get(f.id)?.liked) return;
  const already = S.shown.has(f.id);
  S.shown.set(f.id, { liked: true, crowd: S.crowds });
  S.taps++;
  el.classList.add("liked");
  el.insertAdjacentHTML("beforeend", '<span class="heart">♥</span>');
  // marca os outros visiveis como "vistos e nao curtidos" so ao avancar;
  // aqui apenas repoe a face tocada depois de mostrar o coracao
  setTimeout(() => {
    const next = nextCandidate();
    if (!next) return;
    S.shown.set(next.id, S.shown.get(next.id) ?? { liked: false, crowd: S.crowds });
    const idx = S.grid.indexOf(f.id);
    if (idx >= 0) S.grid[idx] = next.id;
    el.replaceWith(faceEl(next));
  }, 380);
  updateCounts();
  save();
}

function markGridSeen() {
  for (const id of S.grid) {
    if (!S.shown.has(id)) S.shown.set(id, { liked: false, crowd: S.crowds });
  }
}

function nextCrowd() {
  markGridSeen();
  buildCrowd();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------------- precisao ---------------- */
function estimateAcc() {
  const shownIds = [...S.shown.keys()].map((id) => faceById(id)).filter(Boolean);
  const liked = shownIds.filter((f) => S.shown.get(f.id).liked);
  const skipped = shownIds.filter((f) => !S.shown.get(f.id).liked);
  if (liked.length < 3 || skipped.length < 3) return null;
  const c = ctx();
  const pl = liked.map((f) => predict(f, c));
  const ps = skipped.map((f) => predict(f, c));
  // AUC aproximado: P(p_liked > p_skip)
  let wins = 0, n = 0;
  for (const a of pl) for (const b of ps) { wins += a > b ? 1 : 0.5 * (a === b); n++; }
  return n ? wins / n : null;
}

/* ---------------- resultados ---------------- */
function traitChips() {
  const stats = traitStats();
  const shownN = S.shown.size || 1;
  const base = Math.max(S.taps / shownN, 0.02);
  const rows = Object.entries(stats)
    .filter(([, s]) => s.shown >= 4)
    .map(([t, s]) => ({ t, lift: (s.liked / s.shown) / base, liked: s.liked, shown: s.shown }));
  const likes = rows.filter((r) => r.lift >= 1.15 && r.liked >= 2).sort((a, b) => b.lift - a.lift).slice(0, 8);
  const dislikes = rows.filter((r) => r.lift <= 0.75 && r.shown >= 6).sort((a, b) => a.lift - b.lift).slice(0, 6);
  return { likes, dislikes };
}

function gridInto(sel, faces, tagFn) {
  const g = $(sel);
  g.innerHTML = "";
  if (!faces.length) { g.innerHTML = '<p class="sec-expl">nada ainda — avalie mais multidões.</p>'; return; }
  for (const f of faces.slice(0, 12)) g.appendChild(faceEl(f, true, tagFn ? tagFn(f) : ""));
}

function renderResults() {
  markGridSeen();
  const c = ctx();
  const liked = [...S.shown.entries()].filter(([, s]) => s.liked)
    .map(([id]) => faceById(id)).filter(Boolean);
  const skipped = [...S.shown.entries()].filter(([, s]) => !s.liked)
    .map(([id]) => faceById(id)).filter(Boolean);

  $("#res-summary").textContent =
    `${S.taps} toques em ${S.crowds} ${S.crowds === 1 ? "multidão" : "multidões"} · ${liked.length} rostos curtidos de ${S.shown.size} vistos`;

  const acc = estimateAcc();
  if (acc != null) {
    $("#res-acc-fill").style.width = `${Math.round(acc * 100)}%`;
    $("#res-acc-text").textContent =
      `Precisão estimada do modelo: ~${Math.round(acc * 100)}% ao separar o que você curte do que você passa. Quanto mais você avalia, mais afiado fica.`;
  } else {
    $("#res-acc-fill").style.width = "8%";
    $("#res-acc-text").textContent = "Toque em mais rostos para o modelo calibrar a precisão.";
  }

  const { likes, dislikes } = traitChips();
  $("#res-traits").innerHTML = likes.length
    ? likes.map((r) => `<span class="chip"><b>${r.t}</b> ×${r.lift.toFixed(1)}</span>`).join("")
    : '<p class="sec-expl">Toque em mais rostos para revelar seus traços.</p>';
  $("#res-anti-traits").innerHTML = dislikes.length
    ? dislikes.map((r) => `<span class="chip"><b>${r.t}</b> ×${r.lift.toFixed(1)}</span>`).join("")
    : "";

  const popTag = (f) => `popularidade ${Math.round(f.pop * 100)}%`;

  // top curtidos por score pessoal
  gridInto("#res-top", liked.sort((a, b) => predict(b, c) - predict(a, c)));

  // previsoes: nao vistos com maior p
  const unseen = unseenFaces().map((f) => ({ f, p: predict(f, c) }))
    .sort((a, b) => b.p - a.p).slice(0, 12).map((x) => x.f);
  gridInto("#res-predicted", unseen, (f) => `combina ${Math.round(predict(f, c) * 100)}%`);

  gridInto("#res-pop-like", liked.filter((f) => f.pop >= 0.62).sort((a, b) => b.pop - a.pop), popTag);
  gridInto("#res-pop-skip", skipped.filter((f) => f.pop >= 0.62).sort((a, b) => b.pop - a.pop), popTag);
  gridInto("#res-gems", liked.filter((f) => f.pop <= 0.40).sort((a, b) => a.pop - b.pop), popTag);
}

function copySummary() {
  const { likes } = traitChips();
  const traits = likes.slice(0, 6).map((r) => r.t).join(", ") || "ainda indefinido";
  const acc = estimateAcc();
  const txt = `Meu tipo (meutipo): ${traits}. ${S.taps} toques em ${S.crowds} multidões` +
    (acc != null ? ` · precisão ~${Math.round(acc * 100)}%` : "");
  navigator.clipboard.writeText(txt).then(
    () => toast("Resumo copiado!"),
    () => toast(txt));
}

let toastT;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.add("hidden"), 2400);
}

/* ---------------- navegacao ---------------- */
function show(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
  $(id).classList.remove("hidden");
}

function startApp(saved) {
  if (saved) {
    setPool(saved.pool);
    for (const [id, liked] of Object.entries(saved.shown || {}))
      S.shown.set(+id, { liked: !!liked, crowd: 0 });
    S.taps = saved.taps || 0;
    S.crowds = saved.crowds || 0;
  }
  buildCrowd();
  show("#screen-crowd");
}

/* swipe up = proxima multidao */
let touchY = null;
addEventListener("touchstart", (e) => { touchY = e.touches[0].clientY; }, { passive: true });
addEventListener("touchend", (e) => {
  if (touchY == null || $("#screen-crowd").classList.contains("hidden")) return;
  const dy = touchY - e.changedTouches[0].clientY;
  if (dy > 70) nextCrowd();
  touchY = null;
}, { passive: true });

/* ---------------- boot ---------------- */
(async function boot() {
  try {
    await loadData();
  } catch (e) {
    document.body.innerHTML = '<p style="padding:40px;text-align:center">Erro carregando faces.json — rode <code>tools/fetch_faces.py</code> primeiro.</p>';
    return;
  }

  // faixa demo na landing
  const strip = $("#demo-strip");
  shuffle([...S.manifest]).slice(0, 6).forEach((f) => {
    strip.insertAdjacentHTML("beforeend", `<img loading="lazy" src="faces/${f.f}" alt="">`);
  });

  const saved = load();
  if (saved && saved.taps > 0) {
    const b = document.createElement("button");
    b.className = "btn primary pool"; b.textContent = `Continuar (${saved.taps} toques)`;
    b.addEventListener("click", () => startApp(saved));
    document.querySelector(".pool-btns").prepend(b);
  }

  document.querySelectorAll(".btn.pool[data-pool]").forEach((b) =>
    b.addEventListener("click", () => {
      const prev = load();
      if (prev && prev.taps > 0 && prev.pool !== b.dataset.pool &&
          !confirm("Trocar de grupo apaga seus toques atuais. Continuar?")) return;
      localStorage.removeItem(LS_KEY);
      Object.assign(S, { shown: new Map(), taps: 0, crowds: 0, grid: [] });
      setPool(b.dataset.pool);
      buildCrowd();
      show("#screen-crowd");
    }));

  $("#btn-next").addEventListener("click", nextCrowd);
  $("#btn-results-top").addEventListener("click", () => { renderResults(); show("#screen-results"); });
  $("#btn-continue").addEventListener("click", () => { buildCrowd(); show("#screen-crowd"); });
  $("#btn-continue2").addEventListener("click", () => { buildCrowd(); show("#screen-crowd"); });
  $("#btn-copy").addEventListener("click", copySummary);
  $("#btn-reset").addEventListener("click", () => {
    if (confirm("Apagar todos os seus toques e recomeçar?")) {
      localStorage.removeItem(LS_KEY); location.reload();
    }
  });
})();
