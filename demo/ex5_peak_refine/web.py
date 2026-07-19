"""ex5-web — ЖИВОЙ прототип web-панели: параболическое уточнение пика в движении.

Требование Alex (2026-07-19): «прототип элементов как всё будет на web — всё должно
двигаться и меняться, а не картинки». plotly дома не установлен ⇒ страница —
САМОДОСТАТОЧНЫЙ HTML+JS (vanilla canvas, ноль зависимостей, работает офлайн и на
Debian): данные всех тактов считаются здесь (Python, реюз тракта) и встраиваются
в страницу JSON'ом; анимация/контролы — на стороне браузера.

Движение — реюз ex4: случайные модели `core.motion` (`_MODEL_BUILDERS`) + случайный
старт (`_random_initial_state`) + проекция `Kinematics` в дробные бины (kx, ky);
дальность отображается в несущую (условно: сближение → дрейф частотного бина).
Каждый такт: сцена на ДРОБНЫХ бинах → AmToCube → топ-N NMS → `refine_peak`
(реюз `example.py`). На странице видно главное: траектория argmax — «лесенка»
по целым бинам, парабола — гладкая кривая по истине.

Запуск:  .venv/Scripts/python.exe demo/ex5_peak_refine/web.py
Выход:   demo/graphics/ex5_peak_refine/web/index.html  → открыть в браузере
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex5_peak_refine/web.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.generators.waveforms import AmToCube  # noqa: E402
from core.motion import Kinematics  # noqa: E402
from demo.ex2_am_square.example import add_noise_volume  # noqa: E402
from demo.ex4_flight.example import _MODEL_BUILDERS, _random_initial_state  # noqa: E402
from demo.ex5_peak_refine.example import (  # noqa: E402
    Ex5Params,
    FracObject,
    RefineResult,
    analyze_cube,
    build_clean_volume,
    true_index,
)


@dataclass(frozen=True)
class Ex5WebParams:
    """Параметры web-прототипа (VO, «всё переменное»)."""

    nx: int = 32
    ny: int = 32
    n_axis: int = 128
    depth: int = 128
    fs: float = 500e6
    n_ticks: int = 60
    dt: float = 1.0
    snr_db_list: tuple[float, ...] = (float("inf"), 0.0)
    seed: int = 7
    # разнос целей по полю (смещение угловых бинов поверх кинематики — презентация)
    offsets: tuple[tuple[float, float], ...] = ((-9.0, 6.0), (0.5, -8.0), (9.0, 0.5))
    f_lo: float = 75e6                    # маппинг дальности в несущую: r_hi→f_lo, r_lo→f_hi
    f_hi: float = 140e6
    r_lo: float = 3000.0
    r_hi: float = 13000.0
    guard: tuple[int, int, int] = (3, 3, 6)
    cut_half: int = 5
    names: tuple[str, ...] = ("A", "B", "C")


class MovingSource:
    """Движущийся источник: случайная модель core.motion + проекция в дробные бины.

    Паттерн `FlyingEntity` (demo/ex4_flight/example.py, атрибуция там) — но третья
    координата ex5 это ЧАСТОТНЫЙ бин (CW-несущая), поэтому дальность отображается
    в несущую линейно [r_lo..r_hi] → [f_hi..f_lo] (сближение → рост частоты).
    """

    def __init__(self, name: str, rng: np.random.Generator, kin: Kinematics,
                 p: Ex5WebParams, off: tuple[float, float]) -> None:
        self.name = name
        self._model = _MODEL_BUILDERS[int(rng.integers(len(_MODEL_BUILDERS)))](rng)
        self._state = _random_initial_state(rng)
        self._kin = kin
        self._p = p
        self._off = off
        self.model_name = type(self._model).__name__

    def step(self, rng: np.random.Generator) -> FracObject:
        p = self._p
        self._state = self._model.propagate(self._state, p.dt, rng)
        s = self._kin.project(self._state, p.dt)
        kx = float(np.clip(s.kx + self._off[0], -(p.nx / 2 - 1.6), p.nx / 2 - 2.4))
        ky = float(np.clip(s.ky + self._off[1], -(p.ny / 2 - 1.6), p.ny / 2 - 2.4))
        frac = float(np.clip((s.r - p.r_lo) / (p.r_hi - p.r_lo), 0.0, 1.0))
        freq = p.f_hi - frac * (p.f_hi - p.f_lo)
        return FracObject(self.name, kx=kx, ky=ky, freq_hz=freq)


def _base_params(p: Ex5WebParams) -> Ex5Params:
    return Ex5Params(nx=p.nx, ny=p.ny, n_axis=p.n_axis, depth=p.depth, fs=p.fs,
                     seed=p.seed, scene=(), guard=p.guard, cut_half=p.cut_half)


def _cfg(p: Ex5WebParams) -> ProjectConfig:
    return ProjectConfig(array=ArrayConfig(p.nx, p.ny), modulation="am")


def build_trajectories(p: Ex5WebParams) -> list[tuple[FracObject, ...]]:
    """Такты движения: список длиной n_ticks из кортежей сцены (общие для всех SNR)."""
    rng = np.random.default_rng(p.seed)
    kin = Kinematics(_cfg(p))
    sources = [MovingSource(name, rng, kin, p, off)
               for name, off in zip(p.names, p.offsets, strict=True)]
    return [tuple(src.step(rng) for src in sources) for _ in range(p.n_ticks)]


def _cut(power: np.ndarray, r: RefineResult, ax_i: int, half: int) -> dict[str, Any]:
    """Срез мощности через пик вдоль оси ax_i: {lo, v[дБ отн. пика]} (как _panel_cut)."""
    i0 = r.peak.index[ax_i]
    n = power.shape[ax_i]
    lo, hi = max(0, i0 - half), min(n, i0 + half + 1)
    sel = list(r.peak.index)
    sel[ax_i] = slice(lo, hi)
    line = np.maximum(power[tuple(sel)], 1e-30)
    ref_db = 10.0 * np.log10(max(float(power[r.peak.index]), 1e-30))
    vals = 10.0 * np.log10(line) - ref_db
    return {"lo": int(lo), "v": [round(float(v), 1) for v in vals]}


def _tick_record(cube: Any, results: list[RefineResult], p: Ex5WebParams) -> dict[str, Any]:
    power = cube.magnitude.astype(np.float64) ** 2
    e_db = cube.angular_energy_db()
    targets = []
    for r in results:
        targets.append({
            "tr": [round(v, 3) for v in r.truth],
            "am": list(r.peak.index),
            "pr": [round(v, 3) for v in r.peak.frac_index],
            "cuts": [_cut(power, r, ax_i, p.cut_half) for ax_i in range(3)],
        })
    return {"map": [[round(float(v), 1) for v in row] for row in e_db], "tg": targets}


def simulate(p: Ex5WebParams) -> dict[str, Any]:
    """Полный расчёт данных страницы: движение → кубы по тактам → парабола, все SNR."""
    traj = build_trajectories(p)
    base = _base_params(p)
    cfg = _cfg(p)
    scanner = AmToCube(depth=p.depth, step=64)
    runs: dict[str, Any] = {}
    for snr in p.snr_db_list:
        tag = "clean" if not np.isfinite(snr) else f"snr{snr:+.0f}"
        label = "чистый (∞)" if not np.isfinite(snr) else f"SNR {snr:+.0f} дБ"
        rng = np.random.default_rng(p.seed + 1)
        ticks = []
        for scene in traj:
            pt = replace(base, scene=scene)
            volume = add_noise_volume(build_clean_volume(pt, rng), snr, rng)
            cube = scanner.fill(volume, cfg)
            ticks.append(_tick_record(cube, analyze_cube(cube, pt), p))
        runs[tag] = {"label": label, "ticks": ticks}
    return {"nx": p.nx, "ny": p.ny, "depth": p.depth, "nTicks": p.n_ticks,
            "names": list(p.names), "runs": runs}


# ── страница (самодостаточная: ноль внешних зависимостей, тёмная как ex4) ────
_PAGE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>ex5 · парабола в кубе · живой прототип</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#c9d1d9; --dim:#8b949e;
          --acc:#f0883e; --lime:#7ee787; --blue:#58a6ff; }
  body { background:var(--bg); color:var(--fg); font:13px/1.45 "Segoe UI",sans-serif; margin:0; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex;
           gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0 12px 0 0; color:var(--fg); font-weight:600; }
  button, select { background:var(--panel); color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:4px 10px; cursor:pointer; font-size:13px; }
  button.on { border-color:var(--acc); color:var(--acc); }
  input[type=range] { width:220px; vertical-align:middle; }
  .wrap { display:flex; gap:12px; padding:12px 16px; flex-wrap:wrap; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:8px 10px; }
  .card h2 { font-size:12px; margin:0 0 6px; color:var(--dim); font-weight:600; }
  canvas { display:block; }
  table { border-collapse:collapse; font-size:12px; }
  td, th { padding:2px 8px; text-align:right; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:600; }
  .acc { color:var(--acc); } .lime { color:var(--lime); } .dim { color:var(--dim); }
  #legend span { margin-right:14px; }
</style></head><body>
<header>
  <h1>ex5 · пик между бинами куба — argmax vs парабола (живой прогон)</h1>
  <button id="play" class="on">⏸ пауза</button>
  <select id="speed"><option value="250">0.5×</option><option value="125" selected>1×</option>
    <option value="60">2×</option></select>
  <input type="range" id="tick" min="0" value="0">
  <span id="tickLabel" class="dim"></span>
  <select id="run"></select>
  <select id="target"></select>
  <button id="trails" class="on">трейлы: вкл</button>
</header>
<div class="wrap">
  <div class="card"><h2>угловая карта энергии (дБ) · <span class="lime">＋истина</span> ·
      <span class="dim">×argmax (лесенка)</span> · <span class="acc">○парабола (гладко)</span></h2>
    <canvas id="map" width="520" height="520"></canvas></div>
  <div>
    <div class="card"><h2 id="zoomTitle">зум цели</h2>
      <canvas id="zoom" width="240" height="240"></canvas></div>
    <div class="card" style="margin-top:12px"><h2>срезы через пик (дБ) + парабола</h2>
      <canvas id="cutkx" width="240" height="110"></canvas>
      <canvas id="cutky" width="240" height="110"></canvas>
      <canvas id="cutrange" width="240" height="110"></canvas></div>
  </div>
  <div>
    <div class="card"><h2>угловая ошибка по тактам (бины) — среднее по целям</h2>
      <canvas id="err" width="430" height="180"></canvas></div>
    <div class="card" style="margin-top:12px"><h2>текущий такт</h2>
      <table id="stats"></table></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const AX = ["kx","ky","range"];
let runKey = Object.keys(DATA.runs)[0], tick = 0, sel = 0, playing = true, trails = true;
let timer = null;

// ── viridis (опорные точки + лерп) ──
const VIR = [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
             [39,173,129],[92,200,99],[170,220,50],[253,231,37]];
function vir(t){ t = Math.min(1, Math.max(0, t)); const x = t*(VIR.length-1), i = Math.floor(x),
  f = x-i, a = VIR[i], b = VIR[Math.min(i+1, VIR.length-1)];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")})`; }

const ticksOf = () => DATA.runs[runKey].ticks;
const errAng = (est, tr) => Math.hypot(est[0]-tr[0], est[1]-tr[1]);

// ── угловая карта: heatmap + маркеры + трейлы ──
function drawMap(){
  const c = document.getElementById("map"), g = c.getContext("2d");
  const t = ticksOf()[tick], n = DATA.nx, cell = c.width/n;
  for(let i=0;i<n;i++) for(let j=0;j<DATA.ny;j++){
    g.fillStyle = vir((t.map[i][j]+40)/40);
    g.fillRect(j*cell, c.height-(i+1)*cell, cell+0.5, cell+0.5); }   // x→ky, y→kx (низ=−nx/2)
  const px = q => [ (q[1]+0.5)*cell, c.height-(q[0]+0.5)*cell ];
  if(trails) for(let k=0;k<t.tg.length;k++){
    g.lineWidth = 1.5;
    for(const [key,color] of [["am","rgba(200,200,200,0.55)"],["pr","rgba(240,136,62,0.9)"]]){
      g.strokeStyle = color; g.beginPath();
      for(let s=Math.max(0,tick-24); s<=tick; s++){
        const q = ticksOf()[s].tg[k]; if(!q) continue;
        const [x,y] = px(q[key]); (s===Math.max(0,tick-24)) ? g.moveTo(x,y) : g.lineTo(x,y); }
      g.stroke(); } }
  t.tg.forEach((q,k)=>{
    let [x,y] = px(q.tr); g.strokeStyle = "#7ee787"; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x-7,y); g.lineTo(x+7,y); g.moveTo(x,y-7); g.lineTo(x,y+7); g.stroke();
    [x,y] = px(q.am); g.strokeStyle = "#e6edf3"; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(x-5,y-5); g.lineTo(x+5,y+5); g.moveTo(x+5,y-5); g.lineTo(x-5,y+5); g.stroke();
    [x,y] = px(q.pr); g.strokeStyle = "#f0883e"; g.lineWidth = 2.2;
    g.beginPath(); g.arc(x,y,6,0,7); g.stroke();
    g.fillStyle = k===sel ? "#f0883e" : "#8b949e"; g.font = "bold 12px sans-serif";
    g.fillText(DATA.names[k], x+9, y-9); });
}

// ── зум 7×7 вокруг argmax выбранной цели ──
function drawZoom(){
  const c = document.getElementById("zoom"), g = c.getContext("2d");
  const t = ticksOf()[tick], q = t.tg[sel]; if(!q) return;
  const H = 3, n = 2*H+1, cell = c.width/n, ci = q.am[0], cj = q.am[1];
  g.clearRect(0,0,c.width,c.height);
  for(let di=-H;di<=H;di++) for(let dj=-H;dj<=H;dj++){
    const i = ci+di, j = cj+dj;
    const v = (i>=0 && i<DATA.nx && j>=0 && j<DATA.ny) ? t.map[i][j] : -40;
    g.fillStyle = vir((v+40)/40);
    g.fillRect((dj+H)*cell, c.height-(di+H+1)*cell, cell+0.5, cell+0.5); }
  const px = p => [ (p[1]-cj+H+0.5)*cell, c.height-(p[0]-ci+H+0.5)*cell ];
  let [x,y] = px(q.tr); g.strokeStyle = "#7ee787"; g.lineWidth = 2.4;
  g.beginPath(); g.moveTo(x-10,y); g.lineTo(x+10,y); g.moveTo(x,y-10); g.lineTo(x,y+10); g.stroke();
  [x,y] = px(q.am); g.strokeStyle = "#e6edf3"; g.lineWidth = 2;
  g.beginPath(); g.moveTo(x-7,y-7); g.lineTo(x+7,y+7); g.moveTo(x+7,y-7); g.lineTo(x-7,y+7); g.stroke();
  [x,y] = px(q.pr); g.strokeStyle = "#f0883e"; g.lineWidth = 2.6;
  g.beginPath(); g.arc(x,y,8,0,7); g.stroke();
  document.getElementById("zoomTitle").textContent =
    `зум ${DATA.names[sel]} · ошибка argmax ${errAng(q.am,q.tr).toFixed(2)} → парабола ` +
    `${errAng(q.pr,q.tr).toFixed(3)} бина`;
}

// ── срез оси: точки + лог-парабола по 3 центральным + вертикали ──
function drawCut(axId){
  const c = document.getElementById("cut"+AX[axId]), g = c.getContext("2d");
  const q = ticksOf()[tick].tg[sel]; if(!q) return;
  const cut = q.cuts[axId], v = cut.v, lo = cut.lo;
  g.clearRect(0,0,c.width,c.height);
  const X = b => 14 + (b-lo)/(v.length-1)*(c.width-24);
  const Y = db => 8 + (-db/55)*(c.height-26);
  g.strokeStyle = "#30363d"; g.strokeRect(14, 8, c.width-24, c.height-26);
  const i0 = q.am[axId], k0 = i0-lo;
  if(k0>0 && k0<v.length-1){                       // та же квадратика, что в refine_peak (в дБ)
    const a = 0.5*(v[k0+1]-2*v[k0]+v[k0-1]), b = 0.5*(v[k0+1]-v[k0-1]);
    g.strokeStyle = "#f0883e"; g.lineWidth = 1.4; g.beginPath();
    for(let s=0;s<=60;s++){ const d = -1.4+s*(2.8/60);
      const y = Y(v[k0]+b*d+a*d*d), x = X(i0+d);
      s===0 ? g.moveTo(x,y) : g.lineTo(x,y); }
    g.stroke(); }
  for(const [val,color,w] of [[q.tr[axId],"#7ee787",2],[i0,"#8b949e",1.2],[q.pr[axId],"#f0883e",1.6]]){
    g.strokeStyle = color; g.lineWidth = w; g.beginPath();
    g.moveTo(X(val), 8); g.lineTo(X(val), c.height-18); g.stroke(); }
  g.fillStyle = "#58a6ff";
  v.forEach((db,k)=>{ g.beginPath(); g.arc(X(lo+k), Y(db), 2.6, 0, 7); g.fill(); });
  g.fillStyle = "#8b949e"; g.font = "11px sans-serif";
  g.fillText(`${AX[axId]} · истина ${q.tr[axId].toFixed(2)} · парабола ${q.pr[axId].toFixed(2)}`,
             16, c.height-4);
}

// ── ошибка по тактам ──
function drawErr(){
  const c = document.getElementById("err"), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  const T = ticksOf(), n = T.length;
  const mean = (t,key) => { const es = t.tg.map(q=>errAng(q[key],q.tr));
    return es.reduce((a,b)=>a+b,0)/Math.max(1,es.length); };
  const X = i => 30 + i/(n-1)*(c.width-40), Y = e => c.height-16 - e/0.6*(c.height-30);
  g.strokeStyle = "#30363d"; g.strokeRect(30, 6, c.width-40, c.height-22);
  g.fillStyle = "#8b949e"; g.font = "10px sans-serif";
  [0, 0.25, 0.5].forEach(e=>{ g.fillText(e.toFixed(2), 2, Y(e)+3); });
  for(const [key,color] of [["am","#8b949e"],["pr","#f0883e"]]){
    g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath();
    for(let i=0;i<n;i++){ const x = X(i), y = Y(Math.min(0.6, mean(T[i],key)));
      i===0 ? g.moveTo(x,y) : g.lineTo(x,y); }
    g.stroke(); }
  g.strokeStyle = "#58a6ff"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(X(tick), 6); g.lineTo(X(tick), c.height-16); g.stroke();
}

function drawStats(){
  const t = ticksOf()[tick];
  let h = "<tr><th>цель</th><th>истина kx/ky/f</th><th>парабола</th>" +
          "<th>ош.argmax</th><th>ош.парабола</th></tr>";
  t.tg.forEach((q,k)=>{
    h += `<tr><td class="${k===sel?'acc':''}">${DATA.names[k]}</td>` +
         `<td class="lime">${q.tr.map(v=>v.toFixed(2)).join(" / ")}</td>` +
         `<td class="acc">${q.pr.map(v=>v.toFixed(2)).join(" / ")}</td>` +
         `<td class="dim">${errAng(q.am,q.tr).toFixed(2)}</td>` +
         `<td class="acc">${errAng(q.pr,q.tr).toFixed(3)}</td></tr>`; });
  document.getElementById("stats").innerHTML = h;
}

function drawAll(){
  document.getElementById("tick").value = tick;
  document.getElementById("tickLabel").textContent = `такт ${tick+1}/${DATA.nTicks}`;
  drawMap(); drawZoom(); [0,1,2].forEach(drawCut); drawErr(); drawStats();
}

// ── контролы ──
function schedule(){ clearInterval(timer);
  if(playing) timer = setInterval(()=>{ tick = (tick+1)%DATA.nTicks; drawAll(); },
                                  +document.getElementById("speed").value); }
document.getElementById("play").onclick = e => { playing = !playing;
  e.target.textContent = playing ? "⏸ пауза" : "▶ пуск";
  e.target.classList.toggle("on", playing); schedule(); };
document.getElementById("speed").onchange = schedule;
document.getElementById("tick").oninput = e => { tick = +e.target.value; drawAll(); };
document.getElementById("trails").onclick = e => { trails = !trails;
  e.target.textContent = "трейлы: " + (trails ? "вкл" : "выкл");
  e.target.classList.toggle("on", trails); drawAll(); };
const runSel = document.getElementById("run");
for(const k of Object.keys(DATA.runs))
  runSel.add(new Option(DATA.runs[k].label, k));
runSel.onchange = () => { runKey = runSel.value; drawAll(); };
const tgSel = document.getElementById("target");
DATA.names.forEach((nm,k)=>tgSel.add(new Option("цель "+nm, k)));
tgSel.onchange = () => { sel = +tgSel.value; drawAll(); };
document.getElementById("tick").max = DATA.nTicks-1;
drawAll(); schedule();
</script></body></html>
"""


def build_page(data: dict[str, Any], out_path: Path) -> Path:
    """Встроить данные в шаблон и записать самодостаточную страницу."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(_PAGE.replace("__DATA__", payload), encoding="utf-8")
    return out_path


def main() -> None:
    p = Ex5WebParams()
    data = simulate(p)
    out = Path(__file__).resolve().parents[1] / "graphics" / "ex5_peak_refine" / "web" / "index.html"
    path = build_page(data, out)
    n_kb = path.stat().st_size // 1024
    print(f"ex5-web: {p.n_ticks} тактов × {len(p.snr_db_list)} SNR · {n_kb} КБ")
    print("  открыть в браузере:", path)


if __name__ == "__main__":
    main()
