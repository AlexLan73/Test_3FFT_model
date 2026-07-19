/*
 * app.js — тонкий клиент живой панели ex4 (канон §1.6, P6). Подписка на
 * `WebSocketTransport` (core/runtime/transport.py), декод — web/msgpack.js
 * (зеркало схемы core/runtime/codec.py, ноль внешних URL, офлайн).
 *
 * Клиент НЕ считает сцену: сервер (demo/ex4_flight/server.py) шлёт МИРОВЫЕ
 * координаты (kx,ky,r) + примитивы; проекция мир→экран — единой камерой
 * core.graphics.Projection (параметры meta.cam = Projection.as_js(), тот же
 * источник знаков осей, что тесты test_camera.py). Вращение мышью меняет ТОЛЬКО
 * ракурс (az,el) локально — виды поле②/3D① не могут разойтись by construction.
 *
 * Каналы: "meta" (апертура/камера/станции/финал-признаки, раз на цикл) +
 * "tick" (такт: истина/точки/треки/срезы). Такты копятся в буфер `ticks[]` по
 * индексу — play/pause/слайдер работают по принятому буферу, поток обновляет его.
 */
"use strict";

const C = { trk:"#58a6ff", comb:"#f0883e", bar:"#f85149", fg:"#c9d1d9", dim:"#8b949e",
            radar:"#a371f7", ham:"#3fb950" };
const FEATS = [["pr","PR","пик/фон: у точечной цели мал, у шума ~сотни"],
  ["hoyer","Hoyer","разреженность спектра окна (0=размазан, 1=один пик)"],
  ["main_frac","MainFrac","доля энергии главного лепестка"],
  ["lobe_ratio","LobeRatio","боковой лепесток / главный"],
  ["max_mean","MaxMean","контраст: максимум / среднее окна"],
  ["energy","Energy","полная энергия окна"]];

// ── состояние клиента ──
let meta = null;                 // {nx,ny,nAxis,kTrail,nTicks,stats,finalFeats,stations,cam}
const ticks = [];                // буфер тактов по индексу (пополняется по WS)
let seen = 0;                    // сколько РАЗНЫХ индексов принято (для слайдера)
let NX = 64, NY = 64, NAX = 4096, K = 8;
let AZ0 = 0.42, EL0 = -0.32, FIT = 0.7;
let FAZ = Math.PI, FEL = 0, FFIT = 0.92;   // field-камера (вид с нулевой дальности, az=π)
let tick = 0, playing = true, trails = true, sel = null, timer = null, zoom = 1.0;
let ws = null;

// ═══ Projection — ЕДИНАЯ модель камеры сцены (параметры из core.graphics.Projection) ═══
// Конвенция знаков осей общая для ОБОИХ видов: +kx → влево, +ky → вверх, дальность → вглубь.
// Поле② = проекция без наклона; 3D① = та же камера + орбита (az,el). Формула повторяет
// core/graphics/camera.py::Projection.project (единый источник знаков ⇒ зеркальность невозможна).
const Projection = {
  az: 0.42, el: -0.32,
  reset(){ this.az = AZ0; this.el = EL0; },
  // ЕДИНАЯ формула проекции — зеркало core/graphics/camera.py::Projection.project
  // (ndc x,y + depth). Оба окна зовут ЕЁ, разница только в ракурсе камеры (az,el,fit).
  _ndc(az, el, fit, kx, ky, r){
    const ax = kx/(NX/2), ay = ky/(NY/2), zc = (r/NAX - 0.5)*2;
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const xr = ax*ca - zc*sa, zr = ax*sa + zc*ca;
    const yr = ay*ce - zr*se, depth = ay*se + zr*ce;
    return [-xr*fit, -yr*fit, depth];             // −x: kx; −y: ky (как camera.py)
  },
  // 3D ① — вращаемый облётный ракурс (this.az/el, зум)
  scene(kx, ky, r, W, H){
    const s = 175*zoom, n = this._ndc(this.az, this.el, FIT, kx, ky, r);
    return [W/2 + n[0]*s, H/2 + n[1]*s, n[2]];
  },
  // поле ② — field-камера (вид С НУЛЕВОЙ дальности, az=π из meta.cam.field), дальность схлопнута
  field(kx, ky, W, H){
    const n = this._ndc(FAZ, FEL, FFIT, kx, ky, 0);
    return [W/2 + n[0]*(W/2-30), H/2 + n[1]*(H/2-32)];
  },
};

// turbo-колормапа (опорные точки + лерп), диапазон дБ [-25, 0]
const TB = [[48,18,59],[70,107,227],[40,187,235],[74,236,152],[182,240,66],
            [249,189,38],[245,96,23],[190,25,7]];
function turbo(db){ const t = Math.min(1, Math.max(0, (db+25)/25));
  const x = t*(TB.length-1), i = Math.floor(x), f = x-i, a = TB[i],
        b = TB[Math.min(i+1, TB.length-1)];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")})`; }

// ── значки помех (canvas-глифы) ──
function glyphTri(g, x, y, color){
  g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
  g.moveTo(x, y-7); g.lineTo(x-6, y+5); g.lineTo(x+6, y+5); g.closePath(); g.stroke(); }
function glyphSquare(g, x, y, color, h=4){
  g.strokeStyle = color; g.lineWidth = 2; g.strokeRect(x-h, y-h, 2*h, 2*h); }
function glyphAntenna(g, x, y, color){
  g.strokeStyle = color; g.lineWidth = 1.6;
  g.beginPath(); g.moveTo(x, y+8); g.lineTo(x, y-5); g.stroke();
  g.beginPath(); g.moveTo(x-5, y+8); g.lineTo(x, y+1); g.lineTo(x+5, y+8); g.stroke();
  for(const r of [4, 7]){ g.beginPath();
    g.arc(x, y-5, r, -Math.PI*0.6, Math.PI*0.1); g.stroke(); } }
function glyphHam(g, x, y, color){
  g.fillStyle = color; g.beginPath(); g.arc(x, y, 2.6, 0, 7); g.fill();
  g.strokeStyle = color; g.lineWidth = 1;
  for(const r of [5, 8]){ g.beginPath(); g.arc(x, y, r, 0, 7); g.stroke(); } }
function drawStation(g, x, y, type){
  if(type === "radar") glyphAntenna(g, x, y, C.radar); else glyphHam(g, x, y, C.ham); }

const T = () => ticks[tick];               // текущий такт (может быть undefined до прихода)
const ready = () => meta !== null && T() !== undefined;

// ── ① 3D ──
function p3(kx, pos, ky){
  const c = document.getElementById("c3d");
  return Projection.scene(kx, ky, pos, c.width, c.height);
}
function line3(g, a, b){ g.beginPath(); g.moveTo(a[0],a[1]); g.lineTo(b[0],b[1]); g.stroke(); }
function poly3(g, pts){ g.beginPath();
  pts.forEach((p,i)=> i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1])); g.closePath(); g.stroke(); }
function tri(g, p, color){ g.fillStyle = color; g.beginPath();
  g.moveTo(p[0], p[1]-6); g.lineTo(p[0]-5, p[1]+4); g.lineTo(p[0]+5, p[1]+4);
  g.closePath(); g.fill(); }
const ticksArr = (a, b, n) => Array.from({length:n+1}, (_,i)=> a+(b-a)*i/n);
const fmt = v => Math.round(v).toString();
function depth(kx, pos, ky){ return p3(kx, pos, ky)[2]; }
function draw3d(){
  const c = document.getElementById("c3d"), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  if(!ready()) return;
  const t = T(), E = NX/2, F = NY/2;
  const TKX = ticksArr(-E, E, 4), TKY = ticksArr(-F, F, 4), TP = ticksArr(0, NAX, 4);
  const wKx = depth(-E, NAX/2, 0) >= depth(E, NAX/2, 0) ? -E : E;
  const wP  = depth(0, 0, 0)      >= depth(0, NAX, 0)    ?  0 : NAX;
  const wKy = depth(0, NAX/2, -F) >= depth(0, NAX/2, F)  ? -F : F;
  g.setLineDash([3, 4]);
  g.strokeStyle = "#3a4457"; g.lineWidth = 1;
  for(const pv of TP) line3(g, p3(wKx, pv, -F), p3(wKx, pv, F));
  for(const kv of TKY) line3(g, p3(wKx, 0, kv), p3(wKx, NAX, kv));
  for(const kv of TKX) line3(g, p3(kv, wP, -F), p3(kv, wP, F));
  for(const kv of TKY) line3(g, p3(-E, wP, kv), p3(E, wP, kv));
  for(const kv of TKX) line3(g, p3(kv, 0, wKy), p3(kv, NAX, wKy));
  for(const pv of TP) line3(g, p3(-E, pv, wKy), p3(E, pv, wKy));
  g.setLineDash([]);
  g.strokeStyle = "#586275"; g.lineWidth = 1.3;
  poly3(g, [p3(wKx,0,-F), p3(wKx,NAX,-F), p3(wKx,NAX,F), p3(wKx,0,F)]);
  poly3(g, [p3(-E,wP,-F), p3(E,wP,-F), p3(E,wP,F), p3(-E,wP,F)]);
  poly3(g, [p3(-E,0,wKy), p3(E,0,wKy), p3(E,NAX,wKy), p3(-E,NAX,wKy)]);
  g.strokeStyle = "#4a5163"; g.lineWidth = 1;
  for(const s of [-1,1]) for(const q of [-1,1]){
    line3(g, p3(-E, (s+1)/2*NAX, q*F), p3(E, (s+1)/2*NAX, q*F));
    line3(g, p3(s*E, 0, q*F), p3(s*E, NAX, q*F));
    line3(g, p3(s*E, (q+1)/2*NAX, -F), p3(s*E, (q+1)/2*NAX, F)); }
  g.fillStyle = C.dim; g.font = "10px sans-serif"; g.textAlign = "center";
  for(const kv of TKX){ if(kv === wKx) continue;
    const L = p3(kv, wP, wKy); g.fillText(fmt(kv), L[0], L[1]+13); }
  for(const pv of TP){ if(pv === wP) continue;
    const L = p3(wKx, pv, wKy); g.fillText(fmt(pv), L[0], L[1]+13); }
  g.textAlign = "right";
  for(const kv of TKY){ if(kv === wKy) continue;
    const L = p3(wKx, wP, kv); g.fillText(fmt(kv), L[0]-8, L[1]+3); }
  g.fillStyle = C.fg; g.font = "bold 11px sans-serif"; g.textAlign = "center";
  const O = p3(0, NAX/2, 0);
  const outward = (L, k=26) => { const dx = L[0]-O[0], dy = L[1]-O[1],
    n = Math.hypot(dx,dy)||1; return [L[0]+dx/n*k, L[1]+dy/n*k]; };
  let L = outward(p3(0, wP, wKy)); g.fillText("kx (азимут)", L[0], L[1]);
  L = outward(p3(wKx, NAX/2, wKy)); g.fillText("позиция →", L[0], L[1]);
  L = outward(p3(wKx, wP, 0)); g.fillText("ky", L[0], L[1]);
  g.textAlign = "left";
  const b = t.truth.b;
  g.strokeStyle = C.bar; g.lineWidth = 4; g.globalAlpha = 0.5;
  line3(g, p3(b[0], 0, b[1]), p3(b[0], NAX, b[1]));
  g.globalAlpha = 1;
  if(trails) for(const [key,color] of [["t",C.trk],["c",C.comb]]){
    g.strokeStyle = color; g.lineWidth = 1.3; g.globalAlpha = 0.55; g.beginPath();
    let started = false;
    for(let s = Math.max(0, tick-K); s <= tick; s++){
      const tk = ticks[s]; if(tk === undefined) continue;
      const q = tk.truth[key], pp = p3(q[0], q[2], q[1]);
      started ? g.lineTo(pp[0],pp[1]) : (g.moveTo(pp[0],pp[1]), started = true); }
    g.stroke(); g.globalAlpha = 1; }
  t.pts.map(q => [p3(q[0], q[2], q[1]), q[3]])
    .sort((a,b2)=>b2[0][2]-a[0][2])
    .forEach(([pp,db])=>{ g.fillStyle = turbo(db);
      g.beginPath(); g.arc(pp[0], pp[1], 3.5, 0, 7); g.fill(); });
  tri(g, p3(t.truth.t[0], t.truth.t[2], t.truth.t[1]), C.trk);
  tri(g, p3(t.truth.c[0], t.truth.c[2], t.truth.c[1]), C.comb);
  for(const st of meta.stations){
    const pp = p3(st.kx, 0, st.ky); drawStation(g, pp[0], pp[1], st.type); }
  // сквозные номера треков в 3D (те же track_id, что поле/срезы/таблица) — у позиции (kx,pos,ky)
  g.font = "bold 12px sans-serif"; g.textAlign = "left";
  for(const s of t.sl){
    const pp = p3(s.kx, s.pos, s.ky);
    g.fillStyle = "#0d1117"; g.lineWidth = 3; g.strokeStyle = "#0d1117";
    g.strokeText("№"+s.id, pp[0]+7, pp[1]-6);
    g.fillStyle = (sel===s.id) ? C.comb : C.trk;
    g.fillText("№"+s.id, pp[0]+7, pp[1]-6); }
}

// ── ② поле ──
function fx(kx){ const c = document.getElementById("fld");
  return Projection.field(kx, 0, c.width, c.height)[0]; }
function fy(ky){ const c = document.getElementById("fld");
  return Projection.field(0, ky, c.width, c.height)[1]; }
function drawField(){
  const c = document.getElementById("fld"), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  if(!ready()) return;
  g.strokeStyle = "#21262d"; g.lineWidth = 1;
  for(let k = -NX/2; k <= NX/2; k += 10){
    g.beginPath(); g.moveTo(fx(k), fy(-NY/2)); g.lineTo(fx(k), fy(NY/2)); g.stroke();
    g.beginPath(); g.moveTo(fx(-NX/2), fy(k)); g.lineTo(fx(NX/2), fy(k)); g.stroke(); }
  g.strokeStyle = "#30363d"; g.strokeRect(fx(-NX/2), fy(NY/2), fx(NX/2)-fx(-NX/2),
                                          fy(-NY/2)-fy(NY/2));
  g.fillStyle = C.dim; g.font = "10px sans-serif";
  g.textAlign = "center";
  for(let k = -NX/2; k <= NX/2; k += 10) g.fillText(k, fx(k), fy(-NY/2)+14);
  g.textAlign = "right";
  for(let k = -NY/2; k <= NY/2; k += 10) g.fillText(k, fx(-NX/2)-6, fy(k)+3);
  g.textAlign = "center"; g.fillStyle = C.fg; g.font = "bold 11px sans-serif";
  g.fillText("kx (азимут)", c.width/2, c.height-5);
  g.save(); g.translate(12, c.height/2+40); g.rotate(-Math.PI/2);
  g.fillText("ky (угол места)", 0, 0); g.restore();
  g.textAlign = "left";
  const t = T();
  const labels = [];                          // подписи собираем отдельно (anti-overlap ниже)
  for(const tr of t.trk){
    const hist = trails ? tr.h : tr.h.slice(-1), n = hist.length;
    const stable = tr.h.length >= 3;
    hist.forEach((hq,j)=>{ const a = 0.15+0.85*(j+1)/n;
      g.globalAlpha = stable ? a : 0.35; g.fillStyle = C.trk;
      g.beginPath(); g.arc(fx(hq[0]), fy(hq[1]), stable ? (2+4*a) : 2, 0, 7); g.fill(); });
    g.globalAlpha = 1;
    if(stable) labels.push({id: tr.id, ax: fx(tr.kx), ay: fy(tr.ky),
                            text: "№"+tr.id+(tr.mv ? "✈" : "")});
    if(sel===tr.id){ g.strokeStyle = C.comb; g.lineWidth = 1.6;
      g.beginPath(); g.arc(fx(tr.kx), fy(tr.ky), 11, 0, 7); g.stroke(); } }
  // разведение подписей: близкие треки (копии гребёнки) — сдвигаем метку вниз, пока не свободно
  g.font = "bold 11px sans-serif";
  const placed = [];
  for(const L of labels.sort((a,b)=>a.ay-b.ay)){
    let lx = L.ax+8, ly = L.ay-6, guard = 0;
    while(guard++ < 12 && placed.some(p => Math.abs(p.x-lx) < 34 && Math.abs(p.y-ly) < 13))
      ly += 13;
    placed.push({x: lx, y: ly});
    g.strokeStyle = "#0d1117"; g.lineWidth = 3; g.strokeText(L.text, lx, ly);  // обводка-читаемость
    g.fillStyle = (sel===L.id) ? C.comb : C.trk; g.fillText(L.text, lx, ly);
    if(ly !== L.ay-6){ g.strokeStyle = "#3a4457"; g.lineWidth = 0.8;           // лидер-линия к точке
      g.beginPath(); g.moveTo(L.ax, L.ay); g.lineTo(lx-2, ly-3); g.stroke(); } }
  const cb = t.truth.c;
  glyphTri(g, fx(cb[0]), fy(cb[1]), C.comb);
  const b = t.truth.b;
  glyphSquare(g, fx(b[0]), fy(b[1]), C.bar, 5);
  for(const st of meta.stations)
    drawStation(g, fx(st.kx), fy(st.ky), st.type);
}

// ── ③ срезы + таблица ──
function drawSlices(){
  const box = document.getElementById("slices");
  box.innerHTML = "";
  if(!ready()){ box.innerHTML = "<div style='color:#8b949e;padding:24px'>ожидание сервера…</div>";
    return; }
  const t = T();
  if(!t.sl.length)
    box.innerHTML = "<div style='color:#8b949e;padding:24px'>треки формируются " +
                    "(срезы появляются с возраста 2 такта)…</div>";
  for(const s of t.sl){
    const d = document.createElement("div");
    d.className = "sl"+(sel===s.id ? " sel" : "");
    const cv = document.createElement("canvas");
    const n = s.m.length; cv.width = cv.height = 187;
    const g = cv.getContext("2d"), cell = cv.width/n;
    for(let i=0;i<n;i++) for(let j=0;j<s.m[i].length;j++){
      g.fillStyle = turbo(s.m[i][j]);
      g.fillRect(j*cell, cv.height-(i+1)*cell, cell+0.5, cell+0.5); }
    const mx = (s.ky+NY/2-s.y0+0.5)*cell, my = cv.height-(s.kx+NX/2-s.x0+0.5)*cell;
    g.strokeStyle = "#fff"; g.lineWidth = 1.6;
    g.beginPath(); g.arc(mx, my, 9, 0, 7); g.stroke();
    g.fillStyle = "#fff"; g.font = "bold 15px sans-serif"; g.textAlign = "center";  // сквозной №
    g.strokeStyle = "#0d1117"; g.lineWidth = 3; g.strokeText("№"+s.id, mx, my-13);
    g.fillText("№"+s.id, mx, my-13);
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = `№${s.id} окно=${s.pos} kx=${s.kx>=0?"+":""}${s.kx} ` +
                      `ky=${s.ky>=0?"+":""}${s.ky} ${s.mv ? "ЛЕТИТ" : "—"}`;
    d.appendChild(cv); d.appendChild(cap);
    d.onclick = () => { sel = (sel===s.id) ? null : s.id; drawAll(); };
    box.appendChild(d); }
}
function drawTable(){
  const tbl = document.getElementById("tbl");
  if(!ready()){ tbl.innerHTML = ""; return; }
  const t = T();
  let h = "<tr><th>№</th><th>окно</th><th>kx</th><th>ky</th>";
  for(const [,name,tip] of FEATS) h += `<th title="${tip}">${name}</th>`;
  h += `<th title="вердикт из ТРЕКА (is_moving, §4-бис.4), не из куба">вердикт</th></tr>`;
  for(const s of t.sl){
    const ff = meta.finalFeats[String(s.id)];
    h += `<tr data-id="${s.id}" class="${sel===s.id?'sel':''}">` +
         `<td><b>${s.id}</b></td><td>${s.pos}</td><td>${s.kx}</td><td>${s.ky}</td>`;
    for(const [key] of FEATS){
      const v = ff ? ff.f[key] : undefined;
      h += `<td>${v===undefined ? "·" : Number(v).toPrecision(3)}</td>`; }
    h += `<td class="${s.mv?'mv':'no'}">${s.mv ? "✈ ЛЕТИТ" : "—"}</td></tr>`; }
  tbl.innerHTML = h;
  tbl.querySelectorAll("tr[data-id]").forEach(r => r.onclick = () => {
    const id = +r.dataset.id; sel = (sel===id) ? null : id; drawAll(); });
}
function drawStatus(){
  const nT = meta ? meta.nTicks : 0;
  document.getElementById("tick").value = tick;
  const t = T();
  document.getElementById("status").innerHTML = !ready() ? "" :
    `такт ${tick+1}/${nT}` +
    (t.band ? ` · <b>полоса@(${t.band[0]}, ${t.band[1]})→null</b>` : "") +
    (meta.stats.target_found ? ` · цель найдена ${meta.stats.target_found}` : "") +
    (meta.stats.models ? ` · ${meta.stats.models}` : "");
}
function drawAll(){ draw3d(); drawField(); drawSlices(); drawTable(); drawStatus(); }

// ── WebSocket: подписка на meta/tick ──
function setConn(text, live){
  const el = document.getElementById("connState");
  el.textContent = text; el.classList.toggle("live", !!live);
}
function onFrame(topic, payload, tact){
  if(topic === "meta"){
    const wasReady = meta !== null;
    meta = payload;
    NX = meta.nx; NY = meta.ny; NAX = meta.nAxis; K = meta.kTrail;
    const cam = meta.cam;
    AZ0 = cam.az0; EL0 = cam.el0; FIT = cam.scene.fit;
    FAZ = cam.field.az; FEL = cam.field.el; FFIT = cam.field.fit;   // вид поля с нуля (az=π)
    if(!wasReady){ Projection.reset();
      document.getElementById("tick").max = meta.nTicks-1;
      buildLegend();
      const hm = location.hash.match(/t=(\d+)/);   // deep-link #t=N: открыть такт на паузе
      if(hm){ tick = Math.min(meta.nTicks-1, +hm[1]); playing = false;
        const pb = document.getElementById("play");
        pb.textContent = "▶ пуск"; pb.classList.remove("on"); clearInterval(timer); } }
    setConn(`поток · ${meta.nTicks} тактов`, true);
  } else if(topic === "tick"){
    if(ticks[tact] === undefined) seen++;
    ticks[tact] = payload;
    if(meta){ document.getElementById("tick").max = meta.nTicks-1; }
    if(!ready()) return;
    // при живом воспроизведении «догоняем» край потока, если стоим на нём
    drawAll();
  }
}
function connect(url){
  if(ws){ try{ ws.close(); }catch(e){} }
  setConn("подключение…", false);
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setConn("подключено, ждём кадры…", false);
  ws.onclose = () => setConn("соединение закрыто", false);
  ws.onerror = () => setConn("ошибка соединения", false);
  ws.onmessage = ev => {
    try {
      const doc = MPack.decode(new Uint8Array(ev.data));
      onFrame(doc.topic, doc.payload, doc.tact);
    } catch(e){ setConn("ошибка декода: "+e.message, false); }
  };
}

// ── легенда признаков ──
function buildLegend(){
  document.getElementById("legend").innerHTML =
    FEATS.map(([,n,tip]) => `<b>${n}</b> — ${tip}`).join(" · ") +
    " · <b>вердикт</b> — «летит» решает ТРЕК (движение между тактами, §4-бис.4), не куб";
}

// ── контролы ──
function schedule(){ clearInterval(timer);
  if(playing) timer = setInterval(()=>{
    if(!meta) return;
    tick = (tick+1)%meta.nTicks; drawAll(); },
    +document.getElementById("speed").value); }
document.getElementById("connect").onclick = () =>
  connect(document.getElementById("wsUrl").value);
document.getElementById("play").onclick = e => { playing = !playing;
  e.target.textContent = playing ? "⏸ пауза" : "▶ пуск";
  e.target.classList.toggle("on", playing); schedule(); };
document.getElementById("speed").onchange = schedule;
document.getElementById("tick").oninput = e => { tick = +e.target.value; drawAll(); };
document.getElementById("trails").onclick = e => { trails = !trails;
  e.target.textContent = "след: " + (trails ? "вкл" : "выкл");
  e.target.classList.toggle("on", trails); drawAll(); };
document.getElementById("legendBtn").onclick = () =>
  document.getElementById("legend").classList.toggle("show");
document.getElementById("resetView").onclick = () => {
  Projection.reset(); zoom = 1.0; draw3d(); };
const c3d = document.getElementById("c3d");
let drag = null;
c3d.onmousedown = e => drag = [e.clientX, e.clientY];
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
window.onmousemove = e => { if(!drag) return;
  Projection.az += (e.clientX-drag[0])*0.008;
  Projection.el = clamp(Projection.el + (e.clientY-drag[1])*0.008, -1.4, 1.4);
  drag = [e.clientX, e.clientY]; draw3d(); };
window.onmouseup = () => drag = null;
c3d.onwheel = e => { e.preventDefault();
  zoom = Math.max(0.4, Math.min(3, zoom*(e.deltaY < 0 ? 1.1 : 0.9))); draw3d(); };

drawAll(); schedule();
connect(document.getElementById("wsUrl").value);   // автоподключение к дефолтному адресу
