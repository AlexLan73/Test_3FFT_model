/*
 * app.js -- веб-дашборд radar3d (P6, 4b): Three.js 3D-сцена + квадрат 16x16 +
 * контролы + закладка +-N. Подписка на `WebSocketTransport` (core/runtime/
 * transport.py), декод -- `codec.js` (та же msgpack-схема, что десктоп-панель,
 * N2). Самодостаточно: без сборки, открывается прямо в браузере (index.html).
 *
 * Конвенция осей (везде в проекте, TASK P6): дальность (range) -- по горизонтали
 * (вдаль от камеры), kx (азимут) -- вбок, ky (угол места) -- вверх.
 */

const state = {
  ws: null,
  connected: false,
  lastSquare: null,     // {shape:[nx,ny], real: Float32Array} (magnitude, не dB)
  lastTracks: null,     // {targets:[...], jammers:[...]}
  neighborPlanes: 5,
  log: [],
};

function log(msg) {
  state.log.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`);
  state.log = state.log.slice(0, 30);
  const el = document.getElementById("log");
  if (el) el.textContent = state.log.join("\n");
}

// ---------------------------------------------------------------------------
// WebSocket: подключение к WebSocketTransport (python), декод msgpack-кадров.
// ---------------------------------------------------------------------------
function connect(url) {
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { state.connected = true; log(`подключено: ${url}`); };
  ws.onclose = () => { state.connected = false; log("соединение закрыто"); };
  ws.onerror = (e) => log(`ошибка WS: ${e.message || e}`);
  ws.onmessage = (event) => {
    try {
      const frame = decodeFrame(event.data);
      onFrame(frame);
    } catch (err) {
      log(`ошибка декода кадра: ${err}`);
    }
  };
  state.ws = ws;
}

function onFrame(frame) {
  if (frame.topic === "squares") {
    state.lastSquare = { shape: frame.payload.shape, real: frame.payload.real };
    drawSquares(state.lastSquare);
  } else if (frame.topic === "tracks") {
    state.lastTracks = frame.payload;
    drawTracksOverlay(frame.payload);
    updateScene3D(frame.payload);
  } else if (frame.topic === "cube") {
    // сырой объём (nx,ny,N complex) -- полный 3D-обзор по кнопке "снимок" (см. README блока).
    state.lastCube = frame.payload;
  }
}

// ---------------------------------------------------------------------------
// Квадрат 16x16 -- 2D canvas, цветовая карта (аналог color_map образца, N6 --
// пишем своё, образец недоступен).
// ---------------------------------------------------------------------------
function drawSquares(square) {
  const canvas = document.getElementById("squares_canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const [nx, ny] = square.shape;
  const cell = Math.floor(Math.min(canvas.width / nx, canvas.height / ny));
  let vmax = 1e-12;
  for (let i = 0; i < square.real.length; i++) vmax = Math.max(vmax, square.real[i]);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const v = square.real[ix * ny + iy] / vmax;
      ctx.fillStyle = valueToColor(v);
      ctx.fillRect(ix * cell, (ny - 1 - iy) * cell, cell - 1, cell - 1);
    }
  }
}

function valueToColor(t) {
  t = Math.max(0, Math.min(1, t));
  const cold = [13, 17, 23];
  const hot = [248, 81, 73];
  const r = Math.round(cold[0] * (1 - t) + hot[0] * t);
  const g = Math.round(cold[1] * (1 - t) + hot[1] * t);
  const b = Math.round(cold[2] * (1 - t) + hot[2] * t);
  return `rgb(${r},${g},${b})`;
}

function drawTracksOverlay(tracks) {
  const el = document.getElementById("tracks_info");
  if (!el) return;
  const lines = [];
  for (const t of tracks.targets || []) {
    lines.push(`цель #${t.id}: kx=${t.kx.toFixed(2)} ky=${t.ky.toFixed(2)} R=${t.r.toFixed(0)}м`);
  }
  for (const j of tracks.jammers || []) {
    lines.push(`заград[${j.kind}]: kx=${j.kx.toFixed(2)} ky=${j.ky.toFixed(2)}`);
  }
  el.textContent = lines.join("\n") || "(нет треков)";
}

// ---------------------------------------------------------------------------
// Three.js 3D-сцена: дальность -- по горизонтали (X, вдаль), kx -- вбок (Z),
// ky -- вверх (Y). Прообраз -- интерактивный вьюер P1 (реальные данные,
// вращение+play), здесь -- живые точки треков вместо статичного облака.
// ---------------------------------------------------------------------------
let scene3d = null, camera3d = null, renderer3d = null, targetMeshes = {};

function initScene3D() {
  const container = document.getElementById("scene3d");
  if (!container || typeof THREE === "undefined") return;
  scene3d = new THREE.Scene();
  scene3d.background = new THREE.Color(0x0d1117);
  camera3d = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 100000);
  camera3d.position.set(-6000, 3000, 9000);
  camera3d.lookAt(4000, 0, 0);
  renderer3d = new THREE.WebGLRenderer({ antialias: true });
  renderer3d.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer3d.domElement);

  const grid = new THREE.GridHelper(20000, 20, 0x30363d, 0x30363d);
  grid.rotation.x = 0;   // плоскость дальность(X)/kx(Z) -- "пол" сцены
  scene3d.add(grid);
  const axes = new THREE.AxesHelper(2000);
  scene3d.add(axes);

  animate3d();
}

function animate3d() {
  requestAnimationFrame(animate3d);
  if (renderer3d) renderer3d.render(scene3d, camera3d);
}

function updateScene3D(tracks) {
  if (!scene3d) return;
  const seen = new Set();
  for (const t of tracks.targets || []) {
    seen.add(t.id);
    let mesh = targetMeshes[t.id];
    if (!mesh) {
      const geo = new THREE.SphereGeometry(80, 12, 12);
      const mat = new THREE.MeshBasicMaterial({ color: 0x58a6ff });
      mesh = new THREE.Mesh(geo, mat);
      scene3d.add(mesh);
      targetMeshes[t.id] = mesh;
    }
    // конвенция осей: дальность R -> X (вдаль), kx -> Z (вбок), ky -> Y (вверх)
    mesh.position.set(t.r, t.ky * 40, t.kx * 40);
  }
  for (const id of Object.keys(targetMeshes)) {
    if (!seen.has(Number(id))) {
      scene3d.remove(targetMeshes[id]);
      delete targetMeshes[id];
    }
  }
}

// ---------------------------------------------------------------------------
// Контролы: play/пауза (визуальный флаг -- поток кадров решает сервер),
// добавить цель / вкл-выкл заград -- отправка команд НЕ входит в объём
// WebSocketTransport (P6, publish-only шлюз, см. transport.py докстринг) --
// эти кнопки в браузер-варианте информационные (команды -- десктоп-панель).
// ---------------------------------------------------------------------------
function setupControls() {
  const slider = document.getElementById("neighbor_planes");
  if (slider) {
    slider.addEventListener("input", (e) => {
      state.neighborPlanes = Number(e.target.value);
      document.getElementById("neighbor_planes_label").textContent = String(state.neighborPlanes);
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  setupControls();
  initScene3D();
  const url = document.getElementById("ws_url").value;
  document.getElementById("connect_btn").addEventListener("click", () => connect(document.getElementById("ws_url").value));
  log("готово -- нажмите 'Подключиться' для приёма кадров сцены");
});
