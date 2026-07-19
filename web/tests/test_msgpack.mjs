/*
 * test_msgpack.mjs — dev-проверка JS-декодера web/msgpack.js против python-msgpack.
 *
 * НЕ часть python-приёмки (правило 04 — TestRunner; это тонкий JS-клиент, node-инструмент).
 * Гоняется руками при правке декодера/схемы:
 *   1) сгенерировать эталон:  .venv/Scripts/python.exe web/tests/gen_msgpack_ref.py <dir>
 *   2) проверить:             node web/tests/test_msgpack.mjs <dir>
 * <dir> содержит msgs.bin (кадры с 4-байтным BE-префиксом длины) + ref.json (эталон).
 */
import { readFileSync } from "fs";

const dir = process.argv[2];
if (!dir) { console.error("usage: node test_msgpack.mjs <dir>"); process.exit(2); }

const g = globalThis;
await import(new URL("../msgpack.js", import.meta.url).href);  // side-effect: задаёт globalThis.MPack

const buf = readFileSync(dir + "/msgs.bin");
const ref = JSON.parse(readFileSync(dir + "/ref.json", "utf-8"));
let p = 0;
const out = [];
while (p < buf.length) {
  const n = buf.readUInt32BE(p); p += 4;
  const doc = g.MPack.decode(new Uint8Array(buf.subarray(p, p + n))); p += n;
  out.push({ topic: doc.topic, tact: doc.tact, payload: doc.payload });
}
const norm = (o) => JSON.parse(JSON.stringify(o, (k, v) => (Object.is(v, -0) ? 0 : v)));
if (JSON.stringify(norm(out)) === JSON.stringify(norm(ref))) {
  console.log("✅ msgpack.js совпал с python-эталоном (" + out.length + " сообщений)");
} else {
  console.log("❌ РАСХОЖДЕНИЕ msgpack.js ↔ python");
  process.exit(1);
}
