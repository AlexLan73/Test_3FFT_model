/*
 * msgpack.js — компактный самодостаточный msgpack-ДЕКОДЕР (ноль внешних URL, офлайн).
 *
 * Покрывает РОВНО ту схему, что пишет `core/runtime/codec.py`
 * (`msgpack.packb(..., use_bin_type=True)`): map / str / bin / int (все ширины) /
 * float32|64 / array / nil / bool. Этого достаточно для кадров транспорта
 * (topic/tact/kind/payload + сырые байты массивов). Экзотику (ext, map32>2^16 и т.п.)
 * сцена не генерирует — намеренно НЕ реализуем (меньше кода = меньше багов в базе).
 *
 * ⚠️ msgpack НА ПРОВОДЕ — big-endian (сетевой порядок). Не путать с little-endian
 * ВНУТРИ payload.data (сырые numpy-байты массива) — это отдельный слой, его читает codec.js.
 *
 * Проверено против python-msgpack через node (см. web/tests/test_msgpack.mjs).
 */
"use strict";

(function (root) {
  function Decoder(buf) {
    this.u = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    this.dv = new DataView(this.u.buffer, this.u.byteOffset, this.u.byteLength);
    this.p = 0;
  }

  Decoder.prototype.u8 = function () { return this.u[this.p++]; };

  Decoder.prototype.bytes = function (n) {
    const out = this.u.subarray(this.p, this.p + n);
    this.p += n;
    return out;
  };

  Decoder.prototype.str = function (n) {
    // UTF-8 → JS-строка (ключи/значения схемы — ASCII/кириллица, TextDecoder корректен).
    const s = new TextDecoder("utf-8").decode(this.u.subarray(this.p, this.p + n));
    this.p += n;
    return s;
  };

  Decoder.prototype.arr = function (n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.decode();
    return out;
  };

  Decoder.prototype.map = function (n) {
    const out = {};
    for (let i = 0; i < n; i++) {
      const k = this.decode();
      out[k] = this.decode();
    }
    return out;
  };

  Decoder.prototype.decode = function () {
    const b = this.u8();
    // positive fixint 0x00..0x7f
    if (b <= 0x7f) return b;
    // negative fixint 0xe0..0xff
    if (b >= 0xe0) return b - 0x100;
    // fixstr 0xa0..0xbf
    if (b >= 0xa0 && b <= 0xbf) return this.str(b & 0x1f);
    // fixarray 0x90..0x9f
    if (b >= 0x90 && b <= 0x9f) return this.arr(b & 0x0f);
    // fixmap 0x80..0x8f
    if (b >= 0x80 && b <= 0x8f) return this.map(b & 0x0f);

    const dv = this.dv;
    let v;
    switch (b) {
      case 0xc0: return null;             // nil
      case 0xc2: return false;            // false
      case 0xc3: return true;             // true

      case 0xc4: return this.bytes(this.u8());                      // bin8
      case 0xc5: v = dv.getUint16(this.p); this.p += 2; return this.bytes(v);   // bin16
      case 0xc6: v = dv.getUint32(this.p); this.p += 4; return this.bytes(v);   // bin32

      case 0xca: v = dv.getFloat32(this.p); this.p += 4; return v;  // float32
      case 0xcb: v = dv.getFloat64(this.p); this.p += 8; return v;  // float64

      case 0xcc: return this.u8();                                 // uint8
      case 0xcd: v = dv.getUint16(this.p); this.p += 2; return v;   // uint16
      case 0xce: v = dv.getUint32(this.p); this.p += 4; return v;   // uint32
      case 0xcf: v = Number(dv.getBigUint64(this.p)); this.p += 8; return v;    // uint64

      case 0xd0: v = dv.getInt8(this.p); this.p += 1; return v;     // int8
      case 0xd1: v = dv.getInt16(this.p); this.p += 2; return v;    // int16
      case 0xd2: v = dv.getInt32(this.p); this.p += 4; return v;    // int32
      case 0xd3: v = Number(dv.getBigInt64(this.p)); this.p += 8; return v;     // int64

      case 0xd9: return this.str(this.u8());                       // str8
      case 0xda: v = dv.getUint16(this.p); this.p += 2; return this.str(v);     // str16
      case 0xdb: v = dv.getUint32(this.p); this.p += 4; return this.str(v);     // str32

      case 0xdc: v = dv.getUint16(this.p); this.p += 2; return this.arr(v);     // array16
      case 0xdd: v = dv.getUint32(this.p); this.p += 4; return this.arr(v);     // array32

      case 0xde: v = dv.getUint16(this.p); this.p += 2; return this.map(v);     // map16
      case 0xdf: v = dv.getUint32(this.p); this.p += 4; return this.map(v);     // map32
    }
    throw new Error("msgpack.js: неподдерживаемый байт 0x" + b.toString(16));
  };

  /** Декодировать один msgpack-документ из Uint8Array/ArrayBuffer. */
  function decode(buf) {
    return new Decoder(buf).decode();
  }

  root.MPack = { decode: decode };
})(typeof window !== "undefined" ? window : globalThis);
