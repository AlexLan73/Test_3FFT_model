/*
 * codec.js -- JS-декодер msgpack-схемы транспорта (P6, N2). ЗЕРКАЛО
 * `core/runtime/codec.py` -- одна и та же схема, читаемая на двух языках без
 * pickle/py-специфичных расширений (msgpack -- стандарт, декодер --
 * @msgpack/msgpack с CDN, см. index.html).
 *
 * Схема сообщения (msgpack map):
 *   { topic: string, tact: int, kind: "array"|"value", payload: ... }
 *
 * kind == "array" (каналы cube/squares):
 *   payload = { shape: [int,...], dtype: string, data: Uint8Array }
 *   dtype ∈ {"complex64","complex128","float32","float64","int32","int64","uint8"}
 *   data -- raw little-endian байты, C-порядок (row-major), для complex* --
 *   ПАРЫ (re,im) подряд на элемент (как numpy .tobytes() комплексного массива).
 *
 * kind == "value" (каналы tracks/cmd): payload -- уже примитивы (объект/массив).
 */

/** dtype-имя -> [TypedArray-конструктор, число float-компонент на элемент]. */
const DTYPE_TABLE = {
  complex64: [Float32Array, 2],
  complex128: [Float64Array, 2],
  float32: [Float32Array, 1],
  float64: [Float64Array, 1],
  int32: [Int32Array, 1],
  int64: [BigInt64Array, 1],
  uint8: [Uint8Array, 1],
};

/**
 * Декодирует один msgpack-кадр (Uint8Array/ArrayBuffer) -> {topic, tact, payload}.
 * Для kind=="array" payload -- {shape, dtype, real, imag|null} (для complex*
 * действительная/мнимая части разложены в отдельные TypedArray -- удобно для
 * Three.js/canvas без комплексной арифметики на JS-стороне).
 * Требует глобальный `MessagePack` (см. index.html, CDN @msgpack/msgpack).
 */
function decodeFrame(bytes) {
  const body = MessagePack.decode(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
  const { topic, tact, kind, payload } = body;
  if (kind === "array") {
    const entry = DTYPE_TABLE[payload.dtype];
    if (!entry) throw new Error(`codec.js: неизвестный dtype ${payload.dtype}`);
    const [TypedArrayCtor, nComponents] = entry;
    const raw = payload.data; // Uint8Array (little-endian, как записал numpy)
    const flat = new TypedArrayCtor(raw.buffer, raw.byteOffset, raw.byteLength / TypedArrayCtor.BYTES_PER_ELEMENT);
    let real = flat;
    let imag = null;
    if (nComponents === 2) {
      const n = flat.length / 2;
      real = new TypedArrayCtor(n);
      imag = new TypedArrayCtor(n);
      for (let i = 0; i < n; i++) {
        real[i] = flat[2 * i];
        imag[i] = flat[2 * i + 1];
      }
    }
    return { topic, tact, payload: { shape: payload.shape, dtype: payload.dtype, real, imag } };
  }
  if (kind === "value") {
    return { topic, tact, payload };
  }
  throw new Error(`codec.js: неизвестный kind ${kind}`);
}

/** Кодирует команду панель->сервер (та же схема, что `codec.encode_command` в Python). */
function encodeCommand(cmd, args) {
  return MessagePack.encode({ topic: "cmd", tact: 0, kind: "value", payload: { cmd, args } });
}
