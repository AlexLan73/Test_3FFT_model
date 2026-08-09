# Вставить в Claude первым сообщением

Скопируй блок ниже целиком. Файл `rocm-2d-fft-context-for-claude.md` приложи к чату / положи в project.

---

```text
Ты — эксперт по AMD ROCm/HIP, GPU FFT и radar signal processing.
Работай по приложенному CONTEXT (rocm-2d-fft-context-for-claude.md).

Задача:
- Вход с NIC: 512 antenna × 256 range векторов по 5000 time samples
  (смысл: X[ant][range][time], complex).
- На каждый time t нужен 2D FFT по (antenna × range), т.е. 5000 кадров 512×256.
- GPU: AMD RX 9070 и/или MI100.
- Нужен самый быстрый practical pipeline: layout, gather, LDS tiles,
  batched FFT-512 + FFT-256, streams/ring с NIC, без попытки держать
  5000 полных матриц в LDS.

Ограничения из контекста (не нарушать):
1. LDS — только tile/scratch workgroup, не full frame (1 MiB) и не 5000 frames.
2. Не 5000 kernel launch — batch в одном/нескольких launch.
3. Не materialize сырой rectangle в VRAM, если он не нужен после FFT.
4. Coalesced access: threadIdx.x по непрерывной оси; time-major layout предпочтителен.
5. Учитывать peak BW: ~640 GB/s (9070), ~1.23 TB/s (MI100).
6. Предпочитать fused gather+FFT и skip reverse transpose если post это позволяет.

Сейчас сделай по шагам:
A) Кратко подтверди понимание dataflow (5–8 пунктов).
B) Предложи целевую архитектуру pipeline (NIC → GPU → output) с оценкой
   числа VRAM passes и порядка ms на полный куб FP32 complex.
C) Дай скелет HIP: типы, intermediate layout, launch dims kernel A/B,
   LDS tile sizes, что в регистрах vs LDS.
D) Отдельно: вариант на rocFFT (plan, batch, strides) vs custom path —
   когда какой.
E) Список вопросов, без которых нельзя выбрать финальный kernel
   (тип sample, layout NIC, post-FFT, GPUDirect и т.д.).

Отвечай по-русски, конкретно, с числами и кодом. Без воды.
Если чего-то не хватает — спрашивай, не выдумывай hardware limits сверх CONTEXT.
```

---

## Как использовать

1. Новый чат / Project в Claude.
2. Прикрепи `rocm-2d-fft-context-for-claude.md`.
3. Вставь промпт выше.
4. После ответа A–E можно углублять: «напиши полный kernel A», «rocFFT plan», «ring buffer C++», «rocprofv3 checklist».

## Опционально — короткая follow-up после ответа Claude

```text
Ок. Реализуй Kernel A полностью на HIP (device code + host launch):
gather из layout [ant][range][time] ИЛИ уже [t][range][ant]
(сделай #ifdef / template параметр SRC_LAYOUT),
FFT-512 row, write transposed blocked intermediate.
MI100-first (wavefront 64), комментарии где 9070 (warp 32) отличается.
Без rocFFT внутри A — чистый custom. Twiddles: constexpr или таблица.
```
