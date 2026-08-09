# Контекст: ROCm 2D FFT pipeline (RX 9070 / MI100)

Передать в Claude целиком. Это выжимка обсуждения: задача, ограничения, рассуждения, решения.

---

## 1. Кто я и стек

- Senior Software Engineer / Signal Processing / Radar
- GPU: AMD Radeon RX 9070 + AMD Instinct MI100
- Стек: ROCm / HIP / OpenCL, C++, FFT, beamforming, LCM, Doppler
- Linux (Debian/Ubuntu/Astra), remote servers, multi-thread network acquisition
- Данные radar-потока; нужна максимальная скорость, не «красивый» generic код

---

## 2. Исходная формулировка задачи (эволюция)

### Старт
Оценка времени копирования `2×float16` из VRAM в LDS «одной пары» и массива `512×256×5000` complex float на RX 9070 и MI100.

### Уточнение 1
Нужно нарезать данные из разных векторов в один прямоугольник **5000 раз**. Вопрос: можно ли ускорить через локальную ссылку / LDS.

### Уточнение 2
Нужно сформировать **5000 прямоугольников в LDS** для 2D FFT «на один такт». Нужен самый быстрый вариант.

### Финальная физика данных (ключевое)
От сетевой карты приходит:

```text
512 × 256 векторов по 5000 точек
```

То есть layout по смыслу:

```text
X[antenna=512][range=256][time=5000]
```

Для каждого временного отсчёта `t` нужен прямоугольник **512×256 complex** и над ним **2D FFT** (antenna × range).

Итого samples:

```text
512 × 256 × 5000 = 655_360_000 complex
```

При `complex float` (8 байт):

```text
≈ 5.243 GB (≈ 4.88 GiB) на один полный куб
```

---

## 3. Жёсткие аппаратные факты

### Bandwidth (теоретический peak)
| GPU        | Peak VRAM BW |
|-----------|--------------|
| RX 9070   | ~640 GB/s    |
| MI100     | ~1.23 TB/s HBM2 |

### Нижняя граница времени чистого streaming-read 5.243 GB
| GPU      | Absolute min | Реалистично (~70% peak) |
|----------|--------------|-------------------------|
| RX 9070  | ~8.2 ms      | ~11–12 ms               |
| MI100    | ~4.3 ms      | ~6 ms                   |

`2×float16` (4 байта/элемент) при том же N: примерно в 2 раза меньше объёма → ~4.1 / ~2.1 ms min.

### LDS — не «общая память GPU»
- LDS = scratchpad **одного workgroup / CU**, десятки KiB, не GiB.
- Один rectangle `256×512 complex float` = **1 MiB** — уже не влезает в LDS целиком.
- 5000 rectangles = **~4.88 GiB LDS** — физически невозможно.
- «Один такт» в смысле одного hardware cycle — невозможно.
- Правильная цель: **один (или 2–3) kernel launch**, внутри — тысячи workgroup параллельно по CU.

---

## 4. Главные выводы / решения

### 4.1. Не держать 5000 матриц в LDS
LDS только для:
- tile transpose (`32×32` complex + padding ≈ 8 KiB);
- scratch butterfly между wavefront-ами;
- double-buffer маленького tile.

Не для полного frame и не для всех 5000 кадров сразу.

### 4.2. Layout — критичен
Плохо (как часто приходит с NIC / векторов):

```text
input[antenna][range][time]   // time last → frame[:, :, t] strided
```

Хорошо для FFT кадра `t`:

```text
frame[time][range][antenna]   // [5000][256][512]
// idx = (t * 256 + range) * 512 + antenna
```

Соседние lanes читают соседние `float2` → coalesced.

### 4.3. Не делать полный materialize rectangle + отдельный FFT
Избегать цепочки:

```text
gather all 5000 rect → write VRAM → read VRAM → FFT
```

Это лишний полный write+read (~10 GiB трафика на FP32 complex).

Лучше fused / staged:

```text
NIC → pinned ring → H2D/GPUDirect
  → Kernel A: gather + FFT-512 по antenna   (batch = 5000×256 = 1_280_000 × len 512)
  → transposed blocked layout в VRAM
  → Kernel B: FFT-256 по range              (batch = 5000×512 = 2_560_000 × len 256)
  → post (magnitude / CFAR / Doppler / beamform) — fused если можно
```

Если следующий этап принимает transposed spectrum — **не делать обратный transpose**.

### 4.4. «Локальная ссылка» не убирает чтение VRAM
Указатель / offset table в LDS или constant memory:
- убирает повторный расчёт индексов;
- **не** убирает чтение самих samples из VRAM.

LDS выгодна только если один sample/tile используется **несколько раз** (несколько строк, несколько ops, overlapping windows).  
Если 1 read → 1 write в dst — это bandwidth-bound gather; LDS + barrier могут замедлить.

### 4.5. Карта индексов
Если gather нерегулярный — таблица один раз на CPU/GPU, жить в VRAM на все 5000:

```cpp
// Предпочтительно единый slab, не float2**
const float2* pool;           // все векторы подряд
uint64_t vectorBase[id];
struct RowDesc { uint32_t vector_id; uint32_t offset; };
// addr = vectorBase[desc.vector_id] + desc.offset + col
```

Для регулярной геометрии — считать индекс из `blockIdx/threadIdx` + stride, без таблицы.

### 4.6. Parallelism / launch
- Не 5000 kernel launch.
- `grid.z` или flat grid по batch.
- `threadIdx.x` по непрерывной оси (antenna / col).
- Block size: 128 / 256 / 512; MI100 wavefront 64; RX 9070 warp 32 — кратность важна.
- Pinned host buffers + 2–3 HIP streams + ring chunks (16–64 time frames), не ждать все 5000.

### 4.7. FFT implementation path
1. **Baseline:** rocFFT batched 1D/2D на уже contiguous layout — измерить потолок.
2. **Custom:** если NIC layout нерегулярный — свой fused gather+row-FFT; column-FFT можно rocFFT или свой.
3. Внутри small FFT (256/512):
   - butterfly в регистрах;
   - wave shuffle / `__builtin_amdgcn_wave_*` внутри wave;
   - LDS только для cross-wave;
   - twiddles compile-time или constant, не копировать на каждый rectangle.

### 4.8. Streaming с NIC
Полный куб 5.24 GB — ring buffer чанками:

```text
ring[chunk][time_in_chunk][range][antenna]
```

Async H2D chunk N while GPU processes chunk N−1.  
Идеально: DMA/GPUDirect сразу в time-major layout.

---

## 5. Целевая архитектура (самый быстрый практичный вариант)

```text
Network card packets
    │
    ▼
Pinned host ring (time-major if possible)
    │  async H2D / GPUDirect, multi-stream
    ▼
GPU VRAM input slab
    │
    ├─ Kernel A (one launch for all frames in chunk)
    │     coalesced read source rows
    │     FFT length 512 (antenna)
    │     write transposed blocked intermediate
    │
    ├─ Kernel B (one launch)
    │     FFT length 256 (range)
    │     optional fused postprocess
    │
    ▼
Output spectra / detections (only final data to VRAM/host)
```

**Не цели:**
- 5000 full matrices in LDS
- 5000 separate launches
- materialize raw rectangle if not needed later
- `float2**` forests of pointers

**Цели:**
- max coalesced BW
- min VRAM passes (ideally: 1 read sources + 1 write intermediate + 1 read + 1 write result, or less if fused)
- occupancy vs LDS/register balance
- measure with rocprofv3: duration, achieved BW, occupancy, VALU, bank conflicts

---

## 6. Числа для ориентира

| Величина | Значение |
|----------|----------|
| Antennas | 512 |
| Range bins | 256 |
| Time samples | 5000 |
| Total complex | 655_360_000 |
| Bytes FP32 complex | 5_242_880_000 (~5.24 GB) |
| One frame 256×512 cfloat | 1 MiB |
| Batch FFT-512 | 1_280_000 |
| Batch FFT-256 | 2_560_000 |
| LDS tile example | 32×32 cfloat +1 pad ≈ 8.25 KiB |
| RX 9070 peak BW | ~640 GB/s |
| MI100 peak BW | ~1.23 TB/s |

Теоретический floor только на read полного куба: **~8 ms (9070) / ~4 ms (MI100)**.  
Реальный end-to-end 2D FFT pipeline будет выше из-за write intermediate, twiddle, butterfly, barriers, H2D.

---

## 7. Открытые вопросы для продолжения в Claude

1. Точный тип sample: `complex float` / `complex half` / int16 IQ?
2. Layout пакетов NIC: `[ant][range][time]` или уже `[time][range][ant]`?
3. Gather регулярный (фиксированные stride) или table-driven из «разных векторов»?
4. Нужен ли сырой rectangle после FFT или только спектр / detections?
5. Sliding window по time или batch из 5000 независимых кадров?
6. Post после 2D FFT: magnitude, CFAR, Doppler FFT по time, MVDR/beamform?
7. Целевой latency: на chunk из N frames или на весь 5000?
8. Есть ли GPUDirect / RDMA с NIC или только host pinned + hipMemcpyAsync?
9. rocFFT уже в проекте? Версия ROCm?
10. Рабочий GPU сейчас: только 9070, только MI100, или оба (разные kernel paths)?

---

## 8. Краткие «запреты» (anti-patterns)

- Пытаться засунуть 256×512×5000 в LDS
- 5000 hipLaunch kernel
- Полный gather-all → full VRAM rectangle → потом FFT без fuse
- `const float2**` без выравнивания / без единого slab
- `threadIdx.x` по antenna-strided оси при чтении
- Частые hipMalloc/free на каждый кадр
- Обратный transpose «на всякий случай»
- Думать, что pointer в LDS = данные уже локальные

---

## 9. Словарь терминов (как использовались)

| Термин | Смысл здесь |
|--------|-------------|
| Прямоугольник | Матрица одного time: 512 ant × 256 range complex |
| Нарезать | Gather samples из source vectors в эту матрицу |
| Один такт | Пользователь хотел «всё сразу»; реально = один launch / один pipeline pass |
| Локальная ссылка | Идея держать map/offset локально; не заменяет VRAM read payload |
| Fused kernel | Gather + FFT (+ post) без materialize промежуточного rectangle |
| Time-major | Layout `[t][range][ant]` — дружественный к batched 2D FFT по кадрам |

---

## 10. Что попросить у Claude дальше (готовые промпты)

**A. Спроектировать HIP kernel A/B**
> На основе CONTEXT.md спроектируй HIP kernel A (gather + FFT-512) и kernel B (FFT-256) для layout [T][R][A], batch T=chunk. MI100 wavefront 64 и RX 9070 warp 32. LDS tiles, launch dims, intermediate layout. Без full-frame LDS.

**B. rocFFT plan**
> Дай rocFFT C++ код batched 2D FFT для 5000 матриц 256×512 complex float, strides/distances, workspace, stream. Сравни с custom path когда input strided по time.

**C. NIC ring + streams**
> Спроектируй pinned ring + multi-stream H2D + dual-buffer pipeline для chunk=32 frames, overlap copy/compute.

**D. Профилирование**
> Чеклист rocprofv3 метрик и целевые числа achieved BW / occupancy для bandwidth-bound gather+FFT на MI100 и RX 9070.

---

*Конец контекста. Можно вставить в Claude как system/project file или первым сообщением.*
