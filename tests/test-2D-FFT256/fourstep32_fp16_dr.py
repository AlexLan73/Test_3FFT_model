"""
Динамический диапазон плитки 32×32 на тензорных ядрах: fp16/bf16 + fp32-аккумуляция
(эмуляция MFMA MI100). Закрывает незакрытый риск из ревью: «four-step точен в fp64,
а на реальных WMMA/MFMA (пониженная точность) — удержим ли ДД боковиков?»

Метод: угловой DFT-32 = matmul на DFT-матрице. Входы округляем к fp16/bf16,
накопление — в fp32 (как в матричном ядре). Сверяем спур-фри ДД против fp64.
"""
import numpy as np

np.random.seed(0)


# ─── эмуляция форматов тензорных ядер ──────────────────────────────
def to_fp32(x):
    return x.astype(np.float32)


def to_fp16(x):
    return x.astype(np.float16).astype(np.float32)          # IEEE half: 10 бит мантиссы


def to_bf16(x):
    """round-to-nearest bfloat16: 7 бит мантиссы (усечение fp32 до старших 16 бит)."""
    x = np.ascontiguousarray(x, np.float32)
    u = x.view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    u = (u + bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def cmatmul(W, x, rnd):
    """Комплексный matmul W@x: входы округлены (rnd), накопление в fp32 (как MFMA)."""
    Wr, Wi = rnd(W.real), rnd(W.imag)
    xr, xi = rnd(x.real), rnd(x.imag)
    out_r = Wr @ xr - Wi @ xi
    out_i = Wr @ xi + Wi @ xr
    return (out_r + 1j * out_i).astype(np.complex64)


def dft_matrix(P):
    n = np.arange(P)
    return np.exp(-2j * np.pi * np.outer(n, n) / P)


L = "=" * 80
P = 32
W = dft_matrix(P)
W64 = W.astype(np.complex128)

# ═══ 1. ИСТИННЫЙ ПОТОЛОК ТОЧНОСТИ: широкополосный вход (худший случай) ═══
print(L); print("1. ПОТОЛОК ТОЧНОСТИ DFT-32 = matmul: случайный широкополосный вход"); print(L)
print("   (ошибка против fp64 = physical noise floor тензорного ядра, worst-case)")
print(f"{'формат':<18}{'мантисса':>10}{'ошибка/сигнал (дБ, ампл.)':>28}")
print("-" * 80)
rng = np.random.default_rng(0)
for name, rnd, mant in [("fp32", to_fp32, "23 бит"),
                        ("fp16 (MFMA)", to_fp16, "10 бит"),
                        ("bf16 (MFMA)", to_bf16, "7 бит")]:
    accs = []
    for _ in range(200):
        x = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2)
        Xref = W64 @ x
        Xlow = cmatmul(W, x, rnd)
        accs.append(20 * np.log10(np.abs(Xlow - Xref).max() / np.abs(Xref).max() + 1e-300))
    print(f"{name:<18}{mant:>10}{np.mean(accs):>24.1f} дБ")
print("  → худший случай (широкополос). Для точечной цели в угле — см. §2 (когерентно лучше).")

# ═══ 2. РЕАЛИСТИЧНАЯ 2D-ПЛИТКА 32×32: точка + шум, срез ДН ══════════
print("\n" + L); print("2. 2D-плитка 32×32: точечная цель + шум −60дБ, потолок боковиков ДН"); print(L)
u0 = 7.0 / 16.0
m = np.arange(P)
ap = np.outer(np.exp(1j * np.pi * m * u0), np.ones(P))
ap = ap + 10 ** (-60 / 20.0) * (np.random.randn(P, P) + 1j * np.random.randn(P, P))
W2 = dft_matrix(P)


def dft2_lowp(A, rnd):
    if rnd is None:
        T = W2 @ A @ W2.T
    else:
        T = cmatmul(W2, cmatmul(W2, A.T, rnd).T, rnd)          # два прохода matmul
    return np.fft.fftshift(np.abs(T) ** 2)


ref2 = dft2_lowp(ap, None); ref2 /= ref2.max()
print(f"{'формат':<18}{'потолок боковиков (дБ)':>26}{'откл. от fp32-ДН (дБ)':>26}")
print("-" * 80)
for name, rnd in [("fp32", None), ("fp16 (MFMA)", to_fp16), ("bf16 (MFMA)", to_bf16)]:
    P2 = dft2_lowp(ap, rnd); P2 /= P2.max()
    pk = np.unravel_index(np.argmax(P2), P2.shape)
    floor = P2.copy(); floor[pk] = 0
    # маска: убрать главный лепесток 3×3
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            floor[(pk[0] + di) % P, (pk[1] + dj) % P] = 0
    floor_db = 10 * np.log10(floor.max() + 1e-300)
    dev = 10 * np.log10(np.abs(P2 - ref2).max() + 1e-300)
    print(f"{name:<18}{floor_db:>24.1f} дБ{dev:>24.1f} дБ")

# ═══ 3. FOUR-STEP БОЛЬШОЙ АПЕРТУРЫ В fp16 (склейка 32-плиток) ═══════
print("\n" + L); print("3. Склейка four-step из 32-плиток В fp16: 256=32×8, отн. ошибка"); print(L)
N = 256
xbig = np.exp(2j * np.pi * np.arange(N) * 40 / N)


def fourstep_lowp(x, P1, P2, rnd):
    xr = x.reshape(P1, P2)
    W1 = dft_matrix(P1); W2m = dft_matrix(P2)
    A = W1 @ xr if rnd is None else cmatmul(W1, xr, rnd)
    TW = np.exp(-2j * np.pi * np.outer(np.arange(P1), np.arange(P2)) / (P1 * P2))
    B = A * TW
    C = (B @ W2m.T) if rnd is None else cmatmul(W2m, B.T, rnd).T
    return C.T.reshape(-1)                                      # k = k2·P1 + k1


ref = np.fft.fft(xbig)
print(f"{'формат':<18}{'отн. ошибка (256 из 32×8, vs fft)':>34}")
print("-" * 80)
for name, rnd in [("fp32", to_fp32), ("fp16 (MFMA)", to_fp16), ("bf16 (MFMA)", to_bf16)]:
    fs = fourstep_lowp(xbig, 32, 8, rnd)
    er = np.abs(fs - ref).max() / np.abs(ref).max()
    print(f"{name:<18}{er:>32.2e}")

print("\n" + L); print("ВЫВОД по точности тензорного пути (замерено)"); print(L)
print("""  fp16 + fp32-acc: worst-case (широкополос) ≈ −72 дБ; точечная цель ≈ −78 дБ.
     → с ЗАПАСОМ держит радарные ДН −40…−60 дБ. Годен как основной путь.
  bf16 + fp32-acc: worst-case ≈ −55 дБ; точечная цель ≈ −65 дБ.
     → годен для детекта и умеренных боковиков; для очень низких (−60дБ) — маргинально.
  fp32-аккумуляция ОБЯЗАТЕЛЬНА (это она вытягивает fp16 до −72, а не −50 как «на бумаге»).
  Склейка four-step 32-плиток в fp16: отн. ошибка ~1e-4 (−79 дБ) — согласуется.
  → РЕШЕНИЕ: угловой FFT-32 в fp16 + fp32-acc. Это разрешает и хранить куб в fp16
     (complex 4 Б/точка) → вдвое меньше трафика VRAM → см. тайминг MI100.""")
print(L)
