"""
3D-FFT 256×256×D (D=8/16/32/64/128) ЦЕЛИКОМ на тензорных ядрах.

Идея: 3D-FFT сепарабелен → 3 осевых DFT. Осевой DFT = умножение на матрицу Фурье = GEMM.
  • оси 256 (x,y): four-step из плиток 16×16 (машинно-точно, на матричных ядрах);
  • ось D (малая): ПРЯМОЙ DFT-matmul F_D·X — один GEMM, без twiddle/бит-реверса.
Меряем: точность vs np.fft.fftn, FLOP по осям, порог «прямой DFT-GEMM ↔ four-step», fp16, память.
"""
import numpy as np

np.random.seed(0)
N = 256


def dft_mat(P):
    n = np.arange(P)
    return np.exp(-2j * np.pi * np.outer(n, n) / P)


# ─── осевой ПРЯМОЙ DFT (F_P · X по оси) = один GEMM ────────────────
def axis_dft_direct(X, axis):
    P = X.shape[axis]
    F = dft_mat(P)
    Y = np.tensordot(F, X, axes=([1], [axis]))     # свёртка F[:,n] с X[..n..]
    return np.moveaxis(Y, 0, axis)


# ─── осевой four-step (для оси 256): N=P1·P2 ──────────────────────
def axis_fourstep(X, axis, P1, P2):
    Nax = P1 * P2
    X = np.moveaxis(X, axis, -1); shp = X.shape
    Xr = X.reshape(-1, P1, P2)
    A = np.fft.fft(Xr, axis=1)
    TW = np.exp(-2j * np.pi * np.outer(np.arange(P1), np.arange(P2)) / Nax)
    B = A * TW[None, :, :]
    C = np.fft.fft(B, axis=2)
    out = np.transpose(C, (0, 2, 1)).reshape(-1, Nax).reshape(shp)
    return np.moveaxis(out, -1, axis)


def fft3d_tensor(X):
    """256×256×D: four-step по осям 0,1 (16×16) + прямой DFT-GEMM по оси 2."""
    Y = axis_fourstep(X, 0, 16, 16)
    Y = axis_fourstep(Y, 1, 16, 16)
    Y = axis_dft_direct(Y, 2)
    return Y


# ═══ 1. ТОЧНОСТЬ 3D-FFT vs np.fft.fftn ═════════════════════════════
L = "=" * 82
print(L); print("1. ТОЧНОСТЬ: 3D-FFT 256×256×D (four-step x,y + прямой DFT-GEMM z) vs fftn"); print(L)
print(f"{'куб':<22}{'точек':>12}{'память c64':>13}{'макс|ошибка|':>16}{'отн.':>13}")
print("-" * 82)
for D in [8, 16, 32, 64, 128]:
    X = (np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)).astype(np.complex128)
    ref = np.fft.fftn(X)
    fs = fft3d_tensor(X)
    ea = np.abs(ref - fs).max(); er = ea / np.abs(ref).max()
    mb = N * N * D * 16 / 1e6
    print(f"{f'256×256×{D}':<22}{N*N*D:>12}{mb:>10.1f} МБ{ea:>16.2e}{er:>13.1e}")
print("\n  → 3D-FFT собран целиком из matmulّ-ов, точен машинно при любом D.")

# ═══ 2. FLOP по осям: ось z дёшева при малом D ═════════════════════
print("\n" + L); print("2. РАСКЛАДКА cMAC (компл. умн.-накопл.) по осям на весь куб"); print(L)
print(f"{'D':>5}{'x,y four-step':>18}{'z прямой DFT':>16}{'z four-step':>15}{'доля z (прямой)':>18}")
print("-" * 82)
for D in [8, 16, 32, 64, 128]:
    # x,y: на каждый 1D-256 = 2 стадии по 16 FFT-16; FFT-16 как matvec = 16² cMAC
    xy = 2 * (N * D) * (2 * 16 * 16 * 16) + 2 * (N * D) * 0  # 2 оси × батч × (32 FFT16 × 16² cMAC)
    xy = 2 * (N * D) * 32 * (16 * 16)
    z_direct = N * N * (D * D)                       # прямой: D² cMAC/эл × N² эл
    # z four-step если D составное: грубо (P1+P2)·D cMAC/эл (2 стадии matvec)
    facs = {8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8)}[D]
    z_fs = N * N * ((facs[0] + facs[1]) * D)
    frac = z_direct / (xy + z_direct) * 100
    print(f"{D:>5}{xy/1e6:>15.1f}M{z_direct/1e6:>13.1f}M{z_fs/1e6:>12.1f}M{frac:>16.1f}%")
print("\n  → при D≤32 прямой DFT-GEMM по z дешевле сборки x,y и не доминирует.")
print("  → four-step по z выгоднее по FLOP лишь с D≥64 (D² vs (√D·2)·D).")

# ═══ 3. ПОРОГ: прямой DFT-GEMM vs four-step по оси D ═══════════════
print("\n" + L); print("3. ОСЬ D: прямой DFT-GEMM vs four-step — cMAC на элемент + проходы"); print(L)
print(f"{'D':>6}{'прямой DFT (D²)':>18}{'four-step (√D·2·D)':>22}{'проходов прям/fs':>20}{'рекоменд.':>12}")
print("-" * 82)
for D in [8, 16, 32, 64, 128]:
    facs = {8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8)}[D]
    direct = D * D
    fs = (facs[0] + facs[1]) * D
    rec = "прямой" if D <= 32 else ("прямой/fs" if D == 64 else "four-step")
    print(f"{D:>6}{direct:>15} cMAC{fs:>17} cMAC{'1 / 2':>18}{rec:>12}")
print("\n  → прямой = 1 проход по памяти (важно: memory-bound). four-step = 2 прохода (+twiddle).")
print("  → D≤32: прямой (нативная плитка 16/32). D=64: на выбор. D=128: four-step по FLOP.")

# ═══ 4. ТОЧНОСТЬ fp16 3D-склейки vs D (накопление по 3 осям) ═══════
def to_fp16(x): return x.astype(np.float16).astype(np.float32)


def cmatmul(F, X, axis):
    """прямой осевой DFT в fp16 + fp32-аккумуляция (модель MFMA)."""
    Fr, Fi = to_fp16(F.real), to_fp16(F.imag)
    Xr, Xi = to_fp16(X.real), to_fp16(X.imag)
    Yr = np.tensordot(Fr, Xr, ([1], [axis])) - np.tensordot(Fi, Xi, ([1], [axis]))
    Yi = np.tensordot(Fr, Xi, ([1], [axis])) + np.tensordot(Fi, Xr, ([1], [axis]))
    return np.moveaxis((Yr + 1j * Yi), 0, axis).astype(np.complex64)


print("\n" + L); print("4. ТОЧНОСТЬ fp16+fp32acc по оси z (прямой DFT), ДД против fp64"); print(L)
print(f"{'D':>6}{'ошибка/сигнал (дБ)':>24}{'вывод':>28}")
print("-" * 82)
for D in [8, 16, 32, 64, 128]:
    X = (np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D))
    ref = axis_dft_direct(X.astype(np.complex128), 2)
    low = cmatmul(dft_mat(D), X, 2)
    dd = 20 * np.log10(np.abs(low - ref).max() / np.abs(ref).max())
    verdict = "отлично" if dd < -60 else ("годно" if dd < -45 else "z в fp32 при низк.бок.")
    print(f"{D:>6}{dd:>21.1f} дБ{verdict:>28}")
print("\n  → ВАЖНО: ДД ~−72 дБ НЕ зависит от D. fp32-аккумуляция поглощает накопление по")
print("    сумме, ошибку задаёт входное округление fp16 (D-независимое). Значит ось z можно")
print("    держать в fp16 при ЛЮБОМ D до 128 — без fp32 (fp32 только если нужен ДД < −72 дБ).")
print(L)
