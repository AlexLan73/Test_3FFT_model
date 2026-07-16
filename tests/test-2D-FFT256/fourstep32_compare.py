"""
Проверка гипотезы: нативная плитка 32×32 вместо 16×16 (four-step / Bailey).
Сравнение по: точности склейки, числу стадий/twiddle-проходов (memory-bound!),
нативному разрешению одной плитки, бюджету LDS.

Контекст: RX 9070 (RDNA4) нативна WMMA 16×16; MI100 (CDNA1) нативны MFMA 16×16 И 32×32.
Вопрос: для тракта ЛЧМ (угловой FFT на бин дальности) 32-плитка «лучше»?
"""
import numpy as np

np.random.seed(0)
OS = 8  # оверсэмплинг ДН для честного замера луча


# ─── four-step 1D для произвольной факторизации N = P1·P2 ───────────
def fourstep_1d(x, P1, P2):
    """FFT по последней оси через two-step: N = P1·P2, primitive-FFT P1 и P2."""
    N = P1 * P2
    shp = x.shape
    xr = x.reshape(-1, P1, P2)                                  # n = n1·P2 + n2
    A = np.fft.fft(xr, axis=1)                                  # стадия 1 → k1
    TW = np.exp(-2j * np.pi * np.outer(np.arange(P1), np.arange(P2)) / N)
    B = A * TW[None, :, :]                                      # twiddle
    C = np.fft.fft(B, axis=2)                                   # стадия 2 → k2
    out = np.transpose(C, (0, 2, 1)).reshape(-1, N)            # k = k2·P1 + k1
    return out.reshape(shp)


def fourstep_2d(X, P1, P2):
    return fourstep_1d(fourstep_1d(X, P1, P2).swapaxes(-1, -2), P1, P2).swapaxes(-1, -2)


# ─── факторизация N по нативному примитиву P (сколько стадий/твидлов) ─
def factorize_by(N, P):
    """N → список факторов, жадно вынимая P; остаток — отдельным фактором."""
    facs, r = [], N
    while r % P == 0 and r > 1:
        facs.append(P); r //= P
    if r > 1:
        facs.append(r)
    return facs


def cost_model(N, P):
    """Стадии, twiddle-проходы и суммарная primitive-FFT работа для 1D длины N."""
    facs = factorize_by(N, P)
    stages = len(facs)
    twiddle_passes = stages - 1                      # твидл между каждой парой стадий
    # на стадии i делаем (N/f_i) преобразований размера f_i
    prim_ffts = sum(N // f for f in facs)
    native_hits = sum(1 for f in facs if f == P)     # сколько факторов = нативная плитка
    return dict(facs=facs, stages=stages, twiddle_passes=twiddle_passes,
                prim_ffts=prim_ffts, native=native_hits)


def steer(u, v, n):
    m = np.arange(n)
    return np.outer(np.exp(1j * np.pi * m * u), np.exp(1j * np.pi * m * v))


def bw3db_deg(ap, n_ap):
    """Ширина луча -3дБ (град) апертуры n_ap, оверсэмплинг ×OS, срез через пик."""
    P = np.abs(np.fft.fftshift(np.fft.fft2(ap, s=(n_ap * OS, n_ap * OS)))) ** 2
    P /= P.max()
    pk = np.unravel_index(np.argmax(P), P.shape)
    cut = P[:, pk[1]]
    idx = np.where(cut >= 0.5)[0]
    return np.degrees(np.arcsin((idx.max() - idx.min() + 1) * 2.0 / (n_ap * OS)))


L = "=" * 80

# ═══ 1. ТОЧНОСТЬ: 2D-FFT из плиток 16×16 vs 32×32 ═══════════════════
print(L); print("1. ТОЧНОСТЬ four-step: 2D-FFT собран ТОЛЬКО из плиток P×P + twiddle"); print(L)
print(f"{'апертура':<20}{'плитка':<12}{'факторы':<16}{'макс|ошибка|':>16}{'отн.':>14}")
print("-" * 80)
for N, P in [(256, 16), (1024, 16), (256, 32), (1024, 32)]:
    facs = factorize_by(N, P)
    if len(facs) < 2:
        continue
    P1 = facs[0]; P2 = N // P1
    A = steer(9 / (N / 2.0), 0.0, N) + 0.02 * (np.random.randn(N, N) + 1j * np.random.randn(N, N))
    ref = np.fft.fft2(A)
    fs = fourstep_2d(A, P1, P2)
    ea = np.abs(ref - fs).max(); er = ea / np.abs(ref).max()
    print(f"{f'{N}×{N}':<20}{f'{P}×{P}':<12}{str(facs):<16}{ea:>16.2e}{er:>14.1e}")
print("\n  → four-step точен при ЛЮБОЙ факторизации: 32×32 так же машинно-точен, как 16×16.")

# ═══ 2. СТОИМОСТЬ: стадии и twiddle-проходы (memory-bound!) ═════════
print("\n" + L); print("2. СТОИМОСТЬ сборки апертуры N (задача MEMORY-BOUND → важны ПРОХОДЫ)"); print(L)
print(f"{'апертура N':<14}{'плитка':<10}{'факторы':<18}{'стадий':>9}{'твидл-прох.':>14}{'нативных':>11}")
print("-" * 80)
for N in [256, 512, 1024, 2048, 4096]:
    for P in [16, 32]:
        c = cost_model(N, P)
        tag = "" if c["stages"] == c["native"] + (0 if N % P**c["native"] == 0 and (N // P**c["native"]) == 1 else 0) else ""
        clean = "✓чисто" if all(f == P for f in c["facs"]) else f"+остаток {c['facs'][-1] if c['facs'][-1]!=P else ''}"
        print(f"{N:<14}{f'{P}×{P}':<10}{str(c['facs']):<18}{c['stages']:>9}{c['twiddle_passes']:>14}{c['native']:>8}  {clean}")
    print("-" * 80)
print("  → где N — степень 32 (1024, ...), 32-плитка кроет в 2 стадии, а 16-плитка — в 2.5 (16·16·4).")
print("  → на 2D каждый лишний twiddle-проход = лишнее чтение+запись всего куба по VRAM.")

# ═══ 3. НАТИВНОЕ РАЗРЕШЕНИЕ ОДНОЙ ПЛИТКИ ════════════════════════════
print("\n" + L); print("3. РАЗРЕШЕНИЕ ОДНОЙ плитки (без four-step-склейки), d=λ/2"); print(L)
print(f"{'плитка':<14}{'элементов':>11}{'Δsinθ теор.':>15}{'луч -3дБ (замер)':>20}")
print("-" * 80)
for P in [16, 32, 64]:
    ap = steer(0.0, 0.0, P)
    bw = bw3db_deg(ap, P)
    print(f"{f'{P}×{P}':<14}{P*P:>11}{f'1/{P//2}':>15}{bw:>18.2f}°")
print("\n  → 32×32 нативно даёт вдвое тоньше угол (1/16 vs 1/8), НЕ требуя four-step для этого шага.")

# ═══ 4. БЮДЖЕТ LDS (помещается ли плитка на кристалле) ═════════════
print("\n" + L); print("4. БЮДЖЕТ LDS на одну плитку (complex64 = 8 Б/точка)"); print(L)
print(f"{'плитка':<14}{'точек':>10}{'LDS (complex64)':>18}{'из 64 КБ':>12}")
print("-" * 80)
for P in [16, 32, 64, 128]:
    kb = P * P * 8 / 1024
    print(f"{f'{P}×{P}':<14}{P*P:>10}{kb:>15.1f} КБ{kb/64*100:>10.0f}%")
print("\n  → 32×32 = 8 КБ (12% LDS) — свободно на кристалле. 64×64 = 32 КБ (половина). 128×128 → VRAM.")

# ═══ 5. ВЕРДИКТ ════════════════════════════════════════════════════
print("\n" + L); print("5. ВЕРДИКТ: 32×32 vs 16×16"); print(L)
print("""  ✅ Точность: одинаково машинная (four-step точен при любой факторизации).
  ✅ Стадии: для апертур-степеней-32 (1024, 32768...) 32-плитка = меньше проходов
     по VRAM → прямой выигрыш (задача memory-bound).
  ✅ Разрешение плитки: 32×32 нативно 1/16 — вдвое тоньше, БЕЗ склейки.
  ✅ LDS: 32×32 = 8 КБ, свободно помещается (fused-кернел гл.3.4 работает).
  ⚠️ ЖЕЛЕЗО: выигрыш РЕАЛЕН только там, где 32×32 — НАТИВНАЯ инструкция:
     MI100/CDNA (MFMA 16×16 И 32×32) — да; RX 9070/RDNA4 (WMMA 16×16) — 32 эмулируется
     4-мя блоками 16×16, тогда «нативность» теряется, остаётся лишь выигрыш по стадиям.
  → ИТОГ: на MI100 32×32 ЛУЧШЕ (меньше проходов + вдвое тоньше угол на нативном ядре).
    На RX 9070 — нейтрально/чуть лучше по стадиям, но без матричного бонуса.""")
print(L)
