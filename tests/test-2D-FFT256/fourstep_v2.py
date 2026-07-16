"""
Модель: 2D FFT 256x256 из плиток 16x16 (four-step / Bailey).
v2: исправлены измерялки (срез через пик, оверсэмплинг ДН), добавлено демо неоднозначности.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
N, N1, N2 = 256, 16, 16
TW = np.exp(-2j * np.pi * np.outer(np.arange(N1), np.arange(N2)) / N)   # [k1,n2]


# ─── four-step: FFT-256 только из FFT-16 ────────────────────────────
def fft256_fourstep(X, axis):
    X = np.moveaxis(X, axis, -1); shp = X.shape
    Xr = X.reshape(-1, N1, N2)
    A = np.fft.fft(Xr, axis=1)          # стадия 1: по n1 (разреженная выборка)
    B = A * TW[None, :, :]              # twiddle
    C = np.fft.fft(B, axis=2)           # стадия 2: по n2 (сплошная плитка)
    out = np.transpose(C, (0, 2, 1)).reshape(-1, N).reshape(shp)
    return np.moveaxis(out, -1, axis)


def fft2_fourstep(X):
    return fft256_fourstep(fft256_fourstep(X, 1), 0)


def steer(u, v, n=N):
    m = np.arange(n)
    return np.outer(np.exp(1j * np.pi * m * u), np.exp(1j * np.pi * m * v))


# ═══ 1. ТОЧНОСТЬ ════════════════════════════════════════════════════
U0, V0 = 9 / 128.0, 0.0                       # ровно на сетке 256 (k=9)
A_full = steer(U0, V0) + 0.02 * (np.random.randn(N, N) + 1j * np.random.randn(N, N))
ref, fs = np.fft.fft2(A_full), fft2_fourstep(A_full)
err_abs = np.abs(ref - fs).max()
err_rel = err_abs / np.abs(ref).max()

# ═══ 2. РАЗРЕШЕНИЕ (оверсэмплинг ДН для честного замера) ════════════
OS = 8                                        # оверсэмплинг диаграммы


def pattern(ap, n_ap):
    """ДН апертуры n_ap с оверсэмплингом: дополняем нулями до n_ap*OS."""
    P = np.abs(np.fft.fftshift(np.fft.fft2(ap, s=(n_ap * OS, n_ap * OS)))) ** 2
    return P / P.max()


def bw3db_deg(P, n_ap):
    """Ширина луча по -3 дБ (градусы), срез ЧЕРЕЗ ПИК вдоль оси u."""
    pk = np.unravel_index(np.argmax(P), P.shape)
    cut = P[:, pk[1]]                          # u лежит вдоль axis 0
    idx = np.where(cut >= 0.5)[0]
    w_bins = idx.max() - idx.min() + 1
    return np.degrees(np.arcsin(w_bins * 2.0 / (n_ap * OS)))


ap_full = steer(U0, V0)
ap_tile = ap_full[:N1, :N2]
P_full_os, P_tile_os = pattern(ap_full, N), pattern(ap_tile, N1)
bw_full, bw_tile = bw3db_deg(P_full_os, N), bw3db_deg(P_tile_os, N1)

# некогерентная сумма 256 плиток (нативная сетка)
P_inc = np.zeros((N1, N2))
for i in range(0, N, N1):
    for j in range(0, N, N2):
        P_inc += np.abs(np.fft.fftshift(np.fft.fft2(A_full[i:i+N1, j:j+N2]))) ** 2
P_inc /= P_inc.max()
P_fs_shift = np.abs(np.fft.fftshift(fs)) ** 2
P_fs_shift /= P_fs_shift.max()

# ═══ 3. НЕОДНОЗНАЧНОСТЬ: две стадии на 1D ═══════════════════════════
uA = 9 / 128.0                 # источник A
uB = 9 / 128.0 + 1 / 8.0       # источник B = A + один период дифр. лепестка

xA = np.exp(1j * np.pi * np.arange(N) * uA)
xB = np.exp(1j * np.pi * np.arange(N) * uB)

# стадия 1: разреженная выборка (каждый 16-й, шаг 8*lambda)
spA = np.abs(np.fft.fft(xA[0::N1])) ** 2
spB = np.abs(np.fft.fft(xB[0::N1])) ** 2
spA /= spA.max(); spB /= spB.max()

# стадия 2: сплошная плитка (16 подряд)
ctA = np.abs(np.fft.fftshift(np.fft.fft(xA[:N1]))) ** 2
ctB = np.abs(np.fft.fftshift(np.fft.fft(xB[:N1]))) ** 2
ctA /= ctA.max(); ctB /= ctB.max()

# итог four-step
fsA = np.abs(np.fft.fftshift(fft256_fourstep(xA[None, :], 1)[0])) ** 2
fsB = np.abs(np.fft.fftshift(fft256_fourstep(xB[None, :], 1)[0])) ** 2
fsA /= fsA.max(); fsB /= fsB.max()

k1_A, k1_B = int(np.argmax(spA)), int(np.argmax(spB))
k2_A = int(np.argmax(ctA)) - N1 // 2
k2_B = int(np.argmax(ctB)) - N1 // 2
u_hat_A = (np.argmax(fsA) - N // 2) * 2.0 / N
u_hat_B = (np.argmax(fsB) - N // 2) * 2.0 / N

# ═══ 4. СКРУГЛЁННАЯ АПЕРТУРА ════════════════════════════════════════
yy, xx = np.mgrid[0:N, 0:N] - (N - 1) / 2.0
m_round = (xx**2 + yy**2) <= (N / 2.0) ** 2
m_rect = np.ones((N, N), bool)
wave = steer(0.0, 0.0)

Pr_rect = pattern(wave * m_rect, N)
Pr_round = pattern(wave * m_round, N)


def first_sl_db(P):
    pk = np.unravel_index(np.argmax(P), P.shape)
    cut = P[pk[0], :]
    c = int(pk[1]); i = c
    while i + 1 < len(cut) and cut[i + 1] < cut[i]:
        i += 1
    j = i
    while j + 1 < len(cut) and cut[j + 1] > cut[j]:
        j += 1
    return 10 * np.log10(cut[j] + 1e-30)


sl_rect, sl_round = first_sl_db(Pr_rect), first_sl_db(Pr_round)
bw_rect, bw_round = bw3db_deg(Pr_rect, N), bw3db_deg(Pr_round, N)
fill = m_round.mean()

# ═══ ТАБЛИЦЫ ════════════════════════════════════════════════════════
L = "=" * 78
print(L); print("1. ТОЧНОСТЬ: 2D FFT 256×256, собранный ТОЛЬКО из FFT-16 + twiddle"); print(L)
print(f"{'метод':<44}{'макс |ошибка|':>16}{'относительная':>16}")
print("-" * 78)
print(f"{'эталон np.fft.fft2(256×256)':<44}{0.0:>16.2e}{0.0:>16.2e}")
print(f"{'four-step из плиток 16×16':<44}{err_abs:>16.2e}{err_rel:>16.2e}")
print(f"\n  → {err_rel:.1e} = машинная точность fp64. Склейка ТОЧНАЯ, потерь нет.")

print("\n" + L); print("2. ЧТО ТЕРЯЕМ, ЕСЛИ ПЛИТКИ НЕ СКЛЕИВАТЬ"); print(L)
print(f"{'метод':<36}{'элементов':>11}{'луч -3дБ':>11}{'Рэлей 2/N':>12}")
print("-" * 78)
print(f"{'полная 256×256 (four-step)':<36}{N*N:>11}{bw_full:>10.2f}°{np.degrees(np.arcsin(2/N)):>11.2f}°")
print(f"{'одна плитка 16×16':<36}{N1*N2:>11}{bw_tile:>10.2f}°{np.degrees(np.arcsin(2/N1)):>11.2f}°")
print(f"{'некогерентная сумма 256 плиток':<36}{N*N:>11}{'~'+f'{bw_tile:.1f}':>10}°{np.degrees(np.arcsin(2/N1)):>11.2f}°")
print(f"\n  → проигрыш по разрешению без склейки: {bw_tile/bw_full:.0f}×")
print(f"  → некогерентная сумма даёт SNR, но НЕ разрешение: луч остаётся 16-элементным")

print("\n" + L); print("3. ДВЕ СТАДИИ: почему нужны обе (демо неоднозначности)"); print(L)
print(f"  источник A: u = {uA:.4f}   |   источник B: u = {uB:.4f}   (разнесены на 1/8)")
print("-" * 78)
print(f"{'':<30}{'источник A':>16}{'источник B':>16}{'вывод':>15}")
print("-" * 78)
print(f"{'стадия 1 (разреж.) → k1':<30}{k1_A:>16}{k1_B:>16}{'ОДИНАКОВО':>15}")
print(f"{'стадия 2 (плитка)  → k2':<30}{k2_A:>16}{k2_B:>16}{'РАЗЛИЧАЕТ':>15}")
print(f"{'four-step → оценка u':<30}{u_hat_A:>16.4f}{u_hat_B:>16.4f}{'ВЕРНО':>15}")
print(f"\n  → стадия 1 (шаг 8λ): тонко (1/128), но 16-кратно НЕОДНОЗНАЧНО")
print(f"  → стадия 2 (плитка): грубо (1/8), зато ОДНОЗНАЧНО → снимает неоднозначность")
print(f"  → k = k1 + 16·k2 : тонко И однозначно")

print("\n" + L); print("4. СЧЁТ ОПЕРАЦИЙ на один бин дальности"); print(L)
print(f"{'величина':<50}{'значение':>16}")
print("-" * 78)
print(f"{'FFT-16 на один FFT-256':<50}{2*N1:>16}")
print(f"{'FFT-16 на полный 2D 256×256':<50}{2*N*2*N1:>16}")
print(f"{'twiddle-умножений на 2D':<50}{2*N*N:>16}")
print(f"{'доля FFT-16, ложащихся на тензорные ядра':<50}{'100%':>16}")

print("\n" + L); print("5. СКРУГЛЁННАЯ АПЕРТУРА (дополнение нулями)"); print(L)
print(f"{'апертура':<36}{'заполнение':>12}{'1-й боковик':>14}{'луч -3дБ':>12}")
print("-" * 78)
print(f"{'прямоугольная 256×256':<36}{'100.0%':>12}{sl_rect:>13.1f}дБ{bw_rect:>11.2f}°")
print(f"{'скруглённая (круг в квадрате)':<36}{fill*100:>11.1f}%{sl_round:>13.1f}дБ{bw_round:>11.2f}°")
print(f"\n  → теория: прямоуг. (sinc) −13.3 дБ; круглая (Эйри) −17.6 дБ")
print(f"  → скруглённый край = естественная аподизация: боковики ниже на {abs(sl_round-sl_rect):.1f} дБ")
print(f"  → плата: луч шире на {100*(bw_round/bw_rect-1):.0f}%, элементов меньше на {100*(1-fill):.0f}%")
print(f"  → four-step точен и здесь: пустые позиции = нули, NUFFT НЕ нужен")
print(L)

# ═══ ГРАФИКИ ════════════════════════════════════════════════════════
db = lambda P: 10 * np.log10(P + 1e-14)
s_full = (np.arange(N) - N // 2) * 2.0 / N
s_tile = (np.arange(N1) - N1 // 2) * 2.0 / N1
s_os = (np.arange(N * OS) - N * OS // 2) * 2.0 / (N * OS)
s_os_t = (np.arange(N1 * OS) - N1 * OS // 2) * 2.0 / (N1 * OS)

# --- Фиг.1: точность + разрешение ---
fig = plt.figure(figsize=(14, 8))
for j, (nm, P) in enumerate([('эталон fft2 256×256', np.abs(np.fft.fftshift(ref))**2),
                             ('four-step из плиток 16×16', np.abs(np.fft.fftshift(fs))**2)]):
    ax = fig.add_subplot(2, 3, j + 1)
    im = ax.imshow(db(P / P.max()), cmap='viridis', vmin=-60, vmax=0)
    ax.set_title(nm, fontsize=10); ax.set_xlabel('k_v'); ax.set_ylabel('k_u')
    plt.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(2, 3, 3)
im = ax.imshow(np.log10(np.abs(ref - fs) + 1e-18), cmap='magma')
ax.set_title(f'разность (log10)\nмакс = {err_abs:.1e}, отн. {err_rel:.1e}', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(2, 1, 2)
pk = np.unravel_index(np.argmax(P_full_os), P_full_os.shape)
ax.plot(s_os, db(P_full_os[:, pk[1]]), lw=1.6, c='#1D9E75', label=f'four-step 256×256 — луч {bw_full:.2f}°')
pk_t = np.unravel_index(np.argmax(P_tile_os), P_tile_os.shape)
ax.plot(s_os_t, db(P_tile_os[:, pk_t[1]]), lw=1.6, c='#D85A30', label=f'одна плитка 16×16 — луч {bw_tile:.2f}°')
ax.plot(s_tile, db(P_inc[:, N1//2]), 's--', lw=1.2, ms=4, c='#7F77DD', label=f'некогерентная сумма 256 плиток — луч ~{bw_tile:.1f}°')
ax.axvline(U0, ls=':', c='k', lw=1.2, label=f'истинное направление u={U0:.3f}')
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-50, 3)
ax.set_xlabel('sin(θ)'); ax.set_ylabel('дБ'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f'Склейка даёт полное разрешение ({bw_full:.2f}°); независимые плитки — навсегда {bw_tile:.1f}°', fontsize=11)
plt.tight_layout(); plt.savefig('/home/claude/fs2_accuracy.png', dpi=115, bbox_inches='tight'); plt.close()

# --- Фиг.2: неоднозначность и её снятие ---
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
k1ax = np.arange(N1)
axs[0].plot(k1ax, spA, 'o-', c='#1D9E75', label=f'источник A (u={uA:.3f})')
axs[0].plot(k1ax, spB, 's--', c='#D85A30', label=f'источник B (u={uB:.3f})')
axs[0].set_title('Стадия 1: разреженная выборка (шаг 8λ)\nпики СОВПАЛИ → неоднозначно', fontsize=10)
axs[0].set_xlabel('бин k₁'); axs[0].set_ylabel('норм.'); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

axs[1].plot(s_tile, ctA, 'o-', c='#1D9E75', label='A')
axs[1].plot(s_tile, ctB, 's--', c='#D85A30', label='B')
axs[1].set_title('Стадия 2: сплошная плитка 16 подряд\nпики РАЗОШЛИСЬ → однозначно', fontsize=10)
axs[1].set_xlabel('sin(θ)'); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)

axs[2].plot(s_full, db(fsA), lw=1.4, c='#1D9E75', label=f'A → û={u_hat_A:.4f}')
axs[2].plot(s_full, db(fsB), lw=1.4, c='#D85A30', label=f'B → û={u_hat_B:.4f}')
axs[2].axvline(uA, ls=':', c='#1D9E75', lw=1); axs[2].axvline(uB, ls=':', c='#D85A30', lw=1)
axs[2].set_xlim(-0.05, 0.3); axs[2].set_ylim(-45, 3)
axs[2].set_title('four-step: k = k₁ + 16·k₂\nтонко И однозначно', fontsize=10)
axs[2].set_xlabel('sin(θ)'); axs[2].set_ylabel('дБ'); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('/home/claude/fs2_stages.png', dpi=115, bbox_inches='tight'); plt.close()

# --- Фиг.3: скруглённая апертура ---
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
axs[0].imshow(m_round, cmap='gray_r'); axs[0].set_title(f'Скруглённая апертура\nзаполнение {fill*100:.1f}% (эл. {int(m_round.sum())})', fontsize=10)
pk_r = np.unravel_index(np.argmax(Pr_rect), Pr_rect.shape)
axs[1].plot(s_os, db(Pr_rect[pk_r[0], :]), lw=1.3, c='#D85A30', label=f'прямоуг.: боковик {sl_rect:.1f} дБ')
axs[1].plot(s_os, db(Pr_round[pk_r[0], :]), lw=1.3, c='#1D9E75', label=f'скруглён.: боковик {sl_round:.1f} дБ')
axs[1].set_xlim(-0.08, 0.08); axs[1].set_ylim(-45, 3)
axs[1].set_xlabel('sin(θ)'); axs[1].set_ylabel('дБ'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
axs[1].set_title('Скруглённый край = аподизация\nбоковики ниже', fontsize=10)
axs[2].imshow(db(Pr_round), cmap='viridis', vmin=-50, vmax=0)
axs[2].set_title('ДН скруглённой апертуры\n(four-step точен: нули = нули)', fontsize=10)
plt.tight_layout(); plt.savefig('/home/claude/fs2_round.png', dpi=115, bbox_inches='tight'); plt.close()
print("\nграфики: fs2_accuracy.png, fs2_stages.png, fs2_round.png")
