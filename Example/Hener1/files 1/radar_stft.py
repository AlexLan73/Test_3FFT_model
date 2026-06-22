import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

rng = np.random.default_rng(7)

# ---------- параметры записи ----------
N      = 10000          # комплексных отсчётов после гетеродина
WIN    = 64             # длина окна FFT
HOP    = 32             # шаг
NB     = WIN            # бины
fsweep = 0.4            # свип ЛЧМ: -0.4 .. +0.4 (норм. к fs)

# ---------- генератор одного ЛЧМ-импульса ----------
def chirp(start, length, amp=1.0, f0=-fsweep, f1=+fsweep, ph=0.0):
    """комплексный ЛЧМ, вставленный в массив длиной N начиная со start"""
    s = np.zeros(N, dtype=complex)
    k = np.arange(length)
    mu = (f1 - f0) / length                       # наклон по частоте
    f_inst = f0 + mu * k                           # мгн. норм. частота
    phase = 2*np.pi*(f0*k + 0.5*mu*k**2) + ph
    pulse = amp * np.exp(1j*phase)
    end = min(start+length, N)
    s[start:end] = pulse[:end-start]
    return s

def cnoise(power):
    return np.sqrt(power/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))

# ---------- 1. одна цель ----------
LC = 6000
sig_target = chirp(2000, LC, amp=1.0) + cnoise(0.01)

# ---------- 2. заградительная помеха ----------
# слабая цель + мощный широкополосный шум на всю полосу
barrage = cnoise(0.8)
sig_barrage = chirp(2000, LC, amp=0.7) + barrage

# ---------- 3. помеха-гребёнка (DRFM, ложные цели) ----------
sig_comb = chirp(2000, LC, amp=1.0)               # истинная цель
offsets  = [1200, 2400, 3600, 4800, 6000]         # ложные копии по дальности/времени
amps     = [0.9, 0.8, 0.7, 0.6, 0.5]
for off, a in zip(offsets, amps):
    sig_comb = sig_comb + chirp(2000+off, LC, amp=a,
                                ph=rng.uniform(0, 2*np.pi))
sig_comb = sig_comb + cnoise(0.01)

scenarios = [
    ("Одна цель",            sig_target),
    ("Заградительная помеха", sig_barrage),
    ("Помеха-гребёнка (DRFM)", sig_comb),
]

# ---------- STFT: окно 64, шаг 32, fftshift ----------
def stft(sig):
    w = np.hanning(WIN)
    starts = np.arange(0, N-WIN+1, HOP)
    S = np.empty((len(starts), NB))
    for i, st in enumerate(starts):
        X = np.fft.fftshift(np.fft.fft(sig[st:st+WIN]*w))
        S[i] = np.abs(X)
    return starts, S

bins = np.arange(-NB//2, NB//2)   # -32 .. +31  (отрицательные и положительные)

# ================= 3D объёмы (линейная амплитуда) =================
fig = plt.figure(figsize=(18, 6))
for idx, (title, sig) in enumerate(scenarios, 1):
    starts, S = stft(sig)
    Xf = starts                       # ось x: кадр/выборка по времени
    Yf = bins                         # ось y: бины +/-
    Xg, Yg = np.meshgrid(Xf, Yf, indexing="ij")
    ax = fig.add_subplot(1, 3, idx, projection="3d")
    surf = ax.plot_surface(Xg, Yg, S, cmap=cm.viridis,
                           rstride=2, cstride=1, linewidth=0,
                           antialiased=True)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("выборка (кадр STFT)")
    ax.set_ylabel("бин FFT (±)")
    ax.set_zlabel("|гармоника|")
    ax.view_init(elev=40, azim=-60)
fig.suptitle("ЛЧМ после гетеродина — STFT64/шаг32, объём |X|", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("/home/claude/vol3d_linear.png", dpi=120)
plt.close(fig)

# ================= 3D объёмы (dB) =================
fig = plt.figure(figsize=(18, 6))
for idx, (title, sig) in enumerate(scenarios, 1):
    starts, S = stft(sig)
    SdB = 20*np.log10(S + 1e-3)
    SdB -= SdB.max()
    Xg, Yg = np.meshgrid(starts, bins, indexing="ij")
    ax = fig.add_subplot(1, 3, idx, projection="3d")
    ax.plot_surface(Xg, Yg, SdB, cmap=cm.turbo,
                    rstride=2, cstride=1, linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("выборка (кадр STFT)")
    ax.set_ylabel("бин FFT (±)")
    ax.set_zlabel("дБ отн. макс")
    ax.set_zlim(-60, 0)
    ax.view_init(elev=40, azim=-60)
fig.suptitle("ЛЧМ после гетеродина — STFT64/шаг32, объём в дБ", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("/home/claude/vol3d_db.png", dpi=120)
plt.close(fig)

# ================= 2D спектрограммы (вид сверху) =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, sig) in zip(axes, scenarios):
    starts, S = stft(sig)
    SdB = 20*np.log10(S + 1e-3); SdB -= SdB.max()
    im = ax.imshow(SdB.T, origin="lower", aspect="auto", cmap="turbo",
                   extent=[starts[0], starts[-1], bins[0], bins[-1]],
                   vmin=-60, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("выборка (кадр STFT)")
    ax.set_ylabel("бин FFT (±)")
    fig.colorbar(im, ax=ax, label="дБ")
fig.suptitle("Спектрограммы (вид сверху), дБ отн. макс", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/home/claude/spec2d.png", dpi=120)
plt.close(fig)

print("frames:", len(np.arange(0, N-WIN+1, HOP)), "| bins:", NB)
print("OK")
