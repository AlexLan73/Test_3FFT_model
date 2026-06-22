import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Nr = 256
n = np.arange(Nr)
w = np.hanning(Nr)

# дерампнутый дальностный профиль: передний фронт (истинная цель) + гребёнка DRFM позади
ranges = [40.5, 47.7, 55.2, 62.6, 70.1]            # дробные бины -> straddle на сетке 256
amps   = [1.0,  0.7,  0.6,  0.5,  0.4]
sig = sum(a*np.exp(1j*2*np.pi*(r/Nr)*n) for r,a in zip(ranges,amps))*w
sig = sig + np.sqrt(1e-4/2)*(np.random.default_rng(3).standard_normal(Nr)
                             + 1j*np.random.default_rng(4).standard_normal(Nr))

R256  = np.fft.fft(sig)
P     = 1024
R1024 = np.fft.fft(sig, P)
ref   = np.abs(R1024).max()
dB256  = 20*np.log10(np.abs(R256)/ref + 1e-9)
dB1024 = 20*np.log10(np.abs(R1024)/ref + 1e-9)
b256   = np.arange(Nr)
b1024  = np.arange(P)/4.0                            # в единицах 256-бинов

fig, ax = plt.subplots(figsize=(11,5.5))
lo, hi = 36, 75
m1 = (b1024>=lo)&(b1024<=hi)
ax.plot(b1024[m1], dB1024[m1], color="#1D9E75", lw=2,
        label="1024 = zero-pad ×4 (интерполяция)")
m2 = (b256>=lo)&(b256<=hi)
ax.stem(b256[m2], dB256[m2], linefmt="C0-", markerfmt="C0o", basefmt=" ",
        label="256 реальных бинов")
ax.axvline(ranges[0], color="r", ls="--", lw=1.4, label="передний фронт = истинная цель")
for r in ranges[1:]:
    ax.axvline(r, color="0.6", ls=":", lw=0.9)
ax.text(ranges[0]+0.3, 1.0, "цель", color="r", fontsize=10)
ax.text(ranges[2]-1.0, -16, "ложные цели DRFM (позади)", color="0.4", fontsize=9)

ax.set_title("Рефайнмент дальности: 256 реальных (разрешение) + пэд ×4 до 1024 (субдискретно)")
ax.set_xlabel("бин дальности (в единицах сетки 256)"); ax.set_ylabel("дБ отн. макс")
ax.set_ylim(-30, 4); ax.set_xlim(lo,hi); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("/home/claude/refine_256_1024.png", dpi=120); plt.close(fig)
print("peak256 leading =", round(dB256[41],2), "dB | peak1024 leading near 40.5")
