import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 16
k0 = 4
f0 = (k0 + 0.5)/N                 # тон РОВНО между бинами 4 и 5 (худший straddle)
n = np.arange(N)
w = np.hanning(N)
x = np.exp(1j*2*np.pi*f0*n)*w     # реальные 16 отсчётов, окно Ханна

X16 = np.fft.fft(x);      f16 = np.arange(N)/N
P   = 64                          # zero-pad x4
X64 = np.fft.fft(x, P);   f64 = np.arange(P)/P

ref = np.abs(X64).max()           # истинный пик главного лепестка (интерполированный)
dB16 = 20*np.log10(np.abs(X16)/ref + 1e-12)
dB64 = 20*np.log10(np.abs(X64)/ref + 1e-12)

# --- параболическая интерполяция по 3 бинам вокруг максимума (в дБ) ---
km = int(np.argmax(np.abs(X16)))
a, b, c = dB16[km-1], dB16[km], dB16[km+1]
delta = 0.5*(a - c)/(a - 2*b + c)
peak_bin  = km + delta
peak_freq = peak_bin/N
peak_dB   = b - 0.25*(a - c)*delta

scallop = dB16.max()              # лучший 16-бин ниже истинного пика = потеря на скаллопинг
ferr_raw = f16[km] - f0           # ошибка по «сырому» максимуму
ferr_par = peak_freq - f0         # ошибка после параболики

fig, ax = plt.subplots(figsize=(10,5.5))
m = (f64 >= 0.15) & (f64 <= 0.45)
ax.plot(f64[m], dB64[m], color="#1D9E75", lw=2, label="zero-pad ×4 (интерполяция DTFT)")
mm = (f16 >= 0.15) & (f16 <= 0.45)
ax.stem(f16[mm], dB16[mm], linefmt="C0-", markerfmt="C0o", basefmt=" ",
        label="БПФ-16 (реальные бины)")
ax.axvline(f0, color="k", ls="--", lw=1.2, label=f"истинная частота {f0:.4f}")
ax.plot(peak_freq, peak_dB, "r*", ms=16, label="параболика (3 бина)")
ax.axhline(0, color="gray", ls=":", lw=0.8)

ax.annotate(f"скаллопинг-потеря\n{scallop:+.2f} дБ",
            xy=(f16[km], scallop), xytext=(0.165, -4),
            arrowprops=dict(arrowstyle="->", color="C0"), fontsize=10, color="C0")
ax.set_title("Straddle: тон между бинами — zero-pad ×4 и параболика возвращают пик")
ax.set_xlabel("норм. частота"); ax.set_ylabel("дБ отн. истинного пика")
ax.set_ylim(-25, 3); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("/home/claude/straddle1d.png", dpi=120); plt.close(fig)

print(f"raw bin freq err = {ferr_raw:+.5f} | parabolic freq err = {ferr_par:+.5f}")
print(f"scalloping loss  = {scallop:+.2f} dB | parabolic peak = {peak_dB:+.2f} dB")
