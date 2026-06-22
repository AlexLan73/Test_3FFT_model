import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
L = 255                                  # длина кода (демо; в системе M-seq/Gold)
N = 1024                                  # буфер дальности (чипы)

# агильные коды по импульсам (демо: независимые ±1; в системе — M-послед./Голд)
codes = {j: rng.choice([-1.0,1.0], L) for j in [0,-1,-2,-3]}   # c_n=codes[0], историч. <0
c_now, c_stale = codes[0], codes[-3]      # джаммер отстаёт на m=3 периода

tau0 = 200                                 # дальность истинной цели (передний фронт)
buf = np.zeros(N)
buf[tau0:tau0+L] += 1.0*c_now              # истинная цель: ТЕКУЩИЙ код

comb_delays = [220, 300, 380, 460]         # ложные цели: на дальности цели и ДАЛЕЕ
comb_amps   = [0.8, 0.7, 0.6, 0.5]
for d,a in zip(comb_delays, comb_amps):
    buf[d:d+L] += a*c_stale                 # гребёнка DRFM: УСТАРЕВШИЙ код c_{n-3}
buf += np.sqrt(0.5)*rng.standard_normal(N) # шум

def xcorr(sig, code):
    r = np.correlate(sig, code, mode="valid")    # лаг = задержка
    return r
lags = np.arange(N-L+1)

R_now   = xcorr(buf, c_now)
R_stale = xcorr(buf, c_stale)
ref = L
dB = lambda x: 20*np.log10(np.abs(x)/ref + 1e-6)

fig,(a1,a2) = plt.subplots(2,1,figsize=(12,7.5),sharex=True)

a1.plot(lags, dB(R_now), color="#993C1D", lw=1.0)
a1.axvline(tau0, color="r", ls="--", lw=1.2)
a1.text(tau0+6, -1, "истинная цель\n(передний фронт)", color="r", fontsize=10, va="top")
for d in comb_delays: a1.axvline(d, color="0.7", ls=":", lw=0.8)
a1.set_title("Канал ТЕКУЩЕГО кода c_n  →  только истинная цель, гребёнка подавлена")
a1.set_ylabel("дБ отн. пика"); a1.set_ylim(-40,3); a1.grid(alpha=0.25)

a2.plot(lags, dB(R_stale), color="#534AB7", lw=1.0)
a2.axvline(tau0, color="r", ls="--", lw=1.2)
a2.text(tau0+6, -28, "цель\nподавлена", color="r", fontsize=9, va="top")
for d,a in zip(comb_delays,comb_amps):
    a2.axvline(d, color="0.5", ls=":", lw=0.9)
a2.text(comb_delays[1], -1, "гребёнка DRFM проявилась\n(код c_{n-3}  →  задержка m=3 периода)",
        color="#534AB7", fontsize=10, va="top")
a2.set_title("Канал ИСТОРИЧЕСКОГО кода c_{n-3}  →  всплыла гребёнка, цель подавлена")
a2.set_xlabel("задержка (чипы дальности)"); a2.set_ylabel("дБ отн. пика")
a2.set_ylim(-40,3); a2.grid(alpha=0.25); a2.set_xlim(120,560)
fig.suptitle("FM+m+корреляция: банк кодов разделяет цель и гребёнку по подписи импульса", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("/home/claude/code_bank.png", dpi=120); plt.close(fig)

# числовой контраст
print("пик цели в канале c_n   :", round(dB(R_now)[tau0],1), "дБ")
print("гребёнка в канале c_n   :", round(dB(R_now)[comb_delays[0]],1), "дБ (подавлена)")
print("гребёнка в канале c_n-3 :", round(dB(R_stale)[comb_delays[0]],1), "дБ (всплыла)")
print("теор. подавление 10log10(L) =", round(10*np.log10(L),1), "дБ")
