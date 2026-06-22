import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

rng = np.random.default_rng(7)
N, WIN, HOP, NB = 10000, 64, 32, 64
fsweep = 0.4

def chirp(start, length, amp=1.0, f0=-fsweep, f1=+fsweep, ph=0.0):
    s = np.zeros(N, dtype=complex)
    k = np.arange(length); mu = (f1-f0)/length
    s[start:start+length] = amp*np.exp(1j*(2*np.pi*(f0*k+0.5*mu*k**2)+ph))
    return s

n = np.arange(N); env = np.exp(-((n-N/2)**2)/(2*(N/6)**2))
src = env*chirp(2000, 6000, 1.0)                 # источник: цель, пик энергии в центре записи

# ================= ЧАСТЬ 1: одна антенна, кроп ±10 кадров =================
sig = src + np.sqrt(0.005/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
w = np.hanning(WIN)
starts = np.arange(0, N-WIN+1, HOP)
S = np.array([np.abs(np.fft.fftshift(np.fft.fft(sig[st:st+WIN]*w))) for st in starts])
bins = np.arange(-NB//2, NB//2)

cf = int(np.argmax(S.max(axis=1)))               # кадр центра возникновения
M = 10
fa, fb = cf-M, cf+M+1
Sc = S[fa:fb]; fr = np.arange(fa, fb)
ScdB = 20*np.log10(Sc+1e-3); ScdB -= ScdB.max()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
Xg, Yg = np.meshgrid(fr, bins, indexing="ij")
ax.plot_surface(Xg, Yg, ScdB, cmap=cm.turbo, rstride=1, cstride=1,
                linewidth=0, antialiased=True)
ax.set_title(f"Часть 1 — 1 антенна, центр кадра={cf}, ±{M} шагов FFT")
ax.set_xlabel("кадр STFT"); ax.set_ylabel("бин FFT (±)"); ax.set_zlabel("дБ")
ax.set_zlim(-60,0); ax.view_init(35,-60)
fig.tight_layout(); fig.savefig("/home/claude/crop1d.png", dpi=120); plt.close(fig)

# ================= ЧАСТЬ 2: решётка 16x16, 3D-БПФ 16x16x64 =================
AX, AY = 16, 16
c0 = N//2 - WIN//2                                # окно центрировано на пике источника
snap = src[c0:c0+WIN]                            # boresight: все элементы синфазны
# куб данных: [16,16,64], добавляем независимый тепловой шум на каждый элемент
blk = np.broadcast_to(snap, (AX, AY, WIN)).copy()
blk += np.sqrt(0.01/2)*(rng.standard_normal((AX,AY,WIN))+1j*rng.standard_normal((AX,AY,WIN)))
# весовые окна по всем трём осям (Хэннинг) -> чистый главный лепесток
wx = np.hanning(AX); wy = np.hanning(AY); wf = np.hanning(WIN)
blk *= wx[:,None,None]*wy[None,:,None]*wf[None,None,:]

C = np.fft.fftshift(np.fft.fftn(blk))            # 3D-БПФ
mag = np.abs(C); magdB = 20*np.log10(mag+1e-9); magdB -= magdB.max()

kx = np.arange(-AX//2, AX//2)                    # угловой бин (азимут)
ky = np.arange(-AY//2, AY//2)                    # угловой бин (угол места)
fz = np.arange(-NB//2, NB//2)                    # частотный бин
# кроп частоты ±10 вокруг центра
fmask = np.abs(fz) <= 10
magc = magdB[:,:,fmask]; fzc = fz[fmask]

# объёмный скаттер: рисуем только значимые точки (>-30 дБ), размер/прозрачность ~ амплитуде
KX, KY, FZ = np.meshgrid(kx, ky, fzc, indexing="ij")
v = magc.ravel()
xs, ys, zs = KX.ravel(), KY.ravel(), FZ.ravel()
sel = v > -30
xs, ys, zs, v = xs[sel], ys[sel], zs[sel], v[sel]
size = (v+30)**2.0 * 0.9 + 4
alpha = np.clip((v+30)/30, 0.05, 1)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(xs, ys, zs, c=v, s=size, cmap="turbo", vmin=-30, vmax=0,
               alpha=0.55, edgecolors="none")
ax.set_title("Часть 2 — 3D-БПФ 16×16×64, источник в центре (boresight)")
ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)"); ax.set_zlabel("частотный бин (±10)")
fig.colorbar(p, ax=ax, label="дБ отн.макс", shrink=0.6)
ax.view_init(22, -55)
fig.tight_layout(); fig.savefig("/home/claude/vol3dfft.png", dpi=120); plt.close(fig)

print("cf=", cf, "| cube=", magc.shape, "| points>-30dB:", sel.sum())
