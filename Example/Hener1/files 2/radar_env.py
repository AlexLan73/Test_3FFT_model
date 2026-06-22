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
    k = np.arange(length)
    mu = (f1 - f0) / length
    phase = 2*np.pi*(f0*k + 0.5*mu*k**2) + ph
    end = min(start+length, N)
    s[start:end] = (amp*np.exp(1j*phase))[:end-start]
    return s

def cnoise(power):
    return np.sqrt(power/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))

# ---- гауссова огибающая по оси записи (источник в центре, спад к краям) ----
n = np.arange(N)
sigma_t = N/6
env = np.exp(-((n - N/2)**2)/(2*sigma_t**2))      # 1.0 в центре -> ~0.2 на краях импульса

LC = 6000
thermal = lambda p=0.01: cnoise(p)                # ровный шум приёмника

# 1. цель: огибающая на сигнал, шум ровный
s1 = env*chirp(2000, LC, 1.0) + thermal(0.005)

# 2. заградительная: и ЛЧМ, и шумовая помеха идут от объекта -> огибаются; тепловой ровный
barrage = cnoise(0.8)
s2 = env*(0.7*chirp(2000, LC, 1.0) + barrage) + thermal(0.01)

# 3. гребёнка: истинная цель + ложные копии, всё от объекта -> общая огибающая
comb = chirp(2000, LC, 1.0)
for off, a in zip([1200,2400,3600,4800,6000],[0.9,0.8,0.7,0.6,0.5]):
    comb = comb + chirp(2000+off, LC, a, ph=rng.uniform(0,2*np.pi))
s3 = env*comb + thermal(0.01)

scen = [("Одна цель", s1), ("Заградительная помеха", s2), ("Помеха-гребёнка (DRFM)", s3)]
bins = np.arange(-NB//2, NB//2)

def stft(sig):
    w = np.hanning(WIN)
    starts = np.arange(0, N-WIN+1, HOP)
    S = np.empty((len(starts), NB))
    for i, st in enumerate(starts):
        S[i] = np.abs(np.fft.fftshift(np.fft.fft(sig[st:st+WIN]*w)))
    return starts, S

# ===== радиальный купол (вариант B): множитель в плоскости кадр×бин =====
def radial_dome(starts):
    fi = (starts - starts.mean())/(np.ptp(starts)/2)        # -1..1
    bi = bins/(np.ptp(bins)/2)                                # -1..1
    Fi, Bi = np.meshgrid(fi, bi, indexing="ij")
    return np.exp(-(Fi**2 + Bi**2)/(2*0.45**2))

def surf3d(fname, suptitle, db=False, dome=False):
    fig = plt.figure(figsize=(18,6))
    for idx,(title,sig) in enumerate(scen,1):
        starts,S = stft(sig)
        if dome: S = S*radial_dome(starts)
        Z = S.copy()
        if db:
            Z = 20*np.log10(Z+1e-3); Z -= Z.max()
        Xg,Yg = np.meshgrid(starts, bins, indexing="ij")
        ax = fig.add_subplot(1,3,idx,projection="3d")
        ax.plot_surface(Xg,Yg,Z,cmap=(cm.turbo if db else cm.viridis),
                        rstride=2,cstride=1,linewidth=0,antialiased=True)
        ax.set_title(title,fontsize=12,pad=10)
        ax.set_xlabel("выборка (кадр)"); ax.set_ylabel("бин FFT (±)")
        ax.set_zlabel("дБ отн.макс" if db else "|гармоника|")
        if db: ax.set_zlim(-60,0)
        ax.view_init(elev=38,azim=-60)
    fig.suptitle(suptitle,fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f"/home/claude/{fname}",dpi=120); plt.close(fig)

# A: огибающая по времени
surf3d("env_linear.png", "Вариант A — огибающая по времени, |X|", db=False, dome=False)
surf3d("env_db.png",     "Вариант A — огибающая по времени, дБ",  db=True,  dome=False)
# B: радиальный купол поверх A
surf3d("dome_db.png",    "Вариант B — радиальный купол (время×частота), дБ", db=True, dome=True)

# 2D спектрограммы для варианта A
fig,axes = plt.subplots(1,3,figsize=(18,5))
for ax,(title,sig) in zip(axes,scen):
    starts,S = stft(sig)
    SdB = 20*np.log10(S+1e-3); SdB-=SdB.max()
    im = ax.imshow(SdB.T,origin="lower",aspect="auto",cmap="turbo",
                   extent=[starts[0],starts[-1],bins[0],bins[-1]],vmin=-60,vmax=0)
    ax.set_title(title); ax.set_xlabel("выборка (кадр)"); ax.set_ylabel("бин FFT (±)")
    fig.colorbar(im,ax=ax,label="дБ")
fig.suptitle("Вариант A — спектрограммы (вид сверху)",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("/home/claude/env_spec2d.png",dpi=120); plt.close(fig)

print("done")
