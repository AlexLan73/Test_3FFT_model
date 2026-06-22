import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

rng = np.random.default_rng(7)
L = 16; AX = AY = 16            # 16x16x16: родной WMMA-фрагмент RDNA4
k = np.arange(L)

# ---- после ДЕРАМПА (stretch): каждая цель/ложная цель -> один тон (бин дальности) ----
def tone(b, amp=1.0, ph=0.0):
    return amp*np.exp(1j*(2*np.pi*(b/L)*k + ph))

target_ft  = tone(0, 1.0)                                   # 1 цель -> 1 бин (центр)
comb_bins  = [-6,-3,0,3,6]
comb_ft    = sum(tone(b, a, rng.uniform(0,2*np.pi))
                 for b,a in zip(comb_bins,[0.9,0.8,1.0,0.8,0.9]))
barrage_ft = (rng.standard_normal(L)+1j*rng.standard_normal(L))  # широкополосный шум по всем бинам

# ---- куб: когерентный сигнал с нормали (boresight, все элементы синфазны) + тепловой шум ----
def cube(ft, jpow=1.0, thermal=0.02):
    coh = np.sqrt(jpow)*np.broadcast_to(ft,(AX,AY,L)).astype(complex)
    th  = np.sqrt(thermal/2)*(rng.standard_normal((AX,AY,L))+1j*rng.standard_normal((AX,AY,L)))
    blk = coh + th
    wx,wy,wf = np.hanning(AX),np.hanning(AY),np.hanning(L)   # окна по трём осям
    blk = blk*wx[:,None,None]*wy[None,:,None]*wf[None,None,:]
    C = np.fft.fftshift(np.fft.fftn(blk))
    m = 20*np.log10(np.abs(C)+1e-9); m -= m.max()
    return m

scen = [
    ("1. Одна цель -> точка",        cube(target_ft, 1.0, 0.02)),
    ("2. Заградительная -> столб",    cube(barrage_ft, 6.0, 0.02)),
    ("3. Гребёнка -> шашлык",         cube(comb_ft,   1.0, 0.02)),
]

kx = np.arange(-AX//2,AX//2); ky = np.arange(-AY//2,AY//2); fz = np.arange(-L//2,L//2)
KX,KY,FZ = np.meshgrid(kx,ky,fz,indexing="ij")

fig = plt.figure(figsize=(19,6.5))
for i,(title,m) in enumerate(scen,1):
    v = m.ravel(); xs,ys,zs = KX.ravel(),KY.ravel(),FZ.ravel()
    sel = v > -22
    vv = v[sel]
    size = (vv+22)**2*0.8 + 6
    ax = fig.add_subplot(1,3,i,projection="3d")
    p = ax.scatter(xs[sel],ys[sel],zs[sel],c=vv,s=size,cmap="turbo",
                   vmin=-22,vmax=0,alpha=0.6,edgecolors="none")
    ax.set_title(title,fontsize=12,pad=8)
    ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)")
    ax.set_zlabel("бин дальности (дерамп)")
    ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_zlim(-8,8)
    ax.view_init(18,-58)
fig.suptitle("3D-БПФ 16×16×16 ПОСЛЕ ДЕРАМПА — источник на нормали (центр)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/home/claude/cube16_deramp.png", dpi=120); plt.close(fig)
print("ok | pts:", [int((m>-22).sum()) for _,m in scen])
