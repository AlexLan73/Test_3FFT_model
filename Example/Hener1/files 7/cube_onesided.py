import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
AX = AY = 16; Lr = 16; Lz = 64
k = np.arange(Lr)

def tone(bz, amp=1.0, ph=0.0):                 # bz: ПОЛОЖИТЕЛЬНЫЙ бин дальности (задержка>=0)
    return amp*np.exp(1j*(2*np.pi*(bz/Lz)*k + ph))

lead = 4                                        # передний фронт = истинная цель (ближайший)
target_ft  = tone(lead, 1.0)
comb_ft    = sum(tone(b, a, rng.uniform(0,2*np.pi))               # цель + копии ТОЛЬКО позади
                 for b,a in zip([4,10,16,22,28],[1.0,0.8,0.7,0.6,0.5]))
barrage_ft = (rng.standard_normal(Lr)+1j*rng.standard_normal(Lr))  # широкополосный шум

def cube(ft, jpow=1.0, thermal=0.02):
    coh = np.sqrt(jpow)*np.broadcast_to(ft,(AX,AY,Lr)).astype(complex)
    th  = np.sqrt(thermal/2)*(rng.standard_normal((AX,AY,Lr))+1j*rng.standard_normal((AX,AY,Lr)))
    blk = coh + th
    wx,wy,wf = np.hanning(AX),np.hanning(AY),np.hanning(Lr)
    blk = blk*wx[:,None,None]*wy[None,:,None]*wf[None,None,:]
    C = np.fft.fftn(blk, s=(AX,AY,Lz))
    C = np.fft.fftshift(C, axes=(0,1))          # ЦЕНТРИРУЕМ только угол; дальность НЕ трогаем
    m = 20*np.log10(np.abs(C)+1e-9); m -= m.max()
    return m

ZMAX = 34
scen = [("Цель: точка на дальности фронта", cube(target_ft,1.0,0.02)),
        ("Заградительная: заливка всей дальности", cube(barrage_ft,7.0,0.02)),
        ("Гребёнка: зубцы ТОЛЬКО позади фронта", cube(comb_ft,1.0,0.02))]

kx=np.arange(-AX//2,AX//2); ky=np.arange(-AY//2,AY//2); fz=np.arange(0,ZMAX)
KX,KY,FZ=np.meshgrid(kx,ky,fz,indexing="ij")

fig=plt.figure(figsize=(19,6.8))
for i,(title,m) in enumerate(scen,1):
    ms=m[:,:,0:ZMAX]; v=ms.ravel(); sel=v>-20; vv=v[sel]
    ax=fig.add_subplot(1,3,i,projection="3d")
    ax.scatter(KX.ravel()[sel],KY.ravel()[sel],FZ.ravel()[sel],c=vv,s=(vv+20)**2*0.7+5,
               cmap="turbo",vmin=-20,vmax=0,alpha=0.6,edgecolors="none")
    if i!=2:
        ax.plot([-8,7],[0,0],[lead,lead],color="r",lw=1.2,ls="--")   # линия фронта
    ax.set_title(title,fontsize=11.5,pad=8)
    ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)")
    ax.set_zlabel("дальность (задержка ≥ 0)")
    ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_zlim(0,ZMAX); ax.view_init(16,-58)
fig.suptitle("Исправлено: угол центрирован, дальность ОДНОСТОРОННЯЯ (DRFM только отстаёт)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("/home/claude/cube_onesided.png",dpi=120); plt.close(fig)
print("ok")
