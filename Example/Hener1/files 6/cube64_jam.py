import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
AX = AY = 16; Lr = 16; Lz = 64        # реальные 16 по дальности -> zero-pad до 64
k = np.arange(Lr)

def tone(b, amp=1.0, ph=0.0):
    return amp*np.exp(1j*(2*np.pi*(b/Lr)*k + ph))

target_ft  = tone(2.5, 1.0)                                  # straddled цель
comb_ft    = sum(tone(b, a, rng.uniform(0,2*np.pi))
                 for b,a in zip([-6.4,-3.2,0.3,3.4,6.1],[0.9,0.8,1.0,0.8,0.9]))
barrage_ft = (rng.standard_normal(Lr)+1j*rng.standard_normal(Lr))   # широкополосный шум

def cube(ft, jpow=1.0, thermal=0.02):
    coh = np.sqrt(jpow)*np.broadcast_to(ft,(AX,AY,Lr)).astype(complex)
    th  = np.sqrt(thermal/2)*(rng.standard_normal((AX,AY,Lr))+1j*rng.standard_normal((AX,AY,Lr)))
    blk = coh + th
    wx,wy,wf = np.hanning(AX),np.hanning(AY),np.hanning(Lr)
    blk = blk*wx[:,None,None]*wy[None,:,None]*wf[None,None,:]
    C = np.fft.fftshift(np.fft.fftn(blk, s=(AX,AY,Lz)))      # пэд по дальности
    m = 20*np.log10(np.abs(C)+1e-9); m -= m.max()
    return m

scen = [("Цель -> точка", cube(target_ft,1.0,0.02)),
        ("Заградительная -> столб", cube(barrage_ft,7.0,0.02)),
        ("Гребёнка -> шашлык", cube(comb_ft,1.0,0.02))]

kx=np.arange(-AX//2,AX//2); ky=np.arange(-AY//2,AY//2); fz=np.arange(-Lz//2,Lz//2)
KX,KY,FZ=np.meshgrid(kx,ky,fz,indexing="ij")

fig=plt.figure(figsize=(19,6.5))
for i,(title,m) in enumerate(scen,1):
    v=m.ravel(); sel=v>-20; vv=v[sel]
    ax=fig.add_subplot(1,3,i,projection="3d")
    ax.scatter(KX.ravel()[sel],KY.ravel()[sel],FZ.ravel()[sel],c=vv,s=(vv+20)**2*0.7+5,
               cmap="turbo",vmin=-20,vmax=0,alpha=0.6,edgecolors="none")
    ax.set_title(title,fontsize=12,pad=8)
    ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)"); ax.set_zlabel("бин дальности (×4)")
    ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_zlim(-32,32); ax.view_init(16,-58)
fig.suptitle("16×16×64 (пэд по дальности) — помехи по форме отклика", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("/home/claude/cube64_jam.png",dpi=120); plt.close(fig)
print("ok")
