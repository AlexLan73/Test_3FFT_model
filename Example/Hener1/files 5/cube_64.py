import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
AX = AY = 16; Lr = 16             # реальные отсчёты: 16x16x16

b = 2.5                           # дальностный тон МЕЖДУ бинами (straddle)
ft = np.exp(1j*2*np.pi*(b/Lr)*np.arange(Lr))     # дерампнутая цель на нормали

def block():
    coh = np.broadcast_to(ft,(AX,AY,Lr)).astype(complex)
    th  = np.sqrt(0.02/2)*(rng.standard_normal((AX,AY,Lr))+1j*rng.standard_normal((AX,AY,Lr)))
    blk = coh + th
    wx,wy,wf = np.hanning(AX),np.hanning(AY),np.hanning(Lr)   # окна на РЕАЛЬНЫЕ 16
    return blk*wx[:,None,None]*wy[None,:,None]*wf[None,None,:]

def cube_fft(Lz):                 # Lz: длина БПФ по дальности (16 = без пэда, 64 = zero-pad x4)
    C = np.fft.fftshift(np.fft.fftn(block(), s=(AX,AY,Lz)))
    m = 20*np.log10(np.abs(C)+1e-9); m -= m.max()
    return m

def scatter(ax, m, Lz, title):
    kx = np.arange(-AX//2,AX//2); ky = np.arange(-AY//2,AY//2); fz = np.arange(-Lz//2,Lz//2)
    KX,KY,FZ = np.meshgrid(kx,ky,fz,indexing="ij")
    v = m.ravel(); sel = v > -20; vv = v[sel]
    p = ax.scatter(KX.ravel()[sel],KY.ravel()[sel],FZ.ravel()[sel],
                   c=vv,s=(vv+20)**2*0.9+6,cmap="turbo",vmin=-20,vmax=0,
                   alpha=0.65,edgecolors="none")
    ax.set_title(title,fontsize=12,pad=8)
    ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)"); ax.set_zlabel("бин дальности")
    ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_zlim(-Lz//2,Lz//2)
    ax.view_init(16,-58)
    return p

fig = plt.figure(figsize=(15,6.5))
ax1 = fig.add_subplot(1,2,1,projection="3d")
scatter(ax1, cube_fft(16), 16, "16×16×16 — цель straddled (размазана по z)")
ax2 = fig.add_subplot(1,2,2,projection="3d")
p = scatter(ax2, cube_fft(64), 64, "16×16×64 — zero-pad ×4: чёткая точка")
fig.suptitle("Вариант 3: пэд только по дальности (16→64) убирает straddle", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/home/claude/cube_64.png", dpi=120); plt.close(fig)
print("ok")
