import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(7)
AX = AY = 16; Lr = 256
n = np.arange(Lr)

def tone(bz, amp=1.0, ph=0.0):
    return amp*np.exp(1j*(2*np.pi*(bz/Lr)*n + ph))

# --- цель + гребёнка DRFM на нормали (boresight) ---
tgt_ft = sum(tone(b,a,rng.uniform(0,2*np.pi))
             for b,a in zip([40,60,80,100],[1.0,0.7,0.6,0.5]))     # передний фронт = 40

# --- радиолюбитель: CW × сопряжённый ЛЧМ = чирп -> размаз по всей дальности ---
beta = 1.0/Lr
ham_ft = 5.0*np.exp(1j*np.pi*beta*(n - Lr/2)**2)                   # размазанный отклик
kx0, ky0 = 5, -3                                                    # ДРУГОЙ угол прихода
ax_i = np.arange(AX); ay_i = np.arange(AY)
steer = np.exp(1j*2*np.pi*(kx0*ax_i[:,None]/AX + ky0*ay_i[None,:]/AY))

blk  = np.broadcast_to(tgt_ft,(AX,AY,Lr)).astype(complex).copy()    # цель+гребёнка (синфазно)
blk += steer[:,:,None]*ham_ft[None,None,:]                          # радиолюбитель под углом
blk += np.sqrt(0.02/2)*(rng.standard_normal((AX,AY,Lr))+1j*rng.standard_normal((AX,AY,Lr)))

wx,wy,wf = np.hanning(AX),np.hanning(AY),np.hanning(Lr)
blk = blk*wx[:,None,None]*wy[None,:,None]*wf[None,None,:]
C = np.fft.fftn(blk, s=(AX,AY,Lr))
C = np.fft.fftshift(C, axes=(0,1))                                  # угол центрируем, дальность нет
mag = np.abs(C)

kx = np.arange(-AX//2,AX//2); ky = np.arange(-AY//2,AY//2)
ZMAX = 130

# ============ Рис. A: 3D-куб — пространственное разделение ============
fz = np.arange(0,ZMAX); KX,KY,FZ = np.meshgrid(kx,ky,fz,indexing="ij")
ms = mag[:,:,0:ZMAX]; m = 20*np.log10(ms+1e-9); m -= m.max()
v = m.ravel(); sel = v > -22; vv = v[sel]
fig = plt.figure(figsize=(9,7.5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(KX.ravel()[sel],KY.ravel()[sel],FZ.ravel()[sel],c=vv,s=(vv+22)**2*0.5+5,
           cmap="turbo",vmin=-22,vmax=0,alpha=0.55,edgecolors="none")
ax.plot([0,0],[0,0],[40,40],"rv",ms=9)
ax.text(0,0,150,"цель + DRFM\n(boresight)",color="r",fontsize=9,ha="center")
ax.text(5,-3,150,"радиолюбитель\n(другой угол)",color="purple",fontsize=9,ha="center")
ax.set_xlabel("kx (азимут)"); ax.set_ylabel("ky (угол места)"); ax.set_zlabel("дальность (≥0)")
ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_zlim(0,ZMAX); ax.view_init(18,-58)
ax.set_title("16×16×256: цель/DRFM на нормали, радиолюбитель — отдельным столбом под другим углом")
fig.tight_layout(); fig.savefig("/home/claude/scen_cube.png",dpi=120); plt.close(fig)

# ============ Рис. B: угловая карта + дальностные профили ============
E = 20*np.log10(np.sqrt((mag**2).sum(axis=2))+1e-9); E -= E.max()    # энергия по дальности
fig, (a1,a2) = plt.subplots(1,2,figsize=(15,5.6))

im = a1.imshow(E.T, origin="lower", extent=[kx[0],kx[-1]+1,ky[0],ky[-1]+1],
               cmap="turbo", vmin=-25, vmax=0, aspect="equal")
a1.add_patch(Rectangle((-1.5,-1.5),3,3,fill=False,ec="r",lw=2))
a1.text(0,2.2,"гейт обзора\n(цель)",color="r",ha="center",fontsize=10)
a1.text(5,-3,"●",color="white",ha="center",va="center",fontsize=14)
a1.text(5,-5.0,"радиолюбитель\nвне гейта → отсев",color="white",ha="center",fontsize=9)
a1.set_xlabel("kx (азимут)"); a1.set_ylabel("ky (угол места)")
a1.set_title("Угловая карта энергии: источники разнесены по углу")
fig.colorbar(im,ax=a1,label="дБ",shrink=0.85)

r = np.arange(ZMAX)
prof_tgt = 20*np.log10(mag[8,8,0:ZMAX]+1e-9)
prof_ham = 20*np.log10(mag[8+kx0,8+ky0,0:ZMAX]+1e-9)
ref = max(prof_tgt.max(), prof_ham.max())
a2.plot(r, prof_tgt-ref, color="#993C1D", lw=1.8, label="ячейка цели (в гейте)")
a2.plot(r, prof_ham-ref, color="#534AB7", lw=1.6, label="ячейка радиолюбителя")
a2.axvline(40, color="r", ls="--", lw=1.2, label="передний фронт = цель")
a2.set_xlabel("бин дальности (задержка ≥ 0)"); a2.set_ylabel("дБ отн. макс")
a2.set_ylim(-35,3); a2.legend(loc="upper right",fontsize=9); a2.grid(alpha=0.25)
a2.set_title("Дальностные профили: цель = пики, радиолюбитель = размаз")
fig.suptitle("Отсев стороннего излучения: по углу (гейт) и по структуре (нет сжатия в пик)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("/home/claude/scen_filter.png",dpi=120); plt.close(fig)
print("ok")
