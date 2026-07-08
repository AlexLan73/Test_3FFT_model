import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
AX = AY = 16          # угловой квадрат
NR   = 32             # число участков дальности (кубов в стопке)

# ---------- синтез кадра: cube[kx, ky, r] = энергия ----------
cube = 0.4*rng.random((AX, AY, NR))          # шумовой пол

def blob(cx, cy, r, amp, sig=0.9):
    x = np.arange(AX)[:,None]; y = np.arange(AY)[None,:]
    g = amp*np.exp(-((x-cx)**2+(y-cy)**2)/(2*sig**2))
    cube[:,:,r] += g

# цель + гребёнка DRFM на boresight (kx≈8.2, ky≈8.4) — центроид дробный
tgt = (8.2, 8.4)
blob(*tgt, r=6,  amp=100)      # передний край = цель
for m,(rr,a) in enumerate([(12,78),(18,70),(24,63)]):
    blob(*tgt, r=rr, amp=a)    # ложные, ТОЛЬКО дальше края

# радиолюбитель: угол (13.1, 3.0), на ВСЕХ участках дальности
for r in range(NR):
    blob(13.1, 3.0, r, amp=88)

# ---------- ТОКЕНИЗАТОР (заглушка обученного НН) ----------
# на каждый срез дальности: порог -> кластеры ярких ячеек -> центроид+пик
def tokenize(cube, thr=25.0):
    tokens = []
    for r in range(cube.shape[2]):
        sl = cube[:,:,r]
        mask = sl > thr
        if not mask.any():
            continue
        # простая кластеризация связностью 8 (для демо — по локальным максимумам)
        from scipy.ndimage import label, center_of_mass, maximum_position
        lab, n = label(mask)
        for k in range(1, n+1):
            cy, cx = center_of_mass(sl, lab, k)   # субъячеечный центроид (float)
            I = sl[maximum_position(sl, lab, k)]
            tokens.append((round(cx,1), round(cy,1), r, round(float(I))))
    return tokens

try:
    tokens = tokenize(cube)
except ImportError:
    import subprocess,sys
    subprocess.run([sys.executable,"-m","pip","install","scipy","--break-system-packages","-q"])
    tokens = tokenize(cube)

# ---------- вывод разрежённого кода ----------
dense = cube.size
sparse = len(tokens)
print(f"Плотный куб: {AX}x{AY}x{NR} = {dense} чисел")
print(f"Разрежённый код: {sparse} кортежей (сжатие {dense/sparse:.0f}x)\n")
print("КОД СЦЕНЫ  (kx, ky, r, I):")
for t in tokens:
    print("  ", t)

# ---------- читатель физического якоря ----------
print("\n--- разбор кодом (физ.якорь) ---")
# группируем по углу (округл. до целого) чтобы найти столбцы и гребёнки
from collections import defaultdict
by_angle = defaultdict(list)
for cx,cy,r,I in tokens:
    by_angle[(round(cx),round(cy))].append((r,I))
for (cx,cy),lst in by_angle.items():
    rs = sorted(r for r,_ in lst)
    span = len(rs)
    if span >= NR*0.7:
        print(f"  угол(~{cx},{cy}): столб на {span}/{NR} R  ->  ЗАГРАДКА/RFI (режем по углу)")
    elif span >= 2:
        front = rs[0]
        print(f"  угол(~{cx},{cy}): {span} пиков, край R={front} ->  ЦЕЛЬ(край) + ГРЕБЁНКА DRFM позади")
    else:
        print(f"  угол(~{cx},{cy}): одиночный пик R={rs[0]} ->  точечная цель")

# ---------- визуал: плотно vs код ----------
fig = plt.figure(figsize=(13,5.5))
# плотно
ax1 = fig.add_subplot(121, projection='3d')
KX,KY,R = np.meshgrid(np.arange(AX),np.arange(AY),np.arange(NR),indexing='ij')
v = cube.ravel(); sel = v>20
ax1.scatter(R.ravel()[sel], KX.ravel()[sel], KY.ravel()[sel],
            c=v[sel], cmap='turbo', s=6, alpha=0.25)
ax1.set_title(f'плотный куб (~{sel.sum()} вокселей)')
ax1.set_xlabel('дальность'); ax1.set_ylabel('kx'); ax1.set_zlabel('ky')
ax1.view_init(16,-72)
# код
ax2 = fig.add_subplot(122, projection='3d')
for cx,cy,r,I in tokens:
    ax2.scatter(r, cx, cy, s=I*2.2, c=[[I/100,0.1,0.5]], edgecolors='k', linewidths=0.4)
ax2.set_title(f'разрежённый КОД ({sparse} кортежей)')
ax2.set_xlabel('дальность'); ax2.set_ylabel('kx'); ax2.set_zlabel('ky')
ax2.set_xlim(0,NR); ax2.set_ylim(0,AX); ax2.set_zlim(0,AY)
ax2.view_init(16,-72)
plt.suptitle('16×16×32: тот же кадр — плотный тензор vs код сцены (kx,ky,r,I)')
plt.tight_layout(); plt.savefig('/home/claude/tokenize.png', dpi=115, bbox_inches='tight')
print("\nfig ok")
