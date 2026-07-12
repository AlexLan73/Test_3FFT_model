import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Sans"
fig,ax=plt.subplots(figsize=(16,12),dpi=140); ax.set_xlim(0.5,18.2); ax.set_ylim(0.2,11.2); ax.axis("off")
HW,HH=0.55,0.42
N={
 1:(2.0,1.4),2:(2.0,9.3),3:(3.9,9.3),
 5:(5.8,9.3),8:(7.4,9.3),9:(9.0,9.3),10:(10.6,9.3),
 6:(5.8,7.7),11:(7.4,7.7),12:(9.0,7.7),13:(10.6,7.7),
 7:(5.8,6.1),19:(12.4,6.1),20:(14.0,6.1),21:(15.6,6.1),
 16:(7.4,4.6),17:(9.0,4.6),18:(10.6,4.6),
 14:(12.4,9.3),15:(14.0,7.9),4:(10.5,2.4),
}
def box(n):
    x,y=N[n]; ax.add_patch(FancyBboxPatch((x-HW,y-HH),2*HW,2*HH,boxstyle="round,pad=0.02,rounding_size=0.08",fc="white",ec="#111",lw=1.6))
    ax.text(x,y,str(n),ha="center",va="center",fontsize=14,fontweight="bold",color="#111")
def R(n,s):
    x,y=N[n]; return {'r':(x+HW,y),'l':(x-HW,y),'t':(x,y+HH),'b':(x,y-HH)}[s]
def line(pts,dashed=False,arrow=True):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color="#111",lw=1.5,linestyle=(":" if dashed else "-"),solid_capstyle="round",zorder=1)
    if arrow: ax.annotate("",xy=pts[-1],xytext=pts[-2],arrowprops=dict(arrowstyle="-|>",color="#111",lw=1.5,mutation_scale=13),zorder=2)
def dot(x,y): ax.plot([x],[y],'o',color="#111",ms=4,zorder=3)
def lbl(x,y,t,fs=8.2,color="#444",ha="center"): ax.text(x,y,t,ha=ha,va="center",fontsize=fs,color=color,style="italic")

for n in N: box(n)
# антенна
ax.plot([1.72,2.0],[10.55,10.2],color="#111",lw=1.4); ax.plot([2.28,2.0],[10.55,10.2],color="#111",lw=1.4); ax.plot([2.0,2.0],[10.2,9.72],color="#111",lw=1.4)
line([(2.0,10.2),(2.0,9.72)])
# приёмный тракт
line([R(2,'r'),R(3,'l')])
# развилка 3 -> 5/6/7
line([R(3,'r'),R(5,'l')]); dot(4.9,9.3)
line([(4.9,9.3),(4.9,7.7),R(6,'l')]); line([(4.9,9.3),(4.9,6.1),R(7,'l')])
lbl(5.0,9.52,"ЛЧМ",ha="center"); lbl(5.1,7.92,"АМ",ha="center"); lbl(5.1,6.32,"ФМ",ha="center")
# ветки
for a,b in [(5,8),(8,9),(9,10),(6,11),(11,12),(12,13),(16,17),(17,18),(19,20),(20,21)]:
    line([R(a,'r'),R(b,'l')])
line([R(7,'r'),R(19,'l')]); lbl(9.3,6.3,"спектр принятого")
line([R(18,'r'),(12.4,4.6),R(19,'b')]); lbl(11.6,4.85,"опора")
# токены -> 14
line([R(10,'r'),R(14,'l')]); lbl(11.5,9.55,"токены")
line([R(13,'r'),(11.55,7.7),(11.55,9.15),(11.85,9.15)]); lbl(11.75,8.4,"токены",ha='left')
# 14 -> 15 (добор трудных)
line([R(14,'b'),(12.4,7.9),R(15,'l')]); lbl(12.6,7.65,"на добор",ha='left')
# результаты 14/15/21 -> 4 (правая шина -> в блок 4)
line([R(14,'r'),(17.4,9.3)],arrow=False); line([R(15,'r'),(17.4,7.9)],arrow=False); line([R(21,'r'),(17.4,6.1)],arrow=False)
dot(17.4,9.3); dot(17.4,7.9); dot(17.4,6.1)
line([(17.4,9.3),(17.4,2.4),R(4,'r')]); lbl(14.0,2.62,"объекты / токены  →  блок 4")
# 4 -> 1 (матрица токенов)
line([R(4,'l'),(3.2,2.4),(3.2,1.4),R(1,'r')]); lbl(6.6,2.62,"матрица токенов")
# 1 -> 4 (команды)
line([(R(1,'b')[0],1.05),(10.5,1.05),(10.5,N[4][1]-HH)],dashed=True); lbl(6.6,1.25,"команды",color="#666")
# 1 -> 2 (управление РЧ)
line([R(1,'t'),R(2,'b')]); lbl(2.3,5.2,"тип сигнала,\nпараметры",ha='left',color="#666")

ax.set_title("Блок-схема устройства ECCM 3FFT (Фиг. 1)",fontsize=15,fontweight="bold",pad=6)
ax.text(0.6,0.4,"──── данные        ⋯⋯ управление",fontsize=9,color="#444")
plt.savefig("blocksch_ustroystvo.png",bbox_inches="tight",facecolor="white"); print("ok")
