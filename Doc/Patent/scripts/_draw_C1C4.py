import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"]="DejaVu Sans"
fig,ax=plt.subplots(figsize=(16.5,7.6),dpi=150); ax.set_xlim(0,17.0); ax.set_ylim(0,9); ax.axis("off")

def box(x,y,w,h,t,fc,ec,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.12",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,y1,x2,y2,t="",fs=9,rad=0,color="#333"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,lw=1.8,color=color,connectionstyle=f"arc3,rad={rad}"))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+0.30,t,ha="center",va="center",fontsize=fs,color="#444",style="italic")

# входы
box(0.3,6.05,2.9,1.1,"ЛЧМ-интерфейс\n(два FFT · точно)","#e8f0e8","#3a7d3a")
box(0.3,4.05,2.9,1.1,"AM-интерфейс\n(3D-FFT · грубо/быстро)","#fbf6e8","#b09a3a")
ax.text(1.75,7.5,"СМЕННЫЙ ФРОНТЕНД",ha="center",fontsize=10,fontweight="bold",color="#555")

w=2.55; y=4.5; h=1.55
cx=[3.75,6.95,10.15,13.35]
data=[("C1","Формирование куба\nn×n×L\n(два FFT / 3D-FFT)","(а)","#eef3fb","#2E5B9A"),
      ("C2","Токенизатор:\nстраддл-инвариантный\nтокен","(б)","#efeaf9","#6a4fb0"),
      ("C3","ИИ-гейт +\nпричинностный арбитр\n(τ≥0 / код)","(в)","#efeaf9","#6a4fb0"),
      ("C4","Целеуказание\nагильного\nFM-m","(г)","#fbeeea","#b0553a")]
for x,(c,t,el,fc,ec) in zip(cx,data):
    box(x,y,w,h,t,fc,ec,fs=10)
    ax.text(x+0.12,y+h-0.03,c,ha="left",va="top",fontsize=15,fontweight="bold",color=ec)
    ax.text(x+w-0.12,y+0.06,el,ha="right",va="bottom",fontsize=12,color="#999",style="italic")

arr(3.2,6.6,3.75,5.55,rad=-0.15,color="#3a7d3a")
arr(3.2,4.6,3.75,5.0,rad=0.15,color="#8a7a2a")
arr(cx[0]+w,y+h/2,cx[1],y+h/2,"куб n×n×L",fs=8.5)
arr(cx[1]+w,y+h/2,cx[2],y+h/2,"токены",fs=8.5)
arr(cx[2]+w,y+h/2,cx[3],y+h/2,"цель/ложь",fs=8.5)
arr(cx[3]+w,y+h/2,16.85,y+h/2,"→ FM-m",fs=8.5)

x0=cx[1]-0.12; x1=cx[3]+w+0.12; yb=y+h+0.5
ax.plot([x0,x0,x1,x1],[yb-0.25,yb,yb,yb-0.25],color="#6a4fb0",lw=2)
ax.text((x0+x1)/2,yb+0.28,"ЯДРО — инвариантно к фронтенду · токен = универсальный интерфейс",
        ha="center",fontsize=11,fontweight="bold",color="#6a4fb0")
ax.text(cx[0]+w/2,y-0.45,"сменный интерфейс",ha="center",fontsize=10,style="italic",color="#3a7d3a")
ax.text(8.5,1.7,"Совокупность C1–C4 = (а)+(б)+(в)+(г) — ядро новизны независимого пункта",
        ha="center",fontsize=11,color="#333")
ax.set_title("Архитектура C1–C4: одно ядро, сменные интерфейсы",fontsize=14,fontweight="bold",pad=8)
plt.tight_layout(); plt.savefig("arh_C1C4.png",bbox_inches="tight",facecolor="white"); print("ok")
