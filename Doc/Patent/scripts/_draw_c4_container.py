import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Sans"
fig,ax=plt.subplots(figsize=(12.5,16),dpi=140); ax.set_xlim(0,14); ax.set_ylim(0,16); ax.axis("off")

def box(x0,y0,x1,y1,num,fc="#eef3fb",ec="#2E5B9A",ext=False):
    if ext: fc,ec="#eceff2","#7f8890"
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.02,rounding_size=0.10",fc=fc,ec=ec,lw=1.9))
    ax.text((x0+x1)/2,(y0+y1)/2,str(num),ha="center",va="center",fontsize=19,fontweight="bold",color=("#3a424a" if ext else "#14335c"))

def orth(pts,color="#333",dashed=False):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color=color,lw=1.7,solid_capstyle="round",linestyle=(":" if dashed else "-"),zorder=1)
    ax.annotate("",xy=pts[-1],xytext=pts[-2],arrowprops=dict(arrowstyle="-|>",color=color,lw=1.7,mutation_scale=14),zorder=2)
def lab(x,y,t,fs=8.6,ha="left",color="#555"):
    ax.text(x,y,t,ha=ha,va="center",fontsize=fs,color=color,style="italic")

# боксы (только номера)
box(1.2,14.4,5.2,15.4,1,ext=True)          # приёмный тракт
box(1.2,12.4,5.2,13.8,2)                    # фронтенд
box(1.2,10.2,5.2,11.6,3)                    # ядро-токенизатор
box(1.2,7.8,5.2,9.4,4,fc="#efeaf9",ec="#6a4fb0")   # распознавание
box(1.2,5.2,5.2,6.7,6,fc="#e8f2e8",ec="#3a7d3a")   # целеуказание
box(1.2,3.3,5.2,4.3,8,ext=True)            # оператор
box(8.6,12.3,12.6,13.6,7,ext=True)         # формирователь зондов
box(8.6,7.8,12.6,9.4,5,fc="#fbeeea",ec="#b0553a")  # опрос FM-m

# связи (ортогонально, без пересечений и сквозь боксы; подпись рядом с линией)
orth([(3.2,14.4),(3.2,13.8)]);              lab(3.45,14.1,"куб / IQ")
orth([(3.2,12.4),(3.2,11.6)]);              lab(3.45,12.0,"куб 16×16×L")
orth([(3.2,10.2),(3.2,9.4)]);               lab(3.45,9.8,"токены")
orth([(3.2,7.8),(3.2,6.7)]);                lab(3.45,7.25,"кандидаты")
orth([(3.2,5.2),(3.2,4.3)]);                lab(3.45,4.75,"целеуказание")
orth([(5.2,5.9),(6.7,5.9),(6.7,8.3),(8.6,8.3)],color="#b0553a"); lab(6.9,7.1,"пучок/код",color="#b0553a")
orth([(8.6,8.9),(5.2,8.9)],color="#6a4fb0");           lab(6.9,9.15,"отклик",ha="center",color="#6a4fb0")
orth([(5.2,5.5),(13.0,5.5),(13.0,12.95),(12.6,12.95)],color="#3a7d3a"); lab(8.6,5.75,"команда зонда / кода",ha="center",color="#3a7d3a")
orth([(1.2,6.3),(0.5,6.3),(0.5,13.1),(1.2,13.1)],color="#3a7d3a");      lab(0.7,9.7,"новый такт (код / N)",color="#3a7d3a",fs=8.0)

ax.set_title("Структурная схема системы ECCM 3FFT",fontsize=15,fontweight="bold",pad=8)
# описание позиций — под рисунком
leg=[
 "1 — приёмный тракт (АФАР, АЦП, гетеродин-дечирп на ПЛИС);   2 — сменный фронтенд (ЛЧМ / AM), заполняющий куб;",
 "3 — ядро-токенизатор (объёмный, инвариантный к фронтенду);   4 — распознавание (ИИ-гейт, причинностный арбитр τ≥0 / код, L3);",
 "5 — опрос FM-m (коррелятор, генератор агильного кода);   6 — целеуказание и когнитивная петля;",
 "7 — формирователь зондов (ЦАП);   8 — оператор / боевая система.",
]
y=2.55
for s in leg:
    ax.text(0.4,y,s,ha="left",va="center",fontsize=9.2,color="#222"); y-=0.52
ax.text(0.4,0.35,"────── данные        ⇢ управление",ha="left",fontsize=9,color="#555")
plt.savefig("c4_container.png",bbox_inches="tight",facecolor="white"); print("ok")
