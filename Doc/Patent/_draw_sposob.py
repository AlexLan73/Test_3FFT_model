import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mp

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
ax.set_xlim(0, 12); ax.set_ylim(0, 22); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3fb", ec="#2E5B9A", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                fc=fc, ec=ec, lw=1.7))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def diamond(xc, yc, w, h, text, fc="#fdf3e3", ec="#c08a2a", fs=10):
    ax.add_patch(mp.Polygon([(xc, yc+h/2),(xc+w/2, yc),(xc, yc-h/2),(xc-w/2, yc)],
                 closed=True, fc=fc, ec=ec, lw=1.7))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs)

def arr(x1,y1,x2,y2,text="",rad=0.0,color="#333",fs=9):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,
                lw=1.6,color=color,connectionstyle=f"arc3,rad={rad}"))
    if text: ax.text((x1+x2)/2+0.2,(y1+y2)/2,text,ha="left",va="center",fontsize=fs,color="#555",style="italic")

cx=4.0; w=5.4; h=1.2
ys=[20.2,18.4,16.6,14.8,13.0]
labels=[
 "Излучение широкого ЛЧМ",
 "Приём АФАР → АЦП → дечирп\n→ N комплексных отсчётов (N перем.)",
 "Два раздельных FFT →\nобъём 16×16×N",
 "Токенизация срезов →\nматрица токенов",
 "Гейт: классификация среза\n(шум/собранный/размазанный)",
]
for y,l in zip(ys,labels):
    box(cx-w/2,y,w,h,l, bold=(l.startswith("Токениз")))
# downward arrows only
for i in range(len(ys)-1):
    arr(cx, ys[i], cx, ys[i+1]+h)

dcy=11.2
diamond(cx, dcy, 4.6, 1.8, "Структура\nрегулярная?")
arr(cx, ys[-1], cx, dcy+0.9)

box(cx-w/2, 8.6, w, 1.3, "Арбитр: передний край τ≥0\nи/или согласование с кодом", fc="#e8f0e8", ec="#3a7d3a", bold=True)
arr(cx, dcy-0.9, cx, 8.6+1.3, "да")

box(8.3, 10.4, 3.4, 1.6, "L3: LSTM над\nтокенами ROI", fc="#f4f1fb", ec="#6a4fb0")
arr(cx+2.3, dcy, 8.3, 11.2, "нет (трудный)", rad=-0.2)
arr(9.9, 10.4, cx+1.2, 9.2, "уточн. (угол, дальн.)", rad=-0.25, color="#6a4fb0")

box(cx-w/2, 6.6, w, 1.2, "Уточнённые (угол, дальность)")
arr(cx, 8.6, cx, 6.6+1.2)
box(cx-w/2, 4.8, w, 1.2, "Многолучевой опрос FM-m\n(агильный код 2ⁿ)", fc="#fbeeea", ec="#b0553a", bold=True)
arr(cx, 6.6, cx, 4.8+1.2)
box(cx-w/2, 3.0, w, 1.2, "Подтверждение\nкод-корреляцией")
arr(cx, 4.8, cx, 3.0+1.2)
box(cx-w/2, 1.2, w, 1.2, "Целеуказание", fc="#e8f0e8", ec="#3a7d3a", bold=True)
arr(cx, 3.0, cx, 1.2+1.2)
# loop back
arr(cx-w/2, 1.8, 0.5, 1.8)
ax.add_patch(FancyArrowPatch((0.5,1.8),(0.5,20.8),arrowstyle="-",lw=1.4,color="#888"))
arr(0.5,20.8,cx-w/2,20.8, "следующий такт", color="#888")

ax.set_title("Схема осуществления способа (Фиг. 1)", fontsize=14, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("shema_sposob.png", bbox_inches="tight", facecolor="white")
print("saved")
