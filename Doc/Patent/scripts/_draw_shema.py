import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3fb", ec="#2E5B9A", fs=11, lw=1.6, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            wrap=True, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, text="", fs=9, color="#333", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                        lw=1.6, color=color, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.25, text, ha="center", va="center",
                fontsize=fs, color="#555", style="italic")

# Приёмный тракт (вертикально слева)
box(0.6, 10.2, 3.2, 1.2, "1 — АФАР", fc="#e8f0e8", ec="#3a7d3a", bold=True)
box(0.6, 8.4, 3.2, 1.1, "2 — АЦП")
box(0.6, 6.6, 3.2, 1.1, "3 — Гетеродин-дечирп\n(ПЛИС, stretch)")
arrow(2.2, 10.2, 2.2, 9.5, "приём")
arrow(2.2, 8.4, 2.2, 7.7)
arrow(2.2, 6.6, 2.2, 5.2, "16×16×N")

# GPU-вычислитель (широкий блок)
box(0.4, 1.4, 12.4, 3.6, "", fc="#f4f1fb", ec="#6a4fb0", lw=2)
ax.text(0.7, 4.7, "4 — Вычислитель (GPU)", ha="left", va="center",
        fontsize=12, fontweight="bold", color="#6a4fb0")
yb=2.4; wb=2.15; hb=1.5
box(0.7,  yb, wb, hb, "4.1\nдва FFT\n16×16×N", fc="#efeaf9")
box(3.05, yb, wb, hb, "4.2\nтокенизатор\n(признаки)", fc="#efeaf9")
box(5.4,  yb, wb, hb, "4.3\nгейт\n(классиф.)", fc="#efeaf9")
box(7.75, yb, wb, hb, "4.4\nарбитр\n(τ≥0)", fc="#efeaf9")
box(10.1, yb, wb+0.3, hb, "4.5\nкоррелятор\nFM-m", fc="#efeaf9")
for x0 in [2.85, 5.2, 7.55, 9.9]:
    arrow(x0, yb+hb/2, x0+0.2, yb+hb/2)

# Формирователь FM-m
box(13.2, 6.6, 2.6, 1.6, "5 — Формирователь\nагильного FM-m\n(длина 2ⁿ)", fc="#fbeeea", ec="#b0553a", bold=True)
# ЛЧМ-канал
box(13.2, 9.4, 2.6, 1.5, "6 — Канал\nширокого ЛЧМ-\nобзора", fc="#fbf6e8", ec="#b09a3a")

# целеуказание: арбитр -> формирователь
arrow(9.0, 3.9, 13.2, 7.0, "матрица токенов →\nцелеуказание", rad=0.15)
# формирователь -> АФАР (излучение)
arrow(14.5, 8.2, 3.8, 10.9, "излучение FM-m", rad=-0.25, color="#b0553a")
# ЛЧМ-канал -> АФАР
arrow(13.2, 10.2, 3.9, 10.9, "зондирование ЛЧМ", rad=0.12, color="#8a7a2a")

ax.set_title("Структурная схема устройства (Фиг. 1)", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("struct_shema.png", bbox_inches="tight", facecolor="white")
print("saved struct_shema.png")
