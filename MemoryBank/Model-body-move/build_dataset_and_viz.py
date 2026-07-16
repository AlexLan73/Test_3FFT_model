"""
Генерация мини-датасета (6 примеров) и трёх визуализаций:
  1) trajectory_model.png  — 4 модели движения цели в апертуре 16x16
  2) matrices_scenarios.png — матрицы 16x16 в 6 сценариях (цель/помехи)
  3) energy_profile.png     — энергетический профиль по 1249 кадрам
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from radar_simulator import (generate_sample, make_target, noise_thermal,
                             volume_to_matrices, NX, NY, NZ, N_FRAMES)

np.random.seed(42)
OUT = "/home/user/workspace/output"

# ------------------------------------------------------------------
# 1. МИНИ-ДАТАСЕТ (6 примеров)
# ------------------------------------------------------------------
configs = [
    dict(label=1, snr_db=16, motion="cv", iflags={}),                       # чистая цель
    dict(label=1, snr_db=12, motion="ct", turn_rate=1.0,
         iflags={"clutter": {"n_scatterers": 10, "scr_db": 6}}),            # вираж + клаттер
    dict(label=1, snr_db=10, motion="ca",
         iflags={"vfd": {"level_db": -28}}),                                # ускорение + VFD
    dict(label=1, snr_db=8, motion="cv",
         iflags={"cw": {"level_db": 4}, "arc": True}),                      # цель + CW + дуга
    dict(label=0, snr_db=0, motion="cv",
         iflags={"vfd": True, "clutter": {"n_scatterers": 15}}),            # только помехи
    dict(label=0, snr_db=0, motion="cv", iflags={"arc": True}),             # только дуга
]
X, Y, names = [], [], ["clean", "turn+clutter", "accel+vfd",
                       "cw+arc", "noise_vfd", "noise_arc"]
for c in configs:
    mats, lab, meta = generate_sample(**c)
    X.append(mats.astype(np.float32)); Y.append(lab)
X = np.stack(X); Y = np.array(Y, dtype=np.int64)
np.savez_compressed(f"{OUT}/radar_dataset_mini.npz", X=X, y=Y,
                    names=np.array(names))
print("dataset:", X.shape, "labels:", Y.tolist())

# ------------------------------------------------------------------
# 2. ТРАЕКТОРИИ ДВИЖЕНИЯ (4 модели) — в координатах апертуры kx-ky
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10), facecolor="#0d1117")
models = [("cv", 0.0, "Constant Velocity (прямолинейно)"),
          ("ca", 0.0, "Constant Acceleration (ускорение)"),
          ("ct", 1.0, "Coordinated Turn ↻ (вираж вправо)"),
          ("ct", -1.0, "Coordinated Turn ↺ (вираж влево)")]
for ax, (m, tr, title) in zip(axes.ravel(), models):
    vol = np.zeros((NX, NY, NZ), dtype=np.complex128)
    meta = make_target(vol, kx0=8, ky0=8, vel_kx=6.0, vel_ky=3.0,
                       vel_kz=3.0, snr_db=12, model=m, turn_rate=tr)
    kx, ky = meta["kx"], meta["ky"]
    ax.set_facecolor("#0d1117")
    ax.plot(kx, ky, color="#00e5ff", lw=1.0, alpha=0.35)
    tail = slice(-1000, None)
    ax.plot(kx[tail], ky[tail], color="#00ff9c", lw=2.2)
    ax.scatter(kx[0], ky[0], c="white", s=70, zorder=5, label="старт")
    ax.scatter(kx[-1], ky[-1], marker="*", c="#ffdd00", s=260, zorder=5,
               edgecolors="k", label="текущая")
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(-0.5, 15.5)
    ax.set_title(title, color="#e6edf3", fontsize=11, pad=8)
    ax.set_xlabel("kx (азимут)", color="#8b949e", fontsize=9)
    ax.set_ylabel("ky (угол места)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(alpha=0.15, color="#30363d")
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", loc="upper left")
fig.suptitle("Модели движения цели в апертуре 16×16",
             color="#e6edf3", fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/trajectory_model.png", dpi=130, facecolor="#0d1117")
plt.close(fig)
print("saved trajectory_model.png")

# ------------------------------------------------------------------
# 3. МАТРИЦЫ 16x16 В 6 СЦЕНАРИЯХ (на кадре с макс. энергией)
# ------------------------------------------------------------------
scen = [("Чистый сигнал", dict(label=1, snr_db=16, iflags={})),
        ("CW-помеха", dict(label=1, snr_db=10, iflags={"cw": {"level_db": 5}})),
        ("VFD/IGBT", dict(label=1, snr_db=10, iflags={"vfd": {"level_db": -24}})),
        ("Сварочная дуга", dict(label=1, snr_db=10, iflags={"arc": {"peak_db": 15}})),
        ("Клаттер", dict(label=1, snr_db=10, iflags={"clutter": {"n_scatterers": 18, "scr_db": 10}})),
        ("Все помехи", dict(label=1, snr_db=8,
                            iflags={"cw": {"level_db": 3}, "vfd": True,
                                    "arc": True, "clutter": True}))]
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), facecolor="#0d1117")
for ax, (title, cfg) in zip(axes.ravel(), scen):
    mats, _, _ = generate_sample(**cfg)
    n_star = int(mats.reshape(N_FRAMES, -1).sum(1).argmax())
    M = 10 * np.log10(mats[n_star] + 1e-9)
    im = ax.imshow(M, cmap="inferno", origin="lower", aspect="equal")
    ax.set_title(f"{title}\n(кадр n={n_star})", color="#e6edf3", fontsize=10)
    ax.set_xlabel("ky", color="#8b949e", fontsize=8)
    ax.set_ylabel("kx", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)
    cb.set_label("дБ", color="#8b949e", fontsize=8)
fig.suptitle("Матрица энергии 16×16: цель на фоне разных помех",
             color="#e6edf3", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/matrices_scenarios.png", dpi=130, facecolor="#0d1117")
plt.close(fig)
print("saved matrices_scenarios.png")

# ------------------------------------------------------------------
# 4. ЭНЕРГЕТИЧЕСКИЙ ПРОФИЛЬ по 1249 кадрам
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
profiles = [
    ("Чистая цель", dict(label=1, snr_db=16, iflags={}), "#00ff9c"),
    ("Цель + CW", dict(label=1, snr_db=10, iflags={"cw": {"level_db": 5}}), "#00e5ff"),
    ("Цель + VFD", dict(label=1, snr_db=10, iflags={"vfd": {"level_db": -24}}), "#ffdd00"),
    ("Цель + дуга", dict(label=1, snr_db=10, iflags={"arc": {"peak_db": 15}}), "#ff5c8a"),
]
for name, cfg, col in profiles:
    mats, _, _ = generate_sample(**cfg)
    e = mats.reshape(N_FRAMES, -1).sum(1)
    e_db = 10 * np.log10(e / e.min() + 1e-9)
    ax.plot(e_db, color=col, lw=1.3, label=name, alpha=0.9)
ax.set_title("Энергетический профиль по 1249 кадрам дальности",
             color="#e6edf3", fontsize=14)
ax.set_xlabel("Номер кадра n (грубая дальность)", color="#8b949e")
ax.set_ylabel("Суммарная энергия матрицы, дБ", color="#8b949e")
ax.tick_params(colors="#8b949e")
ax.grid(alpha=0.15, color="#30363d")
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/energy_profile.png", dpi=130, facecolor="#0d1117")
plt.close(fig)
print("saved energy_profile.png")
