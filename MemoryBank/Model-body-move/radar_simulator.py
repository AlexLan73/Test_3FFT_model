"""
radar_simulator.py
==================================================================
Симулятор радарного пайплайна:
  - движение цели в прямоугольнике 16x16x10000 (комплексный тензор
    после гетеродина, I/Q base-band);
  - 4 модели производственных помех (VFD/IGBT, сварочная дуга,
    CW чужого радара, клаттер);
  - нарезка по дальности скользящим окном W=16, шаг s=8 (overlap 50%)
    -> 3D FFT -> свёртка по kz -> 1249 матриц 16x16;
  - генерация обучающих примеров для 2D CNN.

Автор:    Александр Ланин
Платформа: AMD RX 9070 (16 GB VRAM) · ROCm · HIP · rocFFT
Примечание: код на NumPy для прототипирования/генерации датасета.
           Для боевого варианта FFT-батч переносится на rocFFT.
==================================================================
"""

from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------
# ГЛОБАЛЬНАЯ ГЕОМЕТРИЯ
# ------------------------------------------------------------------
NX, NY, NZ = 16, 16, 10000     # апертура X, апертура Y, дальность (отсчёты)
WIN = 16                       # длина окна по дальности
STEP = 8                       # шаг окна (overlap 50%)
N_FRAMES = (NZ - WIN) // STEP + 1     # = 1249


# ==================================================================
# ШАГ 1. ДВИЖЕНИЕ ЦЕЛИ
# ==================================================================
def make_target(vol: np.ndarray,
                kx0: float = 5.0, ky0: float = 7.0, kz0: float = 0.0,
                vel_kx: float = 0.0, vel_ky: float = 0.0, vel_kz: float = 3.0,
                snr_db: float = 12.0,
                model: str = "cv",
                turn_rate: float = 0.0) -> dict:
    """
    Добавляет в объём движущуюся точечную цель.

    Позиция в апертуре (kx, ky) и по дальности kz эволюционирует вдоль z.
    Билинейная интерполяция цели в 4 ближайших угловых узла 16x16.

    model:
      'cv' — constant velocity (прямолинейно, равномерно)
      'ca' — constant acceleration (ускорение/торможение по kz)
      'ct' — coordinated turn (вираж по азимуту, turn_rate рад на объём)
    """
    rng_amp = 10.0 ** (snr_db / 20.0)          # амплитуда цели над шумом
    z = np.arange(NZ, dtype=np.float64)
    t = z / NZ                                 # нормированное «время» 0..1

    # --- закон движения ---
    if model == "cv":
        kx = kx0 + vel_kx * z / 1000.0
        ky = ky0 + vel_ky * z / 1000.0
        kz = kz0 + vel_kz * z
    elif model == "ca":
        acc = vel_kz * 0.5
        kx = kx0 + vel_kx * z / 1000.0
        ky = ky0 + vel_ky * z / 1000.0
        kz = kz0 + vel_kz * z + 0.5 * acc * (z / 1000.0) ** 2
    elif model == "ct":
        r = 4.0                                # радиус виража в бинах апертуры
        ang = turn_rate * t * 2.0 * np.pi
        kx = kx0 + r * np.sin(ang)
        ky = ky0 + r * (1.0 - np.cos(ang))
        kz = kz0 + vel_kz * z
    else:
        raise ValueError(f"unknown motion model: {model}")

    # держим цель в апертуре
    kx = np.clip(kx, 0, NX - 1.001)
    ky = np.clip(ky, 0, NY - 1.001)

    # фаза сигнала (когерентная составляющая по дальности)
    phase = np.exp(1j * 2.0 * np.pi * 0.05 * z)

    # --- билинейная раскладка в 4 узла ---
    ix = np.floor(kx).astype(int)
    iy = np.floor(ky).astype(int)
    fx = kx - ix
    fy = ky - iy
    for zi in range(NZ):
        a = rng_amp * phase[zi]
        x0, y0, dx, dy = ix[zi], iy[zi], fx[zi], fy[zi]
        vol[x0,     y0,     zi] += a * (1 - dx) * (1 - dy)
        vol[x0 + 1, y0,     zi] += a * dx       * (1 - dy)
        vol[x0,     y0 + 1, zi] += a * (1 - dx) * dy
        vol[x0 + 1, y0 + 1, zi] += a * dx       * dy

    return {"kx": kx, "ky": ky, "kz": kz, "model": model, "snr_db": snr_db}


# ==================================================================
# ШАГ 2. МОДЕЛИ ПОМЕХ
# ==================================================================
def noise_thermal(vol: np.ndarray, sigma: float = 1.0):
    """Тепловой белый гауссовский шум (комплексный)."""
    vol += (np.random.randn(*vol.shape) +
            1j * np.random.randn(*vol.shape)) * (sigma / np.sqrt(2))


def noise_vfd(vol: np.ndarray, f_sw_norm: float = 0.04,
              n_harm: int = 12, level_db: float = -30.0):
    """VFD/IGBT: гармонический гребень n*f_sw + шумовой фон."""
    lvl = 10.0 ** (level_db / 20.0)
    z = np.arange(NZ)
    comb = np.zeros(NZ, dtype=np.complex128)
    for n in range(1, n_harm + 1):
        comb += np.exp(1j * (2 * np.pi * n * f_sw_norm * z +
                             np.random.uniform(0, 2 * np.pi)))
    comb *= lvl / n_harm
    # наводится изотропно на всю апертуру со слабой пространственной вариацией
    vol += comb[None, None, :] * (0.8 + 0.4 * np.random.rand(NX, NY, 1))


def noise_arc(vol: np.ndarray, rate: float = 0.001,
              peak_db: float = 12.0, tau_decay: float = 1.5):
    """Сварочная дуга: пуассоновские импульсы с тяжёлыми хвостами."""
    pk = 10.0 ** (peak_db / 20.0)
    n_imp = np.random.poisson(rate * NZ)
    for _ in range(n_imp):
        z0 = np.random.randint(0, NZ)
        amp = pk * np.abs(np.random.standard_cauchy())      # тяжёлый хвост
        length = int(max(1, np.random.exponential(tau_decay * 3)))
        zz = np.arange(z0, min(z0 + length, NZ))
        env = amp * np.exp(-(zz - z0) / tau_decay)
        phi = np.random.uniform(0, 2 * np.pi)
        vol[:, :, zz] += (env * np.exp(1j * phi))[None, None, :]


def noise_cw(vol: np.ndarray, f_norm: float = 0.15,
             level_db: float = 3.0, kx_dir: float = 3.0, ky_dir: float = 11.0):
    """CW чужого радара: плоская волна -> острый пик в матрице 16x16."""
    lvl = 10.0 ** (level_db / 20.0)
    x = np.arange(NX)[:, None, None]
    y = np.arange(NY)[None, :, None]
    z = np.arange(NZ)[None, None, :]
    spatial = np.exp(1j * 2 * np.pi * (kx_dir * x / NX + ky_dir * y / NY))
    temporal = np.exp(1j * 2 * np.pi * f_norm * z)
    vol += lvl * spatial * temporal


def noise_clutter(vol: np.ndarray, n_scatterers: int = 15,
                  scr_db: float = 8.0):
    """Клаттер: стационарные log-normal отражатели (нулевой доплер)."""
    scr = 10.0 ** (scr_db / 20.0)
    for _ in range(n_scatterers):
        kx = np.random.randint(0, NX)
        ky = np.random.randint(0, NY)
        amp = scr * np.random.lognormal(0, 0.5)
        phi = np.random.uniform(0, 2 * np.pi)
        vol[kx, ky, :] += amp * np.exp(1j * phi)     # постоянно во всей дальности


# ==================================================================
# ШАГ 3. НАРЕЗКА -> 3D FFT -> СВЁРТКА ПО kz -> 1249 МАТРИЦ 16x16
# ==================================================================
def _windows():
    """Пространственное окно Чебышёва (аппрокс.) x Hann по дальности."""
    wx = np.hamming(NX)
    wy = np.hamming(NY)
    wz = np.hanning(WIN)
    return wx, wy, wz


def volume_to_matrices(vol: np.ndarray, reduce: str = "max") -> np.ndarray:
    """
    Скользящее окно по Z (W=16, step=8) -> 3D FFT -> |F|^2 ->
    свёртка по kz (max или sum) -> (N_FRAMES, 16, 16).
    """
    wx, wy, wz = _windows()
    win3d = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    out = np.empty((N_FRAMES, NX, NY), dtype=np.float64)
    for n in range(N_FRAMES):
        z0 = n * STEP
        cube = vol[:, :, z0:z0 + WIN] * win3d
        F = np.fft.fftn(cube)
        p = np.abs(F) ** 2
        out[n] = p.max(axis=2) if reduce == "max" else p.sum(axis=2)
    return out


# ==================================================================
# ШАГ 4. ГЕНЕРАЦИЯ ОБУЧАЮЩЕГО ПРИМЕРА
# ==================================================================
def generate_sample(label: int = 1,
                    snr_db: float = 12.0,
                    kx0: float = 5.0, ky0: float = 7.0,
                    vel_kz: float = 3.0, vel_kx: float = 0.0, vel_ky: float = 0.0,
                    motion: str = "cv", turn_rate: float = 0.0,
                    iflags: dict | None = None,
                    reduce: str = "max"):
    """
    Один пример датасета.
      label=1 -> цель присутствует, label=0 -> только помехи/шум.
      iflags -> dict включения помех, например:
        {'vfd': {'f_sw_norm':0.04,'level_db':-30},
         'arc': True,
         'cw':  {'f_norm':0.15,'level_db':3},
         'clutter': {'n_scatterers':10,'scr_db':6}}
    Возвращает: (matrices[N_FRAMES,16,16], label, meta_dict)
    """
    iflags = iflags or {}
    vol = np.zeros((NX, NY, NZ), dtype=np.complex128)

    noise_thermal(vol, sigma=1.0)

    meta = {"label": label}
    if label == 1:
        meta["target"] = make_target(
            vol, kx0=kx0, ky0=ky0, vel_kx=vel_kx, vel_ky=vel_ky,
            vel_kz=vel_kz, snr_db=snr_db, model=motion, turn_rate=turn_rate)

    if "vfd" in iflags and iflags["vfd"]:
        kw = iflags["vfd"] if isinstance(iflags["vfd"], dict) else {}
        noise_vfd(vol, **kw)
    if "arc" in iflags and iflags["arc"]:
        kw = iflags["arc"] if isinstance(iflags["arc"], dict) else {}
        noise_arc(vol, **kw)
    if "cw" in iflags and iflags["cw"]:
        kw = iflags["cw"] if isinstance(iflags["cw"], dict) else {}
        noise_cw(vol, **kw)
    if "clutter" in iflags and iflags["clutter"]:
        kw = iflags["clutter"] if isinstance(iflags["clutter"], dict) else {}
        noise_clutter(vol, **kw)

    mats = volume_to_matrices(vol, reduce=reduce)
    meta["interference"] = list(iflags.keys())
    return mats, label, meta


# ==================================================================
# ДЕМО
# ==================================================================
if __name__ == "__main__":
    print(f"Геометрия: {NX}x{NY}x{NZ}, окно={WIN}, шаг={STEP} -> {N_FRAMES} кадров")
    mats, label, meta = generate_sample(
        label=1, snr_db=14.0, kx0=5, ky0=7, vel_kz=3.0, motion="cv",
        iflags={"vfd": {"f_sw_norm": 0.04, "level_db": -28.0},
                "arc": True,
                "clutter": {"n_scatterers": 10, "scr_db": 5.0}})
    print("matrices:", mats.shape, "| label:", label, "| помехи:", meta["interference"])
    n_star = int(mats.reshape(N_FRAMES, -1).sum(1).argmax())
    print("Кадр с максимальной энергией (грубая дальность):", n_star)
