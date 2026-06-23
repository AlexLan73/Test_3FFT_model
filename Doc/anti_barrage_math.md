# 📐 Распознавание цели в заградительном шуме — полная версия с формулами

> Технический компаньон к `Doc/anti_barrage_intro.md` (версия «для чайника») и
> спеке `MemoryBank/specs/anti_barrage_detection_2026-06-23.md`.
> Вся нотация привязана к нашему коду radar3d.

---

## 1. Обозначения

| Символ | Смысл | В коде |
|--------|-------|--------|
| $n_x, n_y$ | элементы решётки по осям | `ArrayConfig(16,16)` |
| $M = n_x n_y$ | всего пространственных каналов | $=256$ |
| $(p,q)$ | индекс элемента, $p=0..n_x{-}1$ | `np.arange(nx)` |
| $(k_x,k_y)$ | угловой бин (азимут, место) | оси куба, $[-8..7]$ |
| $k$ | отсчёт быстрого времени (дальность) | $0..n_{real}{-}1$ |
| $N_f$ | длина БПФ по дальности | `n_fft=64` |
| $r$ | бин дальности цели | `range_bin` |
| $\mathbf a(k_x,k_y)\in\mathbb C^{M}$ | вектор наведения (steering) | `grid.steering` |
| $\mathbf R\in\mathbb C^{M\times M}$ | пространственная ковариация | — |

---

## 2. Сигнальная модель куба

### 2.1. Вектор наведения (steering)
Из `ArrayGrid.steering` фаза элемента $(p,q)$ для прихода с бина $(k_x,k_y)$:

$$
a_{p,q}(k_x,k_y)=\exp\!\Big(j2\pi\big(\tfrac{k_x\,p}{n_x}+\tfrac{k_y\,q}{n_y}\big)\Big),
\qquad \mathbf a=\operatorname{vec}\{a_{p,q}\}\in\mathbb C^{M}.
$$

Свойство: $\mathbf a^{H}\mathbf a = M$, а $\dfrac{1}{M}\mathbf a(\theta_1)^{H}\mathbf a(\theta_2)$ —
пространственная корреляция двух направлений (это и есть «насколько углы близки»).

### 2.2. Тон дальности
Из `_SteeredTone._tone` точечная цель на дальности $r$ даёт быстрый сигнал

$$
s[k]=A\,e^{j(2\pi f_r k+\varphi)},\qquad f_r=\frac{r}{N_f},\quad k=0..n_{real}{-}1,
$$

дополняется нулями до $N_f$ и БПФ $\Rightarrow$ острый пик на бине $r$ (sinc-отклик
шириной $\sim N_f/n_{real}$, у нас $64/16=4$ бина).

### 2.3. Снимок и куб
Снимок (один отсчёт $k$), стек по элементам $\mathbf x_k\in\mathbb C^M$:

$$
\mathbf x_k=\sum_i \alpha_i[k]\,\mathbf a(\theta_i)+\mathbf n_k,
\qquad \mathbf n_k\sim\mathcal{CN}(0,\sigma_n^2 \mathbf I).
$$

3D-БПФ (окна + $\text{fftn}$ + $\text{fftshift}$ по углам) даёт куб
$C[k_x,k_y,\text{range}]$ — `Fft3DModel`.

### 2.4. Цель vs barrage — в формулах
- **Точечная цель:** $\alpha_t[k]=A\,e^{j2\pi f_r k}$ — детерминирован, $|\alpha_t[k]|=A$.
  Локализован по углу ($\mathbf a(\theta_t)$) **и** по дальности (пик на $r$).
- **Заградительная (`BarrageJammer`):** $\mathbf x_k^{J}=\sqrt{P}\,\nu_k\,\mathbf a(\theta_J)$,
  где $\nu_k\sim\mathcal{CN}(0,1)$ — **белый по $k$**. Один угол $\theta_J$, но
  после БПФ по дальности белый шум $\Rightarrow$ **заливка всех бинов дальности**.

Это и есть наша асимметрия: цель — точка, помеха — столб по дальности.

---

## 3. Ковариация и собственная структура

Усредняя $\mathbf R=\frac1K\sum_k \mathbf x_k\mathbf x_k^{H}$ по быстрым отсчётам:

$$
\boxed{\;\mathbf R = A^2\,\mathbf a_t\mathbf a_t^{H}
\;+\;P\,\mathbf a_J\mathbf a_J^{H}\;+\;\sigma_n^2\mathbf I\;}
$$

($\mathbf a_t=\mathbf a(\theta_t)$, $\mathbf a_J=\mathbf a(\theta_J)$). Собственное
разложение $\mathbf R=\sum_m\lambda_m\mathbf e_m\mathbf e_m^{H}$ даёт:

$$
\lambda_1\!\approx\!PM+\sigma_n^2\ (\text{помеха}),\quad
\lambda_2\!\approx\!A^2M+\sigma_n^2\ (\text{цель}),\quad
\lambda_3{=}\dots{=}\lambda_M=\sigma_n^2\ (\text{шум}).
$$

Обычно $P\gg A^2$, поэтому **самое большое с.з. = помеха**, а её собственный
вектор $\mathbf e_1\approx \mathbf a_J/\sqrt M$. Критерий «это barrage»:

$$
\frac{\lambda_1}{\sigma_n^2}>\eta\quad\text{и}\quad
\text{разброс энергии по дальности велик (occupancy)}.
$$

---

## 4. Метод 1 — пространственное подавление ⭐ (есть всё в нашем кубе)

### 4.1. Ортогональная проекция (быстро, грубо)

$$
\mathbf P_J^{\perp}=\mathbf I-\mathbf e_1\mathbf e_1^{H}
\quad\Longrightarrow\quad \mathbf x_k'=\mathbf P_J^{\perp}\mathbf x_k .
$$

Помеха зануляется. Действие на цель:
$\mathbf P_J^{\perp}\mathbf a_t=\mathbf a_t-\mathbf e_1(\mathbf e_1^{H}\mathbf a_t)$.
Утечка $\mathbf e_1^{H}\mathbf a_t=\frac1{\sqrt M}\mathbf a_J^{H}\mathbf a_t$:
- углы **разнесены** $\Rightarrow$ утечка $\approx0$ $\Rightarrow$ цель сохранена ✅
- углы **совпали** (mainlobe) $\Rightarrow$ утечка $\approx1$ $\Rightarrow$ цель тоже гаснет ❌ (см. §6).

### 4.2. Косая проекция (oblique) — сохраняет цель точно
Чтобы не терять амплитуду цели, проецируем **вдоль** подпространства помехи на
подпространство цели:

$$
\boxed{\;\mathbf E_{t|J}=\mathbf a_t\big(\mathbf a_t^{H}\mathbf P_J^{\perp}\mathbf a_t\big)^{-1}\mathbf a_t^{H}\mathbf P_J^{\perp}\;}
$$

Свойства: $\mathbf E_{t|J}\,\mathbf a_t=\mathbf a_t$ (цель проходит без искажения),
$\mathbf E_{t|J}\,\mathbf a_J=\mathbf 0$ (помеха убита). Это метод из спеки
(PMC9654238). Для нескольких помех — стек $\mathbf A_J=[\mathbf a_{J_1},\dots]$ и
$\mathbf P_J^{\perp}=\mathbf I-\mathbf A_J(\mathbf A_J^{H}\mathbf A_J)^{-1}\mathbf A_J^{H}$.

### 4.3. Адаптивный луч MVDR (оптимум по SINR)

$$
\mathbf w=\frac{\mathbf R^{-1}\mathbf a_t}{\mathbf a_t^{H}\mathbf R^{-1}\mathbf a_t},
\qquad
\mathrm{SINR}_{\text{out}}=A^2\,\mathbf a_t^{H}\mathbf R_{i+n}^{-1}\mathbf a_t,
$$

где $\mathbf R_{i+n}=P\,\mathbf a_J\mathbf a_J^{H}+\sigma_n^2\mathbf I$. По лемме
Шермана–Моррисона ставит **глубокий ноль** в $\theta_J$, сохраняя единичный
коэффициент усиления на $\theta_t$ ($\mathbf w^H\mathbf a_t=1$).

> 🖥️ Всё это — BLAS: $\mathbf R$ (256×256), EVD/обращение, проекции —
> матричные операции, ложатся на целевые GPU-ядра.

---

## 5. Метод 2 — Доплер (определение скорости) 🎵

### 5.1. Что добавляем в модель
Нужна **ось медленного времени** — пакет из $L$ импульсов (PRI $T_r$). Летящая
цель с радиальной скоростью $v_r$ даёт доплеровский сдвиг

$$
f_d=\frac{2 v_r}{\lambda},
$$

и фазу, нарастающую **от импульса к импульсу** $l=0..L{-}1$:

$$
x[p,q,k,l]=A\,a_{p,q}(\theta_t)\,s[k]\,\underbrace{e^{\,j2\pi f_d T_r l}}_{\text{доплер}}+\dots
$$

Доплеровский вектор и БПФ по $l$:

$$
\mathbf d(f_d)=\big[1,\,e^{j2\pi f_d T_r},\dots,e^{j2\pi f_d T_r (L-1)}\big]^{T},
\qquad
X[\cdot,f]=\sum_{l=0}^{L-1}x[\cdot,l]\,e^{-j2\pi fl/L}.
$$

### 5.2. Почему это разделяет цель и помеху
- **Цель** когерентна по $l$ $\Rightarrow$ суммируется в фазе $\Rightarrow$ **пик**
  на бине $f_d$ с **интеграционным выигрышем** $L$:
  $$\mathrm{SNR}_{\text{после}}=L\cdot\mathrm{SNR}_{\text{до}}.$$
- **Barrage** независим по $l$ ($\nu_l$ белый) $\Rightarrow$ энергия **ровно
  размазана по всем $L$ доплеровским бинам**, в каждом лишь $1/L$ доли. Пика нет.

$$
\boxed{\ \text{выигрыш цель/помеха по Доплеру}\;\approx\;L\ }
$$

Итог: даже если помеха **с того же угла** (где §4 бессилен), **по скорости** цель
(острая «нота» $f_d$) и шум («белое шипение») разделяются. Это прямой признак
**летящего** объекта. Куб становится 4D: $(k_x,k_y,\text{range},f_d)$.

### 5.3. STAP — совместно угол×Доплер (максимум)
Пространственно-временной снимок $\boldsymbol\chi=\operatorname{vec}\{M\times L\}\in\mathbb C^{ML}$,
пространственно-временной steering:

$$
\mathbf v(\theta,f_d)=\mathbf a(\theta)\otimes\mathbf d(f_d),
\qquad
\mathbf w_{\text{STAP}}=\frac{\mathbf R_{st}^{-1}\mathbf v(\theta_t,f_d)}{\mathbf v^{H}\mathbf R_{st}^{-1}\mathbf v}.
$$

Ковариация barrage: $\mathbf R_{st}^{J}=P\,(\mathbf a_J\mathbf a_J^{H})\otimes\mathbf I_L$
(один угол, белый во времени) $\Rightarrow$ STAP ставит **угловой ноль во ВСЕХ
доплерах**, сохраняя цель в её $(\theta_t,f_d)$.

⚠️ Цена: $\mathbf R_{st}$ размером $ML\times ML$ (у нас $256L$), правило RMB —
нужно $\ge 2ML$ обучающих выборок для $\sim$3 дБ потерь SINR. Поэтому на практике —
**reduced-dimension STAP** (бимспейс/факторизованный), §7.

---

## 6. Когда §4 не работает — mainlobe-помеха
Если $\theta_J\to\theta_t$, то $\mathbf a_J^H\mathbf a_t/M\to1$: проекция убивает и
цель. Лечится (спека, IET 2024):
1. точный DOA помехи через MUSIC: $P_{\text{MU}}(\theta)=\big(\mathbf a^H\mathbf U_n\mathbf U_n^H\mathbf a\big)^{-1}$;
2. реконструкция ковариации помехи только в области главного луча;
3. **null-constraints** + восстановление главного луча.

Либо — **Доплер (§5)**: по скорости разделит даже при одном угле. Либо — **3D-CNN (§8)**.

---

## 7. Метод 3 — CFAR-детектор (после подавления)
После §4/§5 — порог по уровню шума. CA-CFAR по $N$ опорным ячейкам:

$$
T=\alpha\cdot\frac1N\sum_{i=1}^{N}P_i,\qquad
\alpha=N\big(P_{fa}^{-1/N}-1\big),
$$

решение «цель», если тестовая ячейка $>T$. $P_{fa}$ — заданная вероятность ложной
тревоги. Локализованный пик (цель) проходит порог, заливка (остаток помехи) — нет.

---

## 8. Метод 4 — 3D-CNN / денойзер (сложные случаи)

### 8.1. Денойзер куба (аналог IET RSN 2024)
Вход — «грязный» куб $\mathbf X$ (цель+помеха), сеть $g_\theta$ (encoder-decoder на
`Conv3d`) восстанавливает «чистый» куб либо маску цели:

$$
\mathcal L_{\text{denoise}}=\big\|g_\theta(\mathbf X_{\text{noisy}})-\mathbf X_{\text{clean}}\big\|_F^2 .
$$

### 8.2. Лёгкая свёртка (depthwise separable) — важно для GPU-каскада
Обычная 3D-свёртка ядром $K^3$: стоимость $K^3 C_{in}C_{out}$.
Depthwise+pointwise: $K^3 C_{in}+C_{in}C_{out}$ — дешевле в $\approx C_{out}$ раз при
тех же рецептивных полях. Отсюда «lightweight» сети для real-time.

### 8.3. Классификатор (наш `Cnn3DClassifier`, LSP к RuleBased)

$$
\hat y=\arg\max_c\,\mathrm{softmax}(f_\theta(\mathbf X))_c,\qquad
\mathcal L_{\text{CE}}=-\sum_c y_c\log\hat p_c .
$$

Берёт mainlobe/low-SNR, где детерминированные методы не уверены.

---

## 9. Привязка к коду и что реализовать

| Метод | Формула-ядро | Где в проекте | Готовность |
|-------|--------------|---------------|------------|
| Пространств. нуллинг §4 | $\mathbf E_{t|J},\ \mathbf P_J^{\perp}$ | новый `SubspaceNuller` в `core/models/` | ✅ куб готов |
| Признак локализации | occupancy, $\lambda_1/\sigma_n^2$ | усилить `RuleBasedClassifier` | ✅ |
| Доплер §5 | $\mathbf d(f_d)$, БПФ по $l$ | +ось импульсов: `RangeConfig`→`+n_pulses`, новый `_tone` с $v_r$, БПФ-4D | 🔧 расширение модели |
| STAP §5.3 | $\mathbf w_{st}=\mathbf R_{st}^{-1}\mathbf v$ | после Доплера, reduced-dim | 🔧 |
| CFAR §7 | $T=\alpha\bar P$ | детектор после нуллинга | 🔧 малый |
| 3D-CNN §8 | $\mathcal L_{CE}/\mathcal L_{denoise}$ | `Cnn3DClassifier` (есть скелет) | 🔧 обучение |

**Порядок (работает→корректно→быстро):**
1. `SubspaceNuller` (§4.2 oblique) + критерий $\lambda_1/\sigma_n^2$ — детерминир.,
   разделит target↔barrage по углу **уже на текущем кубе**.
2. CFAR (§7) — простой детектор пика.
3. **Доплер (§5)** — добавить пакет импульсов; даёт признак «летящий» по скорости,
   разделяет даже mainlobe-помеху.
4. 3D-CNN (§8) — арбитр сложных случаев.

---
*Версия «на пальцах» — `Doc/anti_barrage_intro.md`. Источники и патенты —
`MemoryBank/specs/anti_barrage_detection_2026-06-23.md`.*
