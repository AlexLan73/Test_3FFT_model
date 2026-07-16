"""Триаж прохода 1 -- {noise, source, smeared} по 6 признакам (гл.4 §4.8, §4.11).

Патент предполагает обучаемую MLP 6->16->3 (гл.4 §4.8); torch-free P1-прототип
заменяет её детерминированным RuleBased по разделяющим порогам таблицы §4.11
(осознанная девиация E6 таска -- MLP включается позже той же абстракцией, LSP).

⚠️ Честная девиация (для ревью Кодо): `CubeClassifier` (`core/models/classification/
classifier.py`) классифицирует ЦЕЛЫЙ куб (`classify(cube) -> Classification`, метки
{empty,target,barrage,comb,ham}). Здесь вход -- вектор признаков ОДНОГО среза/окна
(`FeatureVector`), а не куб, и метки другие ({noise,source,smeared}, гл.4 §4.8).
Прямое наследование от `CubeClassifier` нарушило бы LSP (другой контракт входа/
выхода) -- вместо этого заведён параллельный маленький ABC `SliceTriage` той же
Strategy-формы (`classify(f) -> (label, score)`), готовый для будущей MLP-реализации
без изменения вызывающего кода (тот же принцип LSP, другая иерархия).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .features import FeatureVector

NOISE = "noise"
SOURCE = "source"
SMEARED = "smeared"


class SliceTriage(ABC):
    """Strategy: вектор признаков среза/окна -> (метка, скор). LSP-совместимо с MLP."""

    @abstractmethod
    def classify(self, f: FeatureVector) -> tuple[str, float]:
        ...


@dataclass(frozen=True)
class _Anchor:
    label: str
    log_pr: float
    hoyer: float
    main_frac: float
    log1p_lobe: float


class RuleBasedTriage(SliceTriage):
    """Классификация по ближайшему якорю (§4.11) в нормированном пространстве признаков.

    Строгие AND-пороги на сырых значениях хрупки у границ (страддл, шум оценки);
    вместо этого f нормируется и сравнивается с тремя якорями таблицы §4.11 --
    устойчивее, тот же физический смысл ("главные якоря PR/Hoyer/MainFrac/LobeRatio").
    `MaxMean`/`Energy` -- вспомогательные, в дистанцию не входят (§4.5: "вспомогательные").
    """

    # Якоря из таблицы §4.11 ("Три класса (источник на сетке)").
    _ANCHORS: tuple[_Anchor, ...] = (
        _Anchor(SOURCE,  log_pr=math.log10(3.6),  hoyer=0.94, main_frac=0.98,
                log1p_lobe=math.log1p(0.002)),
        _Anchor(SMEARED, log_pr=math.log10(19.0), hoyer=0.81, main_frac=0.40,
                log1p_lobe=math.log1p(0.25)),
        _Anchor(NOISE,   log_pr=math.log10(129.0), hoyer=0.31, main_frac=0.07,
                log1p_lobe=math.log1p(1.03)),
    )

    # Нормирующие масштабы (порядок типичного разноса соседних классов, §4.11).
    _SCALE_LOG_PR = 0.45
    _SCALE_HOYER = 0.20
    _SCALE_MAIN_FRAC = 0.28
    _SCALE_LOG1P_LOBE = 0.18

    def classify(self, f: FeatureVector) -> tuple[str, float]:
        log_pr = math.log10(max(f.pr, 1e-9))
        log1p_lobe = math.log1p(max(f.lobe_ratio, 0.0))

        dists: dict[str, float] = {}
        for anchor in self._ANCHORS:
            d2 = (
                ((log_pr - anchor.log_pr) / self._SCALE_LOG_PR) ** 2
                + ((f.hoyer - anchor.hoyer) / self._SCALE_HOYER) ** 2
                + ((f.main_frac - anchor.main_frac) / self._SCALE_MAIN_FRAC) ** 2
                + ((log1p_lobe - anchor.log1p_lobe) / self._SCALE_LOG1P_LOBE) ** 2
            )
            dists[anchor.label] = math.sqrt(d2)

        best_label = min(dists, key=lambda k: dists[k])
        best_d = dists[best_label]
        others = sorted(v for k, v in dists.items() if k != best_label)
        # Скор -- относительный отрыв от второго по близости якоря (0..1).
        second_d = others[0] if others else best_d
        score = float(second_d / (best_d + second_d)) if (best_d + second_d) > 0 else 0.5
        return best_label, score
