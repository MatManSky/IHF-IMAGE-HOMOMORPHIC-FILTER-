"""
Прикладные операции над изображениями.

Температура освещения («холоднее» / «теплее») меняется в гомоморфном
фильтре: усиливается только НЧ по каналам (R, G, B) после разделения
спектра на НЧ и ВЧ и до их сложения:
I = E·R -> log2 -> БПФ -> НЧ·lf_gain + ВЧ·hf_gain -> ОБПФ -> exp2

«Холоднее»: уменьшить R (lf_gain < 1), увеличитиь B (lf_gain > 1).
"""

import numpy as np

import filtration
from color_filter import color_homomorphic_filter


def temperature_gains(temperature: float) -> tuple:
    # Усиления НЧ по каналам (R, G, B) для сдвига температуры освещения
    if not -1.0 < temperature < 1.0:
        raise ValueError(
            f"temperature={temperature}: ожидается -1 < temperature < 1, "
            f"иначе усиление НЧ станет неположительным"
        )
    return (1.0 - temperature, 1.0, 1.0 + temperature)


def shift_temperature(
    image: np.ndarray,
    temperature: float = 0.01,
    lf_filter: str = filtration.DEFAULT_LF_FILTER,
    hf_filter: str = filtration.DEFAULT_HF_FILTER,
    hf_gain: float = 1.0,
    d0: float = filtration.D0_DEFAULT,
    order: int = filtration.ORDER_DEFAULT,
    pad: str = "reflect",
    window: str = None,
) -> np.ndarray:
    """
    Сдвинуть температуру освещения цветного изображения.

    Параметры:
        image: цветная матрица (n, m, 3), значения > 0
        temperature: > 0 — холоднее, < 0 — теплее (|temperature| < 1)
        остальные: как у color_homomorphic_filter (фильтры, hf_gain,
                   d0, order, pad, window)

    Возвращает:
        numpy.ndarray: int64 формы (n, m, 3); диапазон не ограничен 0..255,
        поэтому перед записью нужен photo_filter.to_uint8
    """
    f = color_homomorphic_filter(
        lf_filter, hf_filter, temperature_gains(temperature), hf_gain,
        d0, order, pad, window,
    )
    return f.apply(image)


def make_colder(image: np.ndarray, strength: float = 0.01, **params) -> np.ndarray:
    # Сделать освещение холоднее (R уменишить, B увеличить)
    if strength < 0:
        raise ValueError(
            f"strength={strength}: ожидается >= 0; для теплее — make_warmer"
        )
    return shift_temperature(image, strength, **params)


def make_warmer(image: np.ndarray, strength: float = 0.01, **params) -> np.ndarray:
    # Сделать освещение теплее (R увеличить, B уменьшить)
    if strength < 0:
        raise ValueError(
            f"strength={strength}: ожидается >= 0; для холоднее — make_colder"
        )
    return shift_temperature(image, -strength, **params)


def balance(values: np.ndarray) -> str:
    # Средние по каналам R, G, B и холодность изображения (B - R)
    means = [float(values[:, :, k].mean()) for k in range(3)]
    return (f"R={means[0]:6.2f} G={means[1]:6.2f} B={means[2]:6.2f}  "
            f"B-R={means[2] - means[0]:+6.2f}")
