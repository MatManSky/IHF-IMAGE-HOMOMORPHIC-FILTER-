"""
Модуль для обработки фотографий

    1. читает и записывает файл с фотографией в формате PNG/JPEG — read_gray, write_gray;
    2. поднимает яркость пикселов до MIN_VALUE, так как log2(0) не определён;
    3. exp2 после усиления ВЧ даёт значения далеко за 255, а в файл нужен
       8 бит — to_uint8 переводит значения в 0..255 способом _output_types
       (по умолчанию линейная растяжка).

Кадр дополняется до степени двойки (pad="reflect" по умолчанию), так как
это нужно для БПФ по методу Кули-Тьюки.
"""

import inspect

import numpy as np
from PIL import Image

import filtration
from filter import homomorphic_filter, MIN_VALUE

# Нижняя граница яркости MIN_VALUE объявлена в filter.py

# Границы растяжки диапазона
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

GAMMA_DEFAULT = 2.2


def read_gray(path) -> np.ndarray:
    """
    Прочитать изображение как одноканальную матрицу float64 формы (n, m).
    """
    with Image.open(path) as picture:
        values = np.asarray(picture.convert("L"), dtype=np.float64)
    return np.maximum(values, MIN_VALUE)


def write_gray(path, values: np.ndarray) -> None:
    """Записать матрицу uint8 формы (n, m)"""
    Image.fromarray(values, mode="L").save(path)


def _limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    # Границы диапазона: low/high — перцентили; None — min/max (выбросы не отсекаются).
    if low is None and high is None:
        return values.min(), values.max()
    if low is None:
        low = 0.0
    if high is None:
        high = 100.0
    return np.percentile(values, [low, high])


def _no_stretch(values: np.ndarray) -> np.ndarray:
    # Без изменений
    return values


def _percentile_stretch(values: np.ndarray,
                        low: float = LOW_PERCENTILE,
                        high: float = HIGH_PERCENTILE) -> np.ndarray:
    # Линейная растяжка отсечек low/high до 0..255.
    lo, hi = _limits(values, low, high)
    if hi <= lo:
        return values
    return (values - lo) * (255.0 / (hi - lo))


def _gamma_stretch(values: np.ndarray, gamma: float = GAMMA_DEFAULT,
                   low: float = None, high: float = None) -> np.ndarray:
    # Гамма-коррекция после нормировки
    lo, hi = _limits(values, low, high)
    if hi <= lo:
        return values
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return 255.0 * np.power(normalized, 1.0 / gamma)


_output_types = {
    "no": _no_stretch,
    "percentile": _percentile_stretch,
    "gamma": _gamma_stretch,
}

OUTPUT_DEFAULT = "percentile"


class photo_filter:
    """
    Пример использования:
        photo = read_gray("examples/pic1.png")
        f = photo_filter(lf_gain=0.75, hf_gain=2.0, d0=30)
        write_gray("examples/pic1_processed.png", f.to_uint8(f.apply(photo)))
    """

    def __init__(
        self,
        lf_filter: str = filtration.DEFAULT_LF_FILTER,
        hf_filter: str = filtration.DEFAULT_HF_FILTER,
        lf_gain: float = 1.0,
        hf_gain: float = 1.0,
        d0: float = filtration.D0_DEFAULT,
        order: int = filtration.ORDER_DEFAULT,
        pad: str = "reflect",
        window: str = None,
    ):
        self._filter = homomorphic_filter(
            lf_filter, hf_filter, lf_gain, hf_gain, d0, order
        )
        self._pad = pad
        self._window = window

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Применить гомоморфный фильтр к фотографии.

        Параметры:
            image: матрица (numpy.ndarray или путь к файлу)

        Возвращает:
            numpy.ndarray: результат (int64) формы входа; диапазон не
                           ограничен 0..255, поэтому нужен to_uint8
        """
        if not isinstance(image, np.ndarray):
            image = read_gray(image)
        image = np.maximum(np.asarray(image, dtype=np.float64), MIN_VALUE)
        return self._filter.apply(image, pad=self._pad, window=self._window)

    @staticmethod
    def to_uint8(values: np.ndarray, output: str = OUTPUT_DEFAULT,
                 **params) -> np.ndarray:
        """
        Перевести результат фильтра в 8 бит

        Параметры:
            values: матрица результата apply (диапазон не ограничен 0..255)
            output: "percentile" — линейная растяжка (по умолчанию),
                    "no" — только обрезка к 0..255,
                    "gamma" — гамма-коррекция после нормировки к [0, 1]
            **params: параметры выбранного режима:
                    "percentile": low, high — перцентили-отсечки;
                    "gamma": gamma, low, high.
        """
        if output not in _output_types:
            raise ValueError(
                f"Неизвестный режим вывода: '{output}'. "
                f"Доступные: {sorted(_output_types)}."
            )
        policy_of = _output_types[output]
        declared = inspect.signature(policy_of).parameters
        unknown = sorted(set(params) - set(declared))
        if unknown:
            raise ValueError(
                f"Режим '{output}' не принимает параметры: {unknown}. "
                f"Доступны: {sorted(declared)}."
            )
        scaled = policy_of(np.asarray(values, dtype=np.float64), **params)
        return np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)
