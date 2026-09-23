"""
Модуль для обработки фотографий

    1. читает и записывает файл с фотографией в формате PNG/JPEG — read_gray, write_gray;
    2. поднимает яркость пикселов до MIN_VALUE, так как log2(0) не определён;
    3. exp2 после усиления ВЧ даёт значения далеко за 255, а в файл нужен
       8 бит — to_uint8 растягивает диапазон.

Кадр дополняется до степени двойки (pad="reflect" по умолчанию), так как
это нужно для БПФ по методу Кули-Тьюки.
"""

import numpy as np
from PIL import Image

import filtration
from filter import homomorphic_filter

# Нижняя граница яркости
MIN_VALUE = 1.0

# Границы растяжки диапазона
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5


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
    def to_uint8(values: np.ndarray) -> np.ndarray:
        """
        Перевести результат фильтра в 8 бит: линейная растяжка по 
        LOW/HIGH_PERCENTILE и округление.
        """
        lo = np.percentile(values, LOW_PERCENTILE)
        hi = np.percentile(values, HIGH_PERCENTILE)
        scaled = (values - lo) * (255.0 / (hi - lo))
        return np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)
