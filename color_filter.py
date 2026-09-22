"""
Гомоморфный фильтр для цветных изображений (3 канала: R, G, B).
"""

import numpy as np

import filtration
from filter import homomorphic_filter

# Число каналов цветного изображения (в RGB = 3)
CHANNELS = 3

class color_homomorphic_filter:
    """
    Пример использования:
    f = color_homomorphic_filter(lf_gain=0.5, hf_gain=2.0)
    result = f.apply(color_image)  # color_image: (n, m, 3)
    """

    def __init__(
        self,
        lf_filter: str = filtration.DEFAULT_LF_FILTER,
        hf_filter: str = filtration.DEFAULT_HF_FILTER,
        lf_gain: float = 1.0,
        hf_gain: float = 1.0,
        d0: float = filtration.D0_DEFAULT,
        order: int = filtration.ORDER_DEFAULT,
        pad: str = None,
        window: str = None,
    ):
        self._filter = homomorphic_filter(
            lf_filter, hf_filter, lf_gain, hf_gain, d0, order)
        self._pad = pad
        self._window = window

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Применить гомоморфный фильтр к цветному изображению.

        Параметры:
            image: входная матрица (numpy.ndarray, shape (n, m, 3),
                   значения > 0; dtype — float64 или int64)

        Возвращает:
            numpy.ndarray: результат (int64, shape (n, m, 3))

        Исключения:
            ValueError: если форма входной матрицы не (n, m, 3)
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != CHANNELS:
            raise ValueError(
                f"Ожидается цветное изображение формы (n, m, {CHANNELS}), "
                f"получена форма {image.shape}"
            )
        channels_out = [
            self._filter.apply(image[:, :, k], pad=self._pad,
                               window=self._window)
            for k in range(CHANNELS)
        ]
        return np.stack(channels_out, axis=2)
