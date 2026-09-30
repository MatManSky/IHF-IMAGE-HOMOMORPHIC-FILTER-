"""
Гомоморфный фильтр для цветных изображений (3 канала: R, G, B).
"""

import numpy as np

import filtration
from filter import homomorphic_filter

# Число каналов цветного изображения (в RGB = 3)
CHANNELS = 3


def _channels(name: str, value) -> tuple:
    # Усиления по каналам: скаляр задаётся всем (R, G, B).
    try:
        values = tuple(float(v) for v in value)
    except TypeError:
        return (float(value),) * CHANNELS
    if len(values) != CHANNELS:
        raise ValueError(
            f"{name}: ожидается скаляр или {CHANNELS} значения (R, G, B), "
            f"получено {len(values)}"
        )
    return values


class color_homomorphic_filter:
    """
    Пример использования:
    f = color_homomorphic_filter(lf_gain=(0.99, 1.0, 1.01), hf_gain=1.0)
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
        self._lf_gain = _channels("lf_gain", lf_gain)
        self._hf_gain = _channels("hf_gain", hf_gain)
        self._pad = pad
        self._window = window
        # Итоговое усиление
        self._filters = {}
        for lf, hf in zip(self._lf_gain, self._hf_gain):
            if (lf, hf) not in self._filters:
                self._filters[(lf, hf)] = homomorphic_filter(
                    lf_filter, hf_filter, lf, hf, d0, order)

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
            self._filters[(self._lf_gain[k], self._hf_gain[k])].apply(
                image[:, :, k], pad=self._pad, window=self._window)
            for k in range(CHANNELS)
        ]
        return np.stack(channels_out, axis=2)
