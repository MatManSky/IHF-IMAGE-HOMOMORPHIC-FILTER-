"""
Гомоморфный фильтр для цветных изображений.

Кадр (n, m, 3) — RGB: три канала фильтруются независимо; (n, m, 4) —
RGBA: четвёртый канал (альфа) проходит без изменений. 
Каналы считаются последовательно или параллельно через пул: 
parallel=True — только параллельно, parallel=None —
автоматически от порога _PARALLEL_MIN_PIXELS, parallel=False — всегда
последовательно.

Ссылки:
 1. Красильников Н.Н. Цифровая обработка 2D- и 3D-изображений: учеб.
    пособие. СПб.: БХВ-Петербург, 2011. 608 с.
"""

import multiprocessing
import threading

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import filtration
from filter import homomorphic_filter

# Число цветовых каналов (RGB); альфа идёт четвёртым каналом
CHANNELS = 3

# Порог автопараллельности: пикселов на канал (128x128 = 2^14)
_PARALLEL_MIN_PIXELS = 16384

_POOL = None
_POOL_LOCK = threading.Lock()


def _filter_channel(job) -> np.ndarray:
    # Рабочая функция пула: фильтрация одного канала в дочернем процессе.
    (channel, lf_filter, hf_filter, lf_gain, hf_gain, d0, order,
     pad, window) = job
    single = homomorphic_filter(lf_filter, hf_filter, lf_gain, hf_gain,
                                d0, order)
    return single.apply(channel, pad=pad, window=window)


def _pool() -> ProcessPoolExecutor:
    # Общий пул: три процесса, по одному на канал.
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            context = multiprocessing.get_context("spawn")
            _POOL = ProcessPoolExecutor(CHANNELS, mp_context=context)
        return _POOL


def _drop_pool() -> None:
    # Пул сломался: сбросить, следующий вызов создаст новый
    global _POOL
    with _POOL_LOCK:
        _POOL = None


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
    result = f.apply(color_image)  # (n, m, 3); RGBA вернёт (n, m, 4)
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
        self._lf_filter = lf_filter
        self._hf_filter = hf_filter
        self._lf_gain = _channels("lf_gain", lf_gain)
        self._hf_gain = _channels("hf_gain", hf_gain)
        self._d0 = d0
        self._order = order
        self._pad = pad
        self._window = window
        # Итоговое усиление
        self._filters = {}
        for lf, hf in zip(self._lf_gain, self._hf_gain):
            if (lf, hf) not in self._filters:
                self._filters[(lf, hf)] = homomorphic_filter(
                    lf_filter, hf_filter, lf, hf, d0, order)

    def apply(self, image: np.ndarray,
              parallel: bool = None) -> np.ndarray:
        """
        Применить гомоморфный фильтр к цветному изображению.

        Параметры:
            image: входная матрица (numpy.ndarray, shape (n, m, 3) или
                   (n, m, 4); для RGBA фильтруются только цветовые
                   каналы, альфа возвращается без изменений; значения
                   цветовых каналов > 0; dtype — float64 или int64)
parallel: None — автоматически, True — паралельно, False — последовательно.

        Возвращает:
            numpy.ndarray: результат (int64, формы входа); альфа
            приводится к int64 округлением к ближайшему целому

        Исключения:
            ValueError: если форма входной матрицы не (n, m, 3) и не
                        (n, m, 4)
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] not in (CHANNELS,
                                                     CHANNELS + 1):
            raise ValueError(
                f"Ожидается цветное изображение формы (n, m, {CHANNELS}) "
                f"или (n, m, {CHANNELS + 1}), получена форма {image.shape}"
            )
        rgb = image[:, :, :CHANNELS]
        result = self._apply_rgb(rgb, parallel)
        if image.shape[2] == CHANNELS + 1:
            alpha = image[:, :, CHANNELS].astype(np.float64)
            alpha = np.floor(alpha + 0.5).astype(result.dtype)
            result = np.concatenate(
                [result, alpha[:, :, np.newaxis]], axis=2)
        return result

    def _apply_rgb(self, rgb: np.ndarray,
                   parallel: bool = None) -> np.ndarray:
        # Три цветовых канала: последовательно или пулом процессов
        if parallel is None:
            parallel = rgb.shape[0] * rgb.shape[1] >= _PARALLEL_MIN_PIXELS
        if not parallel:
            return self._apply_channels([
                self._filters[(self._lf_gain[k], self._hf_gain[k])].apply(
                    rgb[:, :, k], pad=self._pad, window=self._window)
                for k in range(CHANNELS)
            ])
        jobs = [
            (rgb[:, :, k], self._lf_filter, self._hf_filter,
             self._lf_gain[k], self._hf_gain[k], self._d0, self._order,
             self._pad, self._window)
            for k in range(CHANNELS)
        ]
        try:
            executor = _pool()
            channels = list(executor.map(_filter_channel, jobs))
        except (BrokenProcessPool, OSError):
            # Пул недоступен, считаем последовательно
            _drop_pool()
            return self._apply_rgb(rgb, parallel=False)
        return self._apply_channels(channels)

    def _apply_channels(self, channels) -> np.ndarray:
        # Собрать посчитанные каналы в цветной кадр (int64)
        return np.stack(list(channels), axis=2)
