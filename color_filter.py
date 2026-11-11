"""
Гомоморфный фильтр для цветных изображений.

Кадр (n, m, 3) — RGB: три канала фильтруются независимо; (n, m, 4) —
RGBA: четвёртый канал (альфа) проходит без изменений.
Каналы считаются последовательно или параллельно через пул:
parallel=True — только параллельно, parallel=None —
автоматически от порога _PARALLEL_MIN_PIXELS, parallel=False — всегда
последовательно.

Режимы (mode):
    "ycbcr" (по умолчанию) — кадр переводится в YCbCr (ITU-R BT.601)
        в MPFR, фильтруется только яркость Y, кадр переводится обратно;
        усиления должны быть скалярами;
    "rgb" — поканальная фильтрация, усиления задаются по каналам.

Ссылки:
 1. Красильников Н.Н. Цифровая обработка 2D- и 3D-изображений: учеб.
    пособие. СПб.: БХВ-Петербург, 2011. 608 с.
 2. ITU-R BT.601-7. Encoding-Parameter Digital Values of Constant
    Luminance in the Component Digital Television Interface. 2011.
"""

import multiprocessing
import threading

import numpy as np
import gmpy2
from gmpy2 import mpfr
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import filtration
from filter import homomorphic_filter

# Число цветовых каналов (RGB); альфа идёт четвёртым каналом
CHANNELS = 3

# Порог автопараллельности: пикселов на канал (128x128 = 2^14)
_PARALLEL_MIN_PIXELS = 16384

# Допустимые режимы обработки цвета
_COLOR_MODES = ("rgb", "ycbcr")

_POOL = None
_POOL_LOCK = threading.Lock()

# Коэффициенты BT.601
_K_R = mpfr("0.299")
_K_G = mpfr("0.587")
_K_B = mpfr("0.114")
_HALF = mpfr("0.5")
_TWO = mpfr(2)
_OFF = mpfr(128)


# Преобразование между RGB и YCbCr (ITU-R BT.601)
def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Преобразовать RGB в YCbCr по ITU-R BT.601 (full range).
    (float64 используется для тестов, есть ещё mpfr)
    Вход: float64 (n, m, 3), значения R, G, B ≥ 0.
    Возвращает: float64 (n, m, 3) — Y, Cb, Cr.
    Y ∈ [0, 255], Cb/Cr ∈ [0, 256], 128 = нейтральный цвет.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 * (b - y) + 128
    cr = 0.5 * (r - y) + 128
    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Обратное преобразование YCbCr → RGB по ITU-R BT.601.

    Вход: float64 (n, m, 3) — Y, Cb, Cr.
    Возвращает: float64 (n, m, 3) — R, G, B.
    """
    y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    r = y + 2 * (cr - 128)
    b = y + 2 * (cb - 128)
    g = y - (0.299 * (r - y) + 0.114 * (b - y)) / 0.587
    return np.stack([r, g, b], axis=2)


def _rgb_to_ycbcr_mpfr(rgb: np.ndarray) -> list:
    # (n, m, 3) float64 -> (n, m) кортежей mpfr (Y, Cb, Cr).
    out = []
    for row in np.asarray(rgb, dtype=np.float64):
        line = []
        for r, g, b in row:
            mr, mg, mb = mpfr(r), mpfr(g), mpfr(b)
            y = _K_R * mr + _K_G * mg + _K_B * mb
            line.append((y, _HALF * (mb - y) + _OFF,
                         _HALF * (mr - y) + _OFF))
        out.append(line)
    return out


def _y_of_ycbcr(ycbcr: list) -> list:
    # Матрица Y из списка кортежей (Y, Cb, Cr) — вход для фильтра
    return [[px[0] for px in line] for line in ycbcr]


def _merge_y(ycbcr: list, y_new: list) -> list:
    # Заменить Y на отфильтрованный, сохранив Cb/Cr
    return [[(yv, px[1], px[2]) for px, yv in zip(line, ynew)]
            for line, ynew in zip(ycbcr, y_new)]


def _ycbcr_to_rgb_mpfr(ycbcr: list) -> np.ndarray:
    # (n, m) кортежей mpfr (Y, Cb, Cr) -> int64 (n, m, 3).
    floor = gmpy2.floor
    half = _HALF
    out = []
    for line in ycbcr:
        row = []
        for y, cb, cr in line:
            r = y + _TWO * (cr - _OFF)
            b = y + _TWO * (cb - _OFF)
            g = y - (_K_R * (r - y) + _K_B * (b - y)) / _K_G
            row.append((int(floor(r + half)), int(floor(g + half)),
                        int(floor(b + half))))
        out.append(row)
    return np.array(out, dtype=np.int64)


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
        # RGB-режим (поканальный, по умолчанию):
        f = color_homomorphic_filter(lf_gain=(0.99, 1.0, 1.01), hf_gain=1.0)
        result = f.apply(color_image)  # (n, m, 3); RGBA вернёт (n, m, 4)

        # YCbCr-режим (фильтруется только яркость Y:
        f = color_homomorphic_filter(lf_gain=0.75, hf_gain=2.0,
                                     mode="ycbcr")
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
        mode: str = "rgb",
    ):
        if mode not in _COLOR_MODES:
            raise ValueError(
                f"Неизвестный режим цвета: '{mode}'. "
                f"Доступные: {sorted(_COLOR_MODES)}."
            )
        self._lf_filter = lf_filter
        self._hf_filter = hf_filter
        self._lf_gain = _channels("lf_gain", lf_gain)
        self._hf_gain = _channels("hf_gain", hf_gain)
        self._d0 = d0
        self._order = order
        self._pad = pad
        self._window = window
        self._mode = mode
        # В ycbcr-режиме усиления применяются только к Y-каналу
        if mode == "ycbcr":
            for name, gains in (("lf_gain", self._lf_gain),
                                ("hf_gain", self._hf_gain)):
                if len(set(gains)) != 1:
                    raise ValueError(
                        f"В ycbcr-режиме {name} должен быть скаляром "
                        f"(применяется к Y-каналу), получено {gains}."
                    )
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
            parallel: None — автоматически, True — параллельно,
                      False — последовательно (для ycbcr-режима не
                      используется, всегда один канал)

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
        if self._mode == "ycbcr":
            result = self._apply_ycbcr(rgb)
        else:
            result = self._apply_rgb(rgb, parallel)
        if image.shape[2] == CHANNELS + 1:
            alpha = image[:, :, CHANNELS].astype(np.float64)
            alpha = np.floor(alpha + 0.5).astype(result.dtype)
            result = np.concatenate(
                [result, alpha[:, :, np.newaxis]], axis=2)
        return result

    def _apply_ycbcr(self, rgb: np.ndarray) -> np.ndarray:
        # Фильтруется только Y, Cb/Cr не изменяются.
        ycc = _rgb_to_ycbcr_mpfr(rgb)
        lf, hf = self._lf_gain[0], self._hf_gain[0]
        single = self._filters[(lf, hf)]
        y_new = single.apply(_y_of_ycbcr(ycc), pad=self._pad,
                             window=self._window, as_float=True)
        return _ycbcr_to_rgb_mpfr(_merge_y(ycc, y_new))

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
