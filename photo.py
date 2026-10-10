"""
Модуль для обработки фотографий

    1. читает и записывает файл с фотографией в формате PNG/JPEG — read_gray,
       write_gray для ч/б кадра и read_rgb, write_rgb для цветного;
    2. поднимает яркость пикселов до MIN_VALUE, так как log2(0) не определён;
    3. exp2 после усиления ВЧ даёт значения далеко за 255, а в файл нужен
       фиксированный диапазон — to_uint8, to_uint10, to_uint12, to_uint16
       переводят значения в 0..2^bits-1 способом _output_types (по умолчанию
       линейная растяжка);
    4. по желанию убирает шум готового кадра — denoise (билатерный фильтр),
       выключен по умолчанию.

Кадр дополняется до степени двойки (pad="reflect" по умолчанию), так как
это нужно для БПФ по методу Кули-Тьюки.
"""


import inspect

import numpy as np
from PIL import Image

import filtration
from color_filter import CHANNELS # CHANNELS = 3
from filter import homomorphic_filter, MIN_VALUE # MIN_VALUE = 1.0


# Границы растяжки диапазона
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

GAMMA_DEFAULT = 2.2

# Разрядность вывода: bits -> верх диапазона 0..MAX_UINT[bits].
# 10- и 12-битного PNG нет, поэтому dtype для таких — uint16.
# 16-битные TIFF и JPEG 2000 поддерживаются без проблем.
MAX_UINT = {8: 255, 10: 1023, 12: 4095, 16: 65535}
BITS_DEFAULT = 8


def read_gray(path, bits: int = BITS_DEFAULT) -> np.ndarray:
    # Прочитать изображение как одноканальную матрицу float64 формы (n, m)
    if bits not in MAX_UINT:
        raise ValueError(
            f"Неизвестная разрядность: {bits}. Доступные: {sorted(MAX_UINT)}."
        )
    with Image.open(path) as picture:
        if picture.mode == "I;16":
            values = np.asarray(picture, dtype=np.uint16)
            if picture.format == "PNG" and bits < 16:
                values = values >> (16 - bits)
            values = values.astype(np.float64)
        else:
            values = np.asarray(picture.convert("L"), dtype=np.float64)
    return np.maximum(values, MIN_VALUE)


SHIFT_TO_TOP = (".PNG",)


def write_gray(path, values: np.ndarray, bits: int = BITS_DEFAULT) -> None:
    """
    Записать матрицу яркости в файл.

    Параметры:
        path: путь к файлу (расширение выбирает контейнер)
        values: результат to_uint8 (uint8) или to_uint10/to_uint12/to_uint16
        bits: разрядность значений values — 8, 12 или 16
    """
    if bits not in MAX_UINT:
        raise ValueError(
            f"Неизвестная разрядность: {bits}. Доступные: {sorted(MAX_UINT)}."
        )
    if bits == 8:
        Image.fromarray(values, mode="L").save(path)
        return
    values = np.clip(np.asarray(values), 0, MAX_UINT[bits]).astype(np.uint16)
    if str(path).upper().endswith(SHIFT_TO_TOP):
        values = values << (16 - bits)
    Image.fromarray(values).save(path)


def read_rgb(path) -> np.ndarray:
    # Прочитать изображение как трёхканальную матрицу float64 формы (n, m, 3).
    with Image.open(path) as picture:
        values = np.asarray(picture.convert("RGB"), dtype=np.float64)
    return np.maximum(values, MIN_VALUE)


def write_rgb(path, values: np.ndarray) -> None:
    # Записать матрицу uint8 для RGB (n, m, 3). Цветной кадр
    # записывается по 8 бит на канал (8*3=24)
    values = np.asarray(values)
    if values.dtype != np.uint8:
        raise ValueError(
            f"Цветной кадр записывается в 8 бит,"
            f"передан dtype {values.dtype}: "
            f"нужен photo_filter.to_uint8"
        )
    Image.fromarray(values, mode="RGB").save(path)


def _limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    # Границы диапазона: low/high — перцентили; None — min/max (выбросы не отсекаются).
    if low is None and high is None:
        return values.min(), values.max()
    if low is None:
        low = 0.0
    if high is None:
        high = 100.0
    return np.percentile(values, [low, high])


def _no_stretch(values: np.ndarray, maximum: float) -> np.ndarray:
    # Без изменений
    return values


def _percentile_stretch(values: np.ndarray, maximum: float,
                        low: float = LOW_PERCENTILE,
                        high: float = HIGH_PERCENTILE) -> np.ndarray:
    # Линейная растяжка отсечек low/high до 0..maximum.
    lo, hi = _limits(values, low, high)
    if hi <= lo:
        return values
    return (values - lo) * (maximum / (hi - lo))


def _gamma_stretch(values: np.ndarray, maximum: float,
                   gamma: float = GAMMA_DEFAULT,
                   low: float = None, high: float = None) -> np.ndarray:
    # Гамма-коррекция после нормировки
    lo, hi = _limits(values, low, high)
    if hi <= lo:
        return values
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return maximum * np.power(normalized, 1.0 / gamma)


_output_types = {
    "no": _no_stretch,
    "percentile": _percentile_stretch,
    "gamma": _gamma_stretch,
}

OUTPUT_DEFAULT = "percentile"


def _to_uint(values: np.ndarray, bits: int, output: str,
             **params) -> np.ndarray:
    # Общий перевод результата фильтра в bits бит (0..MAX_UINT[bits])
    if bits not in MAX_UINT:
        raise ValueError(
            f"Неизвестная разрядность: {bits}. Доступные: {sorted(MAX_UINT)}."
        )
    if output not in _output_types:
        raise ValueError(
            f"Неизвестный режим вывода: '{output}'. "
            f"Доступные: {sorted(_output_types)}."
        )
    policy_of = _output_types[output]
    declared = [name for name in inspect.signature(policy_of).parameters
                if name not in ("values", "maximum")]
    unknown = sorted(set(params) - set(declared))
    if unknown:
        raise ValueError(
            f"Режим '{output}' не принимает параметры: {unknown}. "
            f"Доступны: {sorted(declared)}."
        )
    maximum = float(MAX_UINT[bits])
    scaled = policy_of(np.asarray(values, dtype=np.float64), maximum, **params)
    clipped = np.clip(np.floor(scaled + 0.5), 0, maximum)
    return clipped.astype(np.uint8 if bits == 8 else np.uint16)


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
        Перевести результат фильтра в 8 бит (0..255, dtype uint8)

        Параметры:
            values: матрица результата apply (диапазон не ограничен 0..255)
            output: "percentile" — линейная растяжка (по умолчанию),
                    "no" — только обрезка к 0..255,
                    "gamma" — гамма-коррекция после нормировки к [0, 1]
            **params: параметры выбранного режима:
                    "percentile": low, high — перцентили-отсечки;
                    "gamma": gamma, low, high.
        """
        return _to_uint(values, 8, output, **params)

    @staticmethod
    def to_uint10(values: np.ndarray, output: str = OUTPUT_DEFAULT,
                  **params) -> np.ndarray:
        # Перевести результат фильтра в 10 бит (0..1023, dtype uint16)
        return _to_uint(values, 10, output, **params)

    @staticmethod
    def to_uint12(values: np.ndarray, output: str = OUTPUT_DEFAULT,
                  **params) -> np.ndarray:
        # Вывод в 12 бит (0..4095, dtype uint16)
        return _to_uint(values, 12, output, **params)

    @staticmethod
    def to_uint16(values: np.ndarray, output: str = OUTPUT_DEFAULT,
                  **params) -> np.ndarray:
        # Вывод в 16 бит (0..65535, dtype uint16)
        return _to_uint(values, 16, output, **params)

# Шумоподавление: билатерный фильтр (ослабляет шум, не трогая границы)

DENOISE_DEFAULT = False  # выключено: шумоподавление нужно не всегда
DENOISE_RADIUS = 2          # окно (2*radius + 1)^2 = 5x5
DENOISE_SIGMA_SPACE = 1.0   # спад веса с расстоянием, пикселы
DENOISE_SIGMA_RANGE = 15.0  # спад веса с разностью яркости, отсчёты шкалы


def bilateral(
    values: np.ndarray,
    radius: int = DENOISE_RADIUS,
    sigma_space: float = DENOISE_SIGMA_SPACE,
    sigma_range: float = DENOISE_SIGMA_RANGE,
) -> np.ndarray:
    """
    Билатерный фильтр: среднее по окну, где вес соседа — произведение
    гауссова веса по расстоянию (sigma_space) и гауссова веса по разности
    яркостей (sigma_range).

    Считается в float64: это последний шаг перед записью, целая шкала уже
    выбрана, mpfr избыточна.

    Параметры:
        values: готовый кадр (n, m) или (n, m, 3) после to_uintN
        radius: полуразмер окна, кадр обрабатывается окном (2*radius + 1)^2
        sigma_space: вес соседа на расстоянии d — гаусс с сигмой в пикселах
        sigma_range: вес соседа с разностью яркости dr — в отсчётах шкалы
                     вывода (15 для 8 бит, ~3850 для 16)
        край кадра дополняется зеркально с повтором края (как pad="reflect")

    Возвращает:
        numpy.ndarray: кадр той же формы и dtype (целой dtype — с округлением
                       к ближайшему и обрезкой диапазона)

    Исключения:
        ValueError: если форма не (n, m) и не (n, m, 3) или параметры невалидны

    Цветной кадр фильтруется одним весом на все каналы: разность яркости —
    сумма квадратов по каналам. Граница, заметная в одном канале, защищает от
    размытия остальные.
    """
    if not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError(f"radius={radius}: ожидается целое число >= 1")
    if sigma_space <= 0 or sigma_range <= 0:
        raise ValueError(
            f"sigma_space={sigma_space}, sigma_range={sigma_range}: "
            f"ожидается оба больше нуля"
        )

    source = np.asarray(values)
    data = source.astype(np.float64)
    if data.ndim == 2:
        data = data[:, :, None]
    elif data.ndim != 3 or data.shape[2] != CHANNELS:
        raise ValueError(
            f"Ожидается кадр формы (n, m) или (n, m, {CHANNELS}), "
            f"получена форма {source.shape}"
        )

    n, m, channels = data.shape
    padded = np.pad(data, ((radius, radius), (radius, radius), (0, 0)),
                    mode="symmetric")
    total = np.zeros((n, m, channels), dtype=np.float64)
    weight_sum = np.zeros((n, m), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            space = (dy * dy + dx * dx) / (2.0 * sigma_space ** 2)
            window = padded[radius + dy: radius + dy + n,
                            radius + dx: radius + dx + m, :]
            diff = window - data
            # Один вес на все каналы: сумма квадратов разностей по каналам
            range_sq = (diff * diff).sum(axis=2) / (2.0 * sigma_range ** 2)
            weight = np.exp(-(space + range_sq))
            total += window * weight[:, :, None]
            weight_sum += weight

    result = total / weight_sum[:, :, None]
    if source.ndim == 2:
        result = result[:, :, 0]
    if not np.issubdtype(source.dtype, np.integer):
        return result
    limit = np.iinfo(source.dtype)
    result = np.clip(np.floor(result + 0.5), max(limit.min, 0), limit.max)
    return result.astype(source.dtype)


def denoise(values: np.ndarray, enabled: bool = DENOISE_DEFAULT,
            **params) -> np.ndarray:
    """
    Шумоподавление; enabled=False — кадр возвращается как есть.
    Идёт ПОСЛЕ to_uint; sigma_range задан в отсчётах шкалы вывода

    Параметры:
        values: кадр после перевода в N бит
        enabled: по умолчанию DENOISE_DEFAULT = False
        **params: radius, sigma_space, sigma_range — как у bilateral

    Возвращает:
        numpy.ndarray: кадр той же формы и dtype
    """
    if not enabled:
        return values
    return bilateral(values, **params)

