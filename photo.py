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
       выключен по умолчанию;
    5. метод process() объединяет вcю обработку (вход → apply → to_uintN)
       с гибким размещением denoise: off/before/after/before_and_after,
       чтобы можно было сравнивать влияние фильтра до и после
       гомоморфного преобразования.

Кадр дополняется до степени двойки (pad="reflect" по умолчанию), так как
это нужно для БПФ по методу Кули-Тьюки.
"""


import inspect

import numpy as np
from PIL import Image

import filtration
from color_filter import (CHANNELS, rgb_to_ycbcr, ycbcr_to_rgb)
# CHANNELS = 3
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


# Режимы с альфа-каналом
_ALPHA_MODES = ("RGBA", "RGBa", "LA", "La", "PA")


def read_rgb(path) -> np.ndarray:
    """
    Прочитать цветной кадр float64: (n, m, 3) или (n, m, 4).

    Источник с альфой (RGBA, LA, PA, палитра с прозрачностью) приводится
    к RGBA — альфа сохраняется четвёртым каналом и не меняется.
    """
    with Image.open(path) as picture:
        has_alpha = (
            picture.mode in _ALPHA_MODES or "transparency" in picture.info
        )
        mode = "RGBA" if has_alpha else "RGB"
        values = np.asarray(picture.convert(mode), dtype=np.float64)
    values[..., :3] = np.maximum(values[..., :3], MIN_VALUE)
    return values


def write_rgb(path, values: np.ndarray) -> None:
    """
    Записать цветной кадр uint8: (n, m, 3) как RGB, (n, m, 4) как RGBA.

    Альфа-канал без изменений — прозрачность сохраняется. Контейнер
    должен поддерживать альфу: PNG и TIFF — да, JPEG — нет.
    """
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[2] not in (CHANNELS, CHANNELS + 1):
        raise ValueError(
            f"Ожидается кадр (n, m, {CHANNELS}) или "
            f"(n, m, {CHANNELS + 1}), получена форма {values.shape}"
        )
    if values.dtype != np.uint8:
        raise ValueError(
            f"Цветной кадр записывается в 8 бит,"
            f"передан dtype {values.dtype}: "
            f"нужен photo_filter.to_uint8"
        )
    mode = "RGBA" if values.shape[2] == CHANNELS + 1 else "RGB"
    Image.fromarray(values, mode=mode).save(path)


def _limits(values: np.ndarray, low: float,
            high: float) -> tuple[float, float]:
    # Границы диапазона: low/high — перцентили; None — min/max
    # (выбросы не отсекаются).
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
    # Линейная растяжка отсечек low/high до 0..maximum
    lo, hi = _limits(values, low, high)
    if hi <= lo:
        return values
    return (values - lo) * (maximum / (hi - lo))


def _percentile_ycbcr_stretch(values: np.ndarray, maximum: float,
                              low: float = LOW_PERCENTILE,
                              high: float = HIGH_PERCENTILE) -> np.ndarray:
    # Перцентильная растяжка только по яркости (Y)
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3:
        return _percentile_stretch(source, maximum, low, high)
    ycc = rgb_to_ycbcr(source)
    lo, hi = _limits(ycc[..., 0], low, high)
    if hi <= lo:
        return source
    ycc[..., 0] = (ycc[..., 0] - lo) * (maximum / (hi - lo))
    return ycbcr_to_rgb(ycc)


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
    "percentile_ycbcr": _percentile_ycbcr_stretch,
    "gamma": _gamma_stretch,
}

OUTPUT_DEFAULT = "percentile"


def _to_uint(values: np.ndarray, bits: int, output: str,
             **params) -> np.ndarray:
    # Общий перевод результата фильтра в bits бит (0..MAX_UINT[bits]).
    # Растяжка считается по цветовым каналам
    source = np.asarray(values)
    alpha = None
    if source.ndim == 3 and source.shape[2] == CHANNELS + 1:
        alpha = source[..., CHANNELS]
        source = source[..., :CHANNELS]
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
    scaled = policy_of(np.asarray(source, dtype=np.float64), maximum,
                       **params)
    clipped = np.clip(np.floor(scaled + 0.5), 0, maximum)
    result = clipped.astype(np.uint8 if bits == 8 else np.uint16)
    if alpha is not None:
        alpha = np.clip(np.floor(np.asarray(alpha, dtype=np.float64) + 0.5),
                        0, maximum)
        result = np.concatenate(
            [result, alpha.astype(result.dtype)[..., np.newaxis]], axis=2)
    return result


# Шумоподавление: билатерный фильтр

DENOISE_DEFAULT = False  # выключено
DENOISE_RADIUS = 2          # окно (2*radius + 1)^2 = 5x5
DENOISE_SIGMA_SPACE = 1.0   # спад веса с расстоянием, пиксели
DENOISE_SIGMA_RANGE = 15.0  # спад веса с разностью яркости, отсчёты шкалы

# Допустимые позиции билатерального фильтра
DENOISE_PLACES = ("off", "before", "after", "before_and_after")
DENOISE_PLACE_DEFAULT = "after"


class photo_filter:
    """
    Пример использования:
        photo = read_gray("examples/pic1.png")
        f = photo_filter(lf_gain=0.75, hf_gain=2.0, d0=30)
        write_gray("examples/pic1_processed.png", f.to_uint8(f.apply(photo)))

    Шумоподавление встроено в обработку через process(): флаг denoise включает
    билатерный фильтр, filter_place задаёт, где именно он стоит — до
    гомоморфного фильтра (before), после него (after) или с обеих сторон
    (before_and_after).
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
        denoise: bool = DENOISE_DEFAULT,
        filter_place: str = DENOISE_PLACE_DEFAULT,
        denoise_radius: int = DENOISE_RADIUS,
        denoise_sigma_space: float = DENOISE_SIGMA_SPACE,
        denoise_sigma_range: float = DENOISE_SIGMA_RANGE,
    ):
        """
        Параметры шумоподавления:
            denoise: master-переключатель билатерного фильтра
            filter_place: "off" | "before" | "after" | "before_and_after";
                имеет смысл только при denoise=True
            denoise_radius / denoise_sigma_space / denoise_sigma_range:
                параметры bilateral; sigma_range задан в отсчётах шкалы
                8 бит и автоматически масштабируется для вывода другой
                разрядности на выходе из обработки
        """
        if filter_place not in DENOISE_PLACES:
            raise ValueError(
                f"Неизвестная позиция шумоподавления: '{filter_place}'. "
                f"Доступные: {DENOISE_PLACES}."
            )
        self._filter = homomorphic_filter(
            lf_filter, hf_filter, lf_gain, hf_gain, d0, order
        )
        self._pad = pad
        self._window = window
        self._denoise = bool(denoise)
        self._filter_place = filter_place
        self._denoise_params = {
            "radius": denoise_radius,
            "sigma_space": denoise_sigma_space,
            "sigma_range": denoise_sigma_range,
        }

    @property
    def filter_place(self) -> str:
        # Куда встроен билатерный фильтр (только при denoise=True)
        return self._filter_place

    @property
    def denoise(self) -> bool:
        return self._denoise

    def _stage_active(self, stage: str, on: bool = None,
                      place: str = None) -> bool:
        # Активен ли denoise на данном этапе ("before"/"after")
        enabled = self._denoise if on is None else bool(on)
        current = self._filter_place if place is None else place
        if not enabled or current == "off":
            return False
        if stage == "before":
            return current in ("before", "before_and_after")
        return current in ("after", "before_and_after")

    def process(
        self,
        image: np.ndarray,
        bits: int = BITS_DEFAULT,
        output: str = OUTPUT_DEFAULT,
        filter_place: str = None,
        denoise: bool = None,
        **params,
    ) -> np.ndarray:
        """
        Полный цикл: вход -> (denoise before) -> apply -> to_uintN ->
        (denoise after).

        Параметры:
            image: матрица (numpy.ndarray или путь к файлу)
            bits: разрядность вывода — 8, 10, 12 или 16
            output: режим растяжки для to_uintN (см. to_uint8)
            filter_place: переопределить позицию из __init__ для этого вызова
            denoise: переопределить флаг из __init__ для этого вызова
            **params: параметры выбранного режима растяжки (low, high, gamma)

        Возвращает:
            numpy.ndarray: кадр dtype uint8 (bits=8) или uint16 (bits>8)

        Исключения:
            ValueError: при неизвестном filter_place или bits/output
        """
        place = self._filter_place if filter_place is None else filter_place
        if place not in DENOISE_PLACES:
            raise ValueError(
                f"Неизвестная позиция шумоподавления: '{place}'. "
                f"Доступные: {DENOISE_PLACES}."
            )
        on = self._denoise if denoise is None else bool(denoise)

        if not isinstance(image, np.ndarray):
            image = read_gray(image)
        source = np.asarray(image, dtype=np.float64)

        if self._stage_active("before", on, place):
            source = bilateral(source, **self._denoise_params)

        filtered = self.apply(source)
        result = _to_uint(filtered, bits, output, **params)

        if self._stage_active("after", on, place):
            kwargs = dict(self._denoise_params)
            if bits != 8:
                # sigma_range откалиброван на шкалу 8 бит
                kwargs["sigma_range"] *= MAX_UINT[bits] / MAX_UINT[8]
            result = bilateral(result, **kwargs)
        return result

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
                    "percentile_ycbcr" — растяжка только по яркости Y,
                    "no" — только обрезка к 0..255,
                    "gamma" — гамма-коррекция после нормировки к [0, 1]
            **params: параметры выбранного режима:
                    "percentile", "percentile_ycbcr": low, high —
                        перцентили-отсечки;
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
        values: готовый кадр (n, m), (n, m, 3) или (n, m, 4) после to_uintN;
                для RGBA альфа-канал не меняется
        radius: полуразмер окна, кадр обрабатывается окном (2*radius + 1)^2
        sigma_space: вес соседа на расстоянии d — гаусс с сигмой в пикселах
        sigma_range: вес соседа с разностью яркости dr — в отсчётах шкалы
                     вывода (15 для 8 бит, ~3850 для 16)
        край кадра дополняется зеркально с повтором края (как pad="reflect")

    Возвращает:
        numpy.ndarray: кадр той же формы и dtype (целой dtype — с округлением
                       к ближайшему и обрезкой диапазона)

    Исключения:
        ValueError: если форма не (n, m), (n, m, 3), (n, m, 4) или параметры
                    невалидны

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
    if source.ndim == 3 and source.shape[2] == CHANNELS + 1:
        # Альфа-канал не учитывается: вес считается только по цветовым каналам
        rgb = bilateral(source[..., :CHANNELS], radius=radius,
                        sigma_space=sigma_space, sigma_range=sigma_range)
        return np.concatenate([rgb, source[..., CHANNELS:]], axis=2)
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
