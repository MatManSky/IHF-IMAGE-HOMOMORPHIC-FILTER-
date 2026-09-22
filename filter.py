"""
Модуль для гомоморфной фильтрации изображений.

Обработка ведётся в арифметике произвольной точности (MPFR),
64-битный numpy используется только для входной/выходной матрицы:

    1. Логарифмирование log2 (mpfr)
    2. Двумерное БПФ
    3. Разделение спектра на НЧ и ВЧ, обработка, соединение
    4. Обратное двумерное БПФ
    5. Антилогарифмирование exp2 (mpfr)
    6. Округление к ближайшему целому с допуском погрешности обработки
"""

import numpy as np
import gmpy2
from gmpy2 import mpfr, mpc # mpc — это mpfr для комплексных чисел
import filtration

# Точность: 256 бит ~ 80 десятичных знаков
_PRECISION_BITS = 256


def _setup_precision() -> None:
    """Установить точность MPFR-контекста."""
    gmpy2.get_context().precision = _PRECISION_BITS


# Точность устанавливается сразу при импорте модуля: все константы
# и поворачивающие множители вычисляются при полной точности
_setup_precision()

# Допуск округления: 2^-100 ~ 8e-31, что выше, чем шаг данных ~ 1e-16
_EPS = mpfr(2) ** mpfr(-100)


def _dft_1d(x: list, inverse: bool = False) -> list:
    """
    Одномерное ДПФ

    X[k] = sum_j x[j] * exp(±2*pi*i*j*k/n)
    """
    n = len(x)
    sign = 1 if inverse else -1
    two_pi = 2 * gmpy2.const_pi()
    out = []
    for k in range(n):
        s = mpc(0)
        for j in range(n):
            ang = sign * two_pi * j * k / n
            s += x[j] * mpc(gmpy2.cos(ang), gmpy2.sin(ang))
        out.append(s)
    return out


def _fft_1d(x: list, inverse: bool = False) -> list:
    """
    Одномерное БПФ по алгоритму Кули-Тьюки (radix-2, итеративный)

    Требует длину, равную степени двойки; иначе — долгий прямой расчёт ДПФ.
    Можно реализовать другой БПФ, чтобы работал быстрее, но пока ограничимся таким.
    (Выполняется интеративно (без рекурсии), так как на Python так быстрее)
    Этапы:
        1. Бит-реверс перестановка: элементы переставляются так,
           чтобы бабочки можно было выполнять «на месте»;
        2. Бабочки: для длин 2, 4, ..., n спектры половинной длины
           объединяются парами: (u + w*v, u - w*v),
           где w — поворачивающий множитель exp(±2*pi*i*k/length).
    """
    n = len(x)
    if n & (n - 1) != 0:
        return _dft_1d(x, inverse)

    a = list(x)

    # 1. Бит-реверс перестановка
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # 2. Бабочки
    sign = 1 if inverse else -1
    two_pi = 2 * gmpy2.const_pi()
    length = 2
    while length <= n:
        half = length // 2
        ang = sign * two_pi / length
        # Таблица множителей уровня: half вычислений cos/sin
        # вместо n/2 (переиспользуется для всех блоков)
        w_table = [
            mpc(gmpy2.cos(ang * k), gmpy2.sin(ang * k))
            for k in range(half)
        ]
        for start in range(0, n, length):
            for k in range(half):
                w = w_table[k]
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
        length <<= 1

    return a


def _fft_2d(data: list, inverse: bool = False) -> list:
    """
    Двумерное БПФ матрицы строчно-столбцовым разложением.

    Сначала БПФ каждой строки, затем каждого столбца.
    По правилам numpy: прямое — без масштаба, обратное — с множителем 1/(n*m)
    """
    n = len(data)
    m = len(data[0])

    # БПФ по строкам
    tmp = [_fft_1d(row, inverse) for row in data]

    # БПФ по столбцам (транспонирование -> БПФ -> транспонирование)
    cols = list(zip(*tmp))
    cols_dft = [_fft_1d(list(col), inverse) for col in cols]
    result = [list(row) for row in zip(*cols_dft)]

    # Умножение на масштабирующий коэф.
    if inverse:
        scale = mpfr(1) / mpfr(n * m)
        result = [[cell * scale for cell in row] for row in result]

    return result

_WINDOW_FLOOR = mpfr(0.05) # Нижний предел окна

def _next_power_of_two(size: int) -> int:
    """Нужно для БПФ по Кули-Тьюки"""
    return 1 << (size - 1).bit_length()


def _mul_rows(data: list, weights: list) -> list:
    """Поэлементное умножение двух матриц"""
    return [
        [v * w for v, w in zip(row, weight_row)]
        for row, weight_row in zip(data, weights)
    ]


def _div_rows(data: list, weights: list) -> list:
    """Компенсация окна: деление, ограниченное снизу _WINDOW_FLOOR"""
    return [
        [v / (w if w > _WINDOW_FLOOR else _WINDOW_FLOOR)
         for v, w in zip(row, weight_row)]
        for row, weight_row in zip(data, weights)
    ]


def _pad_to_power_of_two(data: list, mode: str = "zero") -> list:
    """
    Дополнить кадр до степени двойки:
    1) "zero": нулями в лог области;
    2) "reflect": зеркально, с повтором края.
    """
    rows, cols = len(data), len(data[0])
    target_rows = _next_power_of_two(rows)
    target_cols = _next_power_of_two(cols)

    if target_cols > cols:
        if mode == "zero":
            extra = [mpc(0)] * (target_cols - cols)
            data = [list(row) + extra for row in data]
        else:
            mirror = [cols - 1 - k % cols for k in range(target_cols - cols)]
            data = [list(row) + [row[j] for j in mirror] for row in data]

    if target_rows > rows:
        if mode == "zero":
            blank = [mpc(0)] * target_cols
            data = list(data) + [list(blank) for _ in range(target_rows - rows)]
        else:
            mirror = [rows - 1 - k % rows for k in range(target_rows - rows)]
            data = list(data) + [list(data[j]) for j in mirror]

    return data

def _crop(data: list, n: int, m: int) -> list:
    """Отсечь дополнение: обратно к форме исходного кадра."""
    return [row[:m] for row in data[:n]]


class homomorphic_filter:
    """
    Пример использования:
    f = homomorphic_filter(lf_gain=0.5, hf_gain=2.0)  # ослабить свет, усилить коэф. отр
    result = f.apply(image_matrix)

    Кадр произвольного размера для БПФ radix-2 дополняется:
    f.apply(photo, pad="zero") или f.apply(photo, pad="reflect")
    """

    def __init__(
        self,
        lf_filter: str = filtration.DEFAULT_LF_FILTER,
        hf_filter: str = filtration.DEFAULT_HF_FILTER,
        lf_gain: float = 1.0,
        hf_gain: float = 1.0,
        d0: float = filtration.D0_DEFAULT,
        order: int = filtration.ORDER_DEFAULT,
        ripple_db: float = filtration.RIPPLE_DB_DEFAULT,
    ):
        # Пара ФНЧ и ФВЧ должна быть согласована
        if not (
            lf_filter.startswith("lp_")
            and hf_filter == "hp_" + lf_filter[len("lp_"):]
        ):
            raise ValueError(
                f"Несогласованная пара фильтров: '{lf_filter}' + '{hf_filter}'. "
                f"Ожидается lp_X и hp_X одной семьи, например "
                f"'{filtration.DEFAULT_LF_FILTER}' + "
                f"'{filtration.DEFAULT_HF_FILTER}'."
            )
        self._lf_filter = lf_filter
        self._hf_filter = hf_filter
        self._lf_gain = mpfr(lf_gain) # множитель НЧ-части: <1 ослабляет свет
        self._hf_gain = mpfr(hf_gain) # множитель ВЧ-части: >1 подчёркивает детали

        # Параметры частотной сетки
        self._filter_params = {
            "d0": d0,  # Частота среза в пикселях
            "order": order, # Крутизна среза АЧХ
            "ripple_db": ripple_db, # Неравномерность АЧХ Чебышёва-1 в полосе пропускания
        }

    def apply(
        self,
        image: np.ndarray,
        pad: str = None,
        window: str = None,
    ) -> np.ndarray:
        """
        Применить гомоморфный фильтр к изображению.
        (обработка в mpfr без промежуточных округлений)

        Параметры:
            image: входная матрица (numpy.ndarray, значения > 0)
            pad: "zero" — дополнить кадр нулями до степени двойки,
                 "reflect" — дополнить зеркально; 
            window: имя окна из filtration, после ОБПФ окно компенсируется

        Возвращает:
            numpy.ndarray: результат в формате входной матрицы (int64)
        """
        _setup_precision()
        n, m = image.shape

        # 1. Логарифмирование
        log_data = self._logarithm(image)

        # Окно и дополнение до степени двойки
        window_matrix = (
            filtration.get_window(window, n, m) if window is not None else None
        )
        if window_matrix is not None:
            log_data = _mul_rows(log_data, window_matrix)
        if pad is not None:
            log_data = _pad_to_power_of_two(log_data, pad)

        # 2. Прямое БПФ
        spectrum = _fft_2d(log_data, inverse=False)

        # 3. Обработка спектра
        size = (len(log_data), len(log_data[0]))
        lf_matrix = filtration.get_filter(
            self._lf_filter, *size, **self._filter_params)  # ФНЧ
        hf_matrix = filtration.get_filter(
            self._hf_filter, *size, **self._filter_params)  # ФВЧ

        lf_spectrum = [
            # Спектр * функция фильтра * коэф. усиления
            [s * h * self._lf_gain for s, h in zip(row, matrix_row)]
            for row, matrix_row in zip(spectrum, lf_matrix)
        ]
        hf_spectrum = [
            [s * h * self._hf_gain for s, h in zip(row, matrix_row)]
            for row, matrix_row in zip(spectrum, hf_matrix)
        ]
        spectrum = [
            [lf + hf for lf, hf in zip(lf_row, hf_row)]
            for lf_row, hf_row in zip(lf_spectrum, hf_spectrum)
        ]

        # 4. Обратное БПФ
        restored = _fft_2d(spectrum, inverse=True)

        # Антидополнение до степени двойки и компенсация окна
        restored = _crop(restored, n, m)
        if window_matrix is not None:
            restored = _div_rows(restored, window_matrix)

        # 5-6. Антилогарифмирование и округление для вывода
        result = self._antilogarithm(restored, n, m)

        return result

    def _logarithm(self, image: np.ndarray) -> list:
        """
        Логарифмирование по основанию 2
        """
        rows = np.asarray(image, dtype=np.float64).tolist()
        return [[mpc(gmpy2.log2(mpfr(v))) for v in row] for row in rows]

    def _antilogarithm(self, data: list, n: int, m: int) -> np.ndarray:
        """
        Антилогарифмирование exp2 и округление к ближайшему целому
        """
        result = np.zeros((n, m), dtype=np.int64)

        for i in range(n):
            for j in range(m):
                v = gmpy2.exp2(data[i][j].real)
                result[i, j] = int(gmpy2.round_away(v + _EPS))

        return result
