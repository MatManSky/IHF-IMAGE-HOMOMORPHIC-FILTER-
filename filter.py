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

# Нижняя граница яркости, чтобы log2(x<0) превращался в log2(MIN_VALUE).
MIN_VALUE = 1.0


def _setup_precision() -> None:
    """Установить точность MPFR-контекста."""
    gmpy2.get_context().precision = _PRECISION_BITS


# Точность устанавливается сразу при импорте модуля: все константы
# и поворачивающие множители вычисляются при полной точности
_setup_precision()

# Допуск округления: 2^-100 ~ 8e-31, что выше, чем шаг данных ~ 1e-16
_EPS = mpfr(2) ** mpfr(-100)

# Граница для быстрого exp2 в float64
_ROUND_MARGIN = 1e-9

# Для оптимизации скорости вычислений
_TWO_PI = 2 * gmpy2.const_pi()
_TWIDDLE_CACHE: dict = {}   # (length, sign) -> множители уровня
_ROOT_CACHE: dict = {}      # (n, sign)      -> корни из единицы
_BITREV_CACHE: dict = {}    # n              -> пары индексов


def _twiddle_table(length: int, sign: int) -> list:
    # Множители exp(sign*2*pi*i/length) для i = 0..length/2-1
    key = (length, sign)
    table = _TWIDDLE_CACHE.get(key)
    if table is None:
        ang = sign * _TWO_PI / length
        table = [mpc(gmpy2.cos(ang * k), gmpy2.sin(ang * k))
                 for k in range(length // 2)]
        _TWIDDLE_CACHE[key] = table
    return table


def _roots_of_unity(n: int, sign: int) -> list:
    # Корни n-й степени exp(sign*2*pi*i*t/n), t = 0..n-1
    key = (n, sign)
    table = _ROOT_CACHE.get(key)
    if table is None:
        ang = sign * _TWO_PI / n
        table = [mpc(gmpy2.cos(ang * t), gmpy2.sin(ang * t))
                 for t in range(n)]
        _ROOT_CACHE[key] = table
    return table


def _bitrev_swaps(n: int) -> list:
    # Пары индексов бит-реверс перестановки (оптимизация)
    swaps = _BITREV_CACHE.get(n)
    if swaps is None:
        swaps = []
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                swaps.append((i, j))
        _BITREV_CACHE[n] = swaps
    return swaps


def _dft_1d(x: list, inverse: bool = False) -> list:
    """
    Одномерное дескретное преобразование Фурье:
    X[k] = sum_j x[j] * exp(±2*pi*i*j*k/n)
    """
    n = len(x)
    sign = 1 if inverse else -1
    w = _roots_of_unity(n, sign)
    out = []
    for k in range(n):
        s = mpc(0)
        t = 0
        for j in range(n):
            s += x[j] * w[t]
            t += k
            if t >= n:      # k < n и t < n, значит t + k < 2n — одно вычитание
                t -= n
        out.append(s)
    return out


def _fft_1d(x: list, inverse: bool = False) -> list:
    """
    Одномерное БПФ по алгоритму Кули-Тьюки (radix-2, итеративный)

    Требует длину, равную степени двойки; иначе — долгий прямой расчёт ДПФ.
    """
    n = len(x)
    if n & (n - 1) != 0:
        return _dft_1d(x, inverse)

    a = list(x)

    # 1. Бит-реверс перестановка
    for i, j in _bitrev_swaps(n):
        a[i], a[j] = a[j], a[i]

    # 2. Бабочки
    sign = 1 if inverse else -1
    length = 2
    while length <= n:
        half = length // 2
        w_table = _twiddle_table(length, sign)
        for start in range(0, n, length):
            u = a[start]
            v = a[start + half]
            a[start] = u + v
            a[start + half] = u - v
            for k in range(1, half):
                u = a[start + k]
                v = a[start + k + half] * w_table[k]
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

# Допустимые режимы дополнения кадра до степени двойки
_PAD_MODES = ("zero", "reflect")


def _next_power_of_two(size: int) -> int:
    # Нужно для БПФ по Кули-Тьюки
    return 1 << (size - 1).bit_length()


def _mul_rows(data: list, weights: list) -> list:
    # Поэлементное умножение двух матриц
    return [
        [v * w for v, w in zip(row, weight_row)]
        for row, weight_row in zip(data, weights)
    ]


def _div_rows(data: list, weights: list) -> list:
    # Компенсация окна: деление, ограниченное снизу _WINDOW_FLOOR
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
    if mode not in _PAD_MODES:
        raise ValueError(
            f"Неизвестный режим дополнения: '{mode}'. "
            f"Доступные: {sorted(_PAD_MODES)}."
        )
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


def _print_dataset(label: str, data) -> None:
    """
    Печатает содержимое массива (вход, фильтр, выход).
    Массив выводится построчно и до последнего элемента,
    полный вывод включается ключом --verbose.

    У комплексных чисел печатаются действительная часть и модуль.
    """
    array = np.asarray(data)
    print(f"{label}: shape={array.shape}")
    with np.printoptions(threshold=np.inf, precision=4, suppress=True,
                         linewidth=240):
        print(array)
        if np.iscomplexobj(array):
            print("  modul:")
            print(np.abs(array))
    print()


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
            (
                lf_filter.startswith("lp_")
                and hf_filter == "hp_" + lf_filter[len("lp_"):]
            )
            or lf_filter == "allpass" 
            or hf_filter == "allpass"
        ):
            raise ValueError(
                f"Несогласованная пара фильтров: '{lf_filter}' + '{hf_filter}'. "
                f"Ожидается lp_X и hp_X одной семьи, например "
                f"'{filtration.DEFAULT_LF_FILTER}' + "
                f"'{filtration.DEFAULT_HF_FILTER}', либо 'allpass' с любым "
                f"фильтром."
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

        # Суммарная АЧХ (нужна для оптимизации)
        self._response = None
        self._response_size = None
        self._response_parts = None

    def _get_response(self, size: tuple, verbose: bool) -> tuple:
        # S * (lf_gain*ФНЧ + hf_gain*ФВЧ)
        if self._response_size != size or self._response is None:
            lf_matrix = filtration.get_filter(
                self._lf_filter, *size, **self._filter_params)
            hf_matrix = filtration.get_filter(
                self._hf_filter, *size, **self._filter_params)
            lf_gain, hf_gain = self._lf_gain, self._hf_gain
            self._response = [
                [lf * lf_gain + hf * hf_gain for lf, hf in zip(l_row, h_row)]
                for l_row, h_row in zip(lf_matrix, hf_matrix)
            ]
            self._response_size = size
            self._response_parts = (lf_matrix, hf_matrix) if verbose else None
        if verbose and self._response_parts is None:
            self._response_parts = (
                filtration.get_filter(self._lf_filter, *size,
                                      **self._filter_params),
                filtration.get_filter(self._hf_filter, *size,
                                      **self._filter_params),
            )
        return self._response_parts, self._response

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
            image: входная матрица формы (n, m), значения > 0
                   (numpy.ndarray; список списков приводится к ndarray)
            pad: "zero" — дополнить кадр нулями до степени двойки,
                 "reflect" — дополнить зеркально; 
            window: имя окна из filtration, после ОБПФ окно компенсируется

        Возвращает:
            numpy.ndarray: результат в формате входной матрицы (int64)

        Исключения:
            ValueError: если форма входной матрицы не (n, m)
        """
        return self._apply(image, pad, window, verbose=False)

    # apply с отладочным выводом 
    def apply_info(
        self,
        image: np.ndarray,
        pad: str = None,
        window: str = None,
    ) -> np.ndarray:
        return self._apply(image, pad, window, verbose=True)

    # Функция гомоморфной фильтрации (общая реализация для apply/apply_info)
    def _apply(
        self,
        image: np.ndarray,
        pad: str,
        window: str,
        verbose: bool,
    ) -> np.ndarray:
        _setup_precision()
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(
                f"Ожидается одноканальное изображение формы (n, m), "
                f"получена форма {image.shape}"
            )
        n, m = image.shape

        step = 0

        def report(label: str, data) -> None:
            #Пронумеровать и вывести датасет (для apply_info)
            nonlocal step
            if verbose:
                step += 1
                _print_dataset(f"{step}. {label}", data)

        report("input I", image)

        # 1. Логарифмирование
        log_data = self._logarithm(image)
        report("log2(I)", log_data)

        # Окно и дополнение до степени двойки
        window_matrix = (
            filtration.get_window(window, n, m) if window is not None else None
        )
        if window_matrix is not None:
            log_data = _mul_rows(log_data, window_matrix)
            report(f"multiplication by window '{window}'", log_data)
        if pad is not None:
            log_data = _pad_to_power_of_two(log_data, pad)
            report(f"addition to the power of two ('{pad}')", log_data)

        # 2. Прямое БПФ
        spectrum = _fft_2d(log_data, inverse=False)
        report("FFT", spectrum)

        # 3. Обработка спектра
        size = (len(log_data), len(log_data[0]))
        parts, response = self._get_response(size, verbose)
        if verbose:
            lf_matrix, hf_matrix = parts
            report(f"AFC LPF '{self._lf_filter}' "
                   f"(lf_gain={float(self._lf_gain):g})", lf_matrix)
            report(f"AFC HPF '{self._hf_filter}' "
                   f"(hf_gain={float(self._hf_gain):g})", hf_matrix)

        spectrum = [
            # Спектр * суммарная АЧХ
            [s * h for s, h in zip(row, response_row)]
            for row, response_row in zip(spectrum, response)
        ]
        report("Spectr after processing", spectrum)

        # 4. Обратное БПФ
        restored = _fft_2d(spectrum, inverse=True)
        report("inverse FFT", restored)

        # Антидополнение до степени двойки и компенсация окна
        restored = _crop(restored, n, m)
        report("opposite padding", restored)
        if window_matrix is not None:
            restored = _div_rows(restored, window_matrix)
            report(f"division by window '{window}'", restored)

        # 5-6. Антилогарифмирование и округление для вывода
        result = self._antilogarithm(restored, n, m)
        report("exp2 and rounding", result)

        return result

    def _logarithm(self, image: np.ndarray) -> list:
        # Логарифмирование по основанию 2.
        arr = np.asarray(image, dtype=np.float64)
        count = int((arr <= 0).sum())
        if count:
            print(
                f"log2: поднято до MIN_VALUE={MIN_VALUE:g} неположительных "
                f"пикселов: {count} (минимум входа = {arr.min():g})"
            )
            arr = np.maximum(arr, MIN_VALUE)

        cache: dict = {}
        rows = arr.tolist()
        out = []
        for row in rows:
            line = []
            for v in row:
                c = cache.get(v)
                if c is None:
                    c = mpc(gmpy2.log2(mpfr(v)))
                    cache[v] = c
                line.append(c)
            out.append(line)
        return out

    def _antilogarithm(self, data: list, n: int, m: int) -> np.ndarray:
        # Антилогарифмирование exp2 и округление к ближайшему целому.
        
        vals = [v.real for row in data[:n] for v in row[:m]]
        total = n * m
        x = np.fromiter((float(v) for v in vals), dtype=np.float64,
                        count=total)
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.exp2(x)
        ok = np.isfinite(y) & (y < 9.0e18)
        ok &= np.abs(y - (np.floor(y) + 0.5)) > _ROUND_MARGIN * np.maximum(1.0, y)
        out = np.floor(np.where(ok, y, 0.0) + 0.5).astype(np.int64)
        exact = np.flatnonzero(~ok)
        if exact.size:
            exp2 = gmpy2.exp2
            for i in exact:
                out[i] = int(gmpy2.round_away(exp2(vals[int(i)]) + _EPS))
        return out.reshape(n, m)
