"""
Модуль для создания матриц освещения (E), отражения (R) и составных изображений.

Архитектура:
    - _type_registry — словари с функциями для каждого типа матрицы
    - create_matrix(name, n, m, kind) — создать матрицу (E или R)
    - create_image(e_type, r_type, n, m) — создать изображение I = E * R

Модель изображения: I = E * R, где
    - E — матрица освещения, целые числа (int64) от 1 до 255
    - R — коэффициент отражения, float64 в диапазоне (R_MIN, 1], R > 0 строго
    - I — яркость, float64 в диапазоне (0, 255]
0 < E(x,y) <= 255;
0 < R(x,y) <= 1;
I(x,y) = E(x,y) * R(x,y) => 0 < I(x,y) <= 255;

Коэффициенты отражения (R):
    const    — плоский фон (R = 1)
    chess    — шахматная доска (3 уровня: 1.0, 0.5, 1/255)
    black_square — тёмный прямоугольник на светлом фоне
    white_square — светлый прямоугольник на тёмном фоне
    random   — случайная текстура
    snowman  — снеговик (белые шары на сером фоне, R_MIN для глаз/пуговиц)
    cat      — кот (тёмный силуэт на светлом фоне)

Ссылки:
 1. Красильников Н.Н. Цифровая обработка 2D- и 3D-изображений: учеб. пособие. СПб.: БХВ-Петербург, 2011. 608 с.;
 2. Stockham T.G. Image Processing in the Context of a Visual Model // Proceedings of the IEEE. 1972, Vol. 60, No. 7. P. 828–842;
"""

import numpy as np

# Нижняя граница коэффициента отражения (не 0, чтобы log был определён)
R_MIN = 1.0 / 255.0

# Вспомогательные функции
def _chess_matrix(n: int, m: int) -> np.ndarray:
    # Шахматная доска с тремя уровнями отражения (1.0, 0.5, 1/255)
    r = np.zeros((n, m), dtype=np.float64)
    levels = (1.0, 0.5, R_MIN)
    for i in range(n):
        for j in range(m):
            idx = (i // 2 + j // 2) % 3
            r[i, j] = levels[idx]
    return r

def _random_matrix(n: int, m: int) -> np.ndarray:
    # Случайные коэффициенты отражения от 1/255 до 1
    return np.random.uniform(R_MIN, 1.0, size=(n, m))

def _square_matrix(n: int, m: int, field: float, patch: float) -> np.ndarray:
    # Прямоугольник в центре кадра (от 1/4 до 3/4)
    r = np.full((n, m), field, dtype=np.float64)
    r[n // 4:n - n // 4, m // 4:m - m // 4] = patch
    return r

def _black_square_matrix(n: int, m: int) -> np.ndarray:
    return _square_matrix(n, m, field=R_MIN, patch=1.0)

def _white_square_matrix(n: int, m: int) -> np.ndarray:
    return _square_matrix(n, m, field=1.0, patch=R_MIN)

def _const_matrix_float(n: int, m: int) -> np.ndarray:
    # Константные коэффициенты отражения — все единицы
    return np.ones((n, m), dtype=np.float64)

# Детали для снеговика и кота
def _circle_mask(n: int, m: int, cx: float, cy: float, radius: float) -> np.ndarray:
    # Круг: (cx, cy) — доли размеров, radius — доля меньшей стороны
    y, x = np.ogrid[:n, :m]
    side = min(n, m)
    return ((x - cx * m) ** 2 + (y - cy * n) ** 2) <= (radius * side) ** 2

def _ellipse_mask(n: int, m: int, cx: float, cy: float,
                  rx: float, ry: float) -> np.ndarray:
    # Эллипс: полуоси — доли ширины и высоты
    y, x = np.ogrid[:n, :m]
    return (((x - cx * m) / (rx * m)) ** 2
            + ((y - cy * n) / (ry * n)) ** 2) <= 1.0

def _triangle_mask(n: int, m: int, v0: tuple, v1: tuple, v2: tuple) -> np.ndarray:
    # Треугольник по трём вершинам (доли размеров), барицентрические координаты
    y, x = np.ogrid[:n, :m]
    x0, y0 = v0[0] * m, v0[1] * n
    x1, y1 = v1[0] * m, v1[1] * n
    x2, y2 = v2[0] * m, v2[1] * n

    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(det) < 1e-9:
        return np.zeros((n, m), dtype=bool)

    lam1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
    lam2 = ((y2 - y0) * (x - x0) + (x0 - x2) * (y - y0)) / det
    lam3 = 1.0 - lam1 - lam2
    return (lam1 >= 0) & (lam2 >= 0) & (lam3 >= 0)

def _snowman_matrix(n: int, m: int) -> np.ndarray:
    """
    Снеговик как коэффициент отражения: белые шары (R = 1) на сером снегу
    (R = 0.30), угольки глаз и пуговиц (R = 0.06), морковка (R = 0.50).
    """
    r = np.full((n, m), 0.30, dtype=np.float64)

    r[_circle_mask(n, m, 0.50, 0.72, 0.22)] = 1.00   # нижний шар
    r[_circle_mask(n, m, 0.50, 0.44, 0.16)] = 0.98   # средний шар
    r[_circle_mask(n, m, 0.50, 0.22, 0.11)] = 0.96   # голова

    # Нос-морковка (острие вправо)
    r[_triangle_mask(n, m, (0.62, 0.225), (0.515, 0.21), (0.515, 0.25))] = 0.50

    # Глаза и пуговицы
    r[_circle_mask(n, m, 0.46, 0.20, 0.020)] = 0.06
    r[_circle_mask(n, m, 0.54, 0.20, 0.020)] = 0.06
    r[_circle_mask(n, m, 0.50, 0.40, 0.022)] = 0.06
    r[_circle_mask(n, m, 0.50, 0.48, 0.022)] = 0.06
    r[_circle_mask(n, m, 0.50, 0.66, 0.025)] = 0.06

    return r

def _cat_matrix(n: int, m: int) -> np.ndarray:
    """
    Кот как коэффициент отражения: тёмный силуэт (R = 0.12) на светлом
    фоне (R = 0.85), видымые глаза (R = 0.95), нос (R = 0.40).
    """
    r = np.full((n, m), 0.85, dtype=np.float64)

    body = (0.50, 0.68)
    r[_ellipse_mask(n, m, body[0], body[1], 0.16, 0.22)] = 0.12   # туловище
    r[_circle_mask(n, m, 0.50, 0.34, 0.13)] = 0.14                # голова

    # Уши
    r[_triangle_mask(n, m, (0.40, 0.30), (0.46, 0.30), (0.42, 0.18))] = 0.12
    r[_triangle_mask(n, m, (0.54, 0.30), (0.60, 0.30), (0.58, 0.18))] = 0.12

    # Хвост: дуга из кружков (каждый следующий перекрывает предыдущий)
    for k in range(9):
        t = k / 8.0
        cx = 0.66 + 0.16 * t
        cy = 0.74 - 0.30 * t + 0.10 * t * t
        r[_circle_mask(n, m, cx, cy, 0.030 - 0.018 * t)] = 0.12

    # Глаза
    r[_ellipse_mask(n, m, 0.455, 0.33, 0.028, 0.035)] = 0.95
    r[_ellipse_mask(n, m, 0.545, 0.33, 0.028, 0.035)] = 0.95

    # Нос
    r[_triangle_mask(n, m, (0.48, 0.38), (0.52, 0.38), (0.50, 0.41))] = 0.40

    return r

def _const_matrix(n: int, m: int) -> np.ndarray:
    # Константное освещение — все единицы
    return np.ones((n, m), dtype=np.int64)

def _sawtooth_matrix(n: int, m: int) -> np.ndarray:
    # Пилообразный сигнал: линейный рост от 1 до 255
    e = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        e[i, :] = np.linspace(1, 255, m, dtype=np.int64)
    return e

def _triangular_matrix(n: int, m: int) -> np.ndarray:
    # Треугольный сигнал: рост 1→255 и спад 255→1
    e = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        half = m // 2
        e[i, :half] = np.linspace(1, 255, half, dtype=np.int64)
        e[i, half:] = np.linspace(255, 1, m - half, dtype=np.int64)
    return e

def _exponential_matrix(n: int, m: int) -> np.ndarray:
    # Экспоненциальный сигнал: рост от 1 до 255
    e = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        t = np.linspace(0, 1, m)
        exp_vals = np.exp(t)
        exp_scaled = 1 + (exp_vals - 1) * (254 / (np.e - 1))
        e[i, :] = np.clip(np.round(exp_scaled).astype(np.int64), 1, 255)
    return e

def _slow_changes_matrix(n: int, m: int) -> np.ndarray:
    """
    Медленно меняющееся освещение: один период косинусоиды 1 -> 255 -> 1
    по ширине. Периодична для ДПФ (края стыкуются без разрыва).
    """
    e = np.zeros((n, m), dtype=np.int64)
    j = np.arange(m)
    row = np.round(128 - 127 * np.cos(2 * np.pi * j / m)).astype(np.int64)
    for i in range(n):
        e[i, :] = row
    return e

# Реестр типов матриц
# Типы для R (коэф. отражения)
_r_types = {
    "const": _const_matrix_float,
    "chess": _chess_matrix,
    "black_square": _black_square_matrix,
    "white_square": _white_square_matrix,
    "random": _random_matrix,
    "snowman": _snowman_matrix,
    "cat": _cat_matrix,
}

# Типы для E (освещение)
_e_types = {
    "const": _const_matrix,
    "sawtooth": _sawtooth_matrix,
    "triangular": _triangular_matrix,
    "exponential": _exponential_matrix,
    "slow_changes": _slow_changes_matrix,
}

# Типы по умолчанию
_r_default = _chess_matrix
_e_default = _const_matrix

# Основные функции
def create_matrix(name: str, n: int, m: int, kind: str = "r") -> np.ndarray:
    """
    Создать матрицу заданного типа и размера.

    Параметры:
        name: имя типа матрицы
        n: количество строк
        m: количество столбцов
        kind: тип матрицы — "r" (отражение, float в (R_MIN, 1]) или
              "e" (освещение, int от 1 до 255)

    Возвращает:
        numpy.ndarray: матрица размера n x m (float64 для R, int64 для E)

    Исключения:
        ValueError: если kind не "r" или "e", или если n или m неправильны
    """
    if kind == "r":
        types = _r_types
        default = _r_default
    elif kind == "e":
        types = _e_types
        default = _e_default
    else:
        raise ValueError(f"Неизвестный тип матрицы: '{kind}'. Используйте 'r' или 'e'.")

    if name not in types:
        types = default
    else:
        types = types[name]

    if not isinstance(n, int) or not isinstance(m, int):
        raise ValueError("Параметры n и m должны быть целыми числами")
    if n <= 0 or m <= 0:
        raise ValueError("Параметры n и m должны быть больше нуля")

    return types(n, m)

def create_image(
    e_type: str = "const",
    r_type: str = "chess",
    n: int = 64,
    m: int = 64,
) -> tuple:
    """
    Создать изображение I = E * R.

    Параметры:
        e_type: тип освещения
        r_type: тип отражения
        n: количество строк
        m: количество столбцов

    Возвращает:
        tuple: (e_matrix, r_matrix, image)

        e_matrix: int64, значения 1-255
        r_matrix: float64, значения в (R_MIN, 1]
        image: float64, I = E * R, значения в (0, 255],
        обработка без округления (округление только на выходе фильтра)
    """
    e = create_matrix(e_type, n, m, "e")
    r = create_matrix(r_type, n, m, "r")

    if e.shape != r.shape:
        raise ValueError(
            f"Размеры матриц не совпадают: E={e.shape}, R={r.shape}!"
        )

    image = e * r
    return e, r, image
