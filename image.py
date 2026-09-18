"""
Модуль для создания матриц освещения (E), отражения (R) и составных изображений.

Архитектура:
    - _type_registry — словарь с функциями для каждого типа матрицы
    - create_matrix(name, n, m, kind) — создать матрицу (E или R)
    - create_image(e_type, r_type, n, m) — создать изображение I

Модель изображения: I = E * R, где
    - E — матрица освещения, целые числа (int64) от 1 до 255
    - R — коэффициент отражения, float64 в диапазоне (R_MIN, 1], R > 0 строго
    - I — яркость, float64 в диапазоне (0, 255]

0 < E(x,y) <= 255;
0 < R(x,y) <= 1;
I(x,y) = E(x,y) * R(x,y) => 0 < I(x,y) <= 255;
"""

import numpy as np

# Нижняя граница коэффициента отражения (не 0, чтобы log был определён)
R_MIN = 1.0 / 255.0

# Вспомогательные функции
def _chess_matrix(n: int, m: int) -> np.ndarray:
    """Шахматная доска с тремя уровнями отражения (1.0, 0.5, 1/255)."""
    r = np.zeros((n, m), dtype=np.float64)
    levels = (1.0, 0.5, R_MIN)
    for i in range(n):
        for j in range(m):
            idx = (i // 2 + j // 2) % 3
            r[i, j] = levels[idx]
    return r

def _random_matrix(n: int, m: int) -> np.ndarray:
    """Случайные коэффициенты отражения от 1/255 до 1"""
    return np.random.uniform(R_MIN, 1.0, size=(n, m))

def _const_matrix_float(n: int, m: int) -> np.ndarray:
    """Константные коэффициенты отражения — все единицы"""
    return np.ones((n, m), dtype=np.float64)

def _const_matrix(n: int, m: int) -> np.ndarray:
    """Константное освещение — все единицы"""
    return np.ones((n, m), dtype=np.int64)

def _sawtooth_matrix(n: int, m: int) -> np.ndarray:
    """Пилообразный сигнал: линейный рост от 1 до 255"""
    e = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        e[i, :] = np.linspace(1, 255, m, dtype=np.int64)
    return e

def _triangular_matrix(n: int, m: int) -> np.ndarray:
    """Треугольный сигнал: рост 1→255 и спад 255→1"""
    e = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        half = m // 2
        e[i, :half] = np.linspace(1, 255, half, dtype=np.int64)
        e[i, half:] = np.linspace(255, 1, m - half, dtype=np.int64)
    return e

def _exponential_matrix(n: int, m: int) -> np.ndarray:
    """Экспоненциальный сигнал: рост от 1 до 255"""
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
    "random": _random_matrix,
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
