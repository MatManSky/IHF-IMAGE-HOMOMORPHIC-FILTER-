"""
Модуль фильтров частотной области для гомоморфной фильтрации.

Архитектура (как словари типов в image.py — без ООП):
    - _filter_types — реестр фильтров: имя -> функция матрицы H(u,v)
    - get_filter(name, n, m) — матрица H
    - apply_filter(name, spectrum) — поэлементное умножение спектра на H
"""

from gmpy2 import mpfr


# Фильтры
def _allpass_matrix(n: int, m: int) -> list:
    """Allpass: H(u,v) = 1 — все частоты проходят без изменений."""
    one = mpfr(1)
    return [[one] * m for _ in range(n)]


# Типы фильтров
_filter_types = {
    "allpass": _allpass_matrix,
}

# Фильтр по умолчанию
DEFAULT_FILTER = "allpass"


def get_filter(name: str, n: int, m: int) -> list:
    """
    Получить матрицу H(u,v) размера n x m.

    Параметры:
        name: имя фильтра из реестра
        n: количество строк
        m: количество столбцов

    Возвращает:
        list: матрица H из mpfr (список списков)

    Исключения:
        ValueError: если name не из реестра
    """
    if name not in _filter_types:
        raise ValueError(
            f"Неизвестный тип фильтра: '{name}'. "
            f"Доступные: {sorted(_filter_types)}."
        )
    return _filter_types[name](n, m)


def apply_filter(name: str, spectrum: list) -> list:
    """
    Применить фильтр к спектру поэлементно.

    Параметры:
        name: имя фильтра из реестра
        spectrum: спектр (список списков mpc)

    Возвращает:
        list: отфильтрованный спектр (список списков mpc)
    """
    n = len(spectrum)
    m = len(spectrum[0])
    h = get_filter(name, n, m)
    return [[spectrum[i][j] * h[i][j] for j in range(m)] for i in range(n)]
