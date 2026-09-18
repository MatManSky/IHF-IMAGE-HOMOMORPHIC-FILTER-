"""
Модуль фильтров частотной области для гомоморфной фильтрации.

Архитектура (как словари типов в image.py — без ООП):
    - _filter_types — реестр фильтров: имя -> функция матрицы H(u,v)
    - get_filter(name, n, m) — матрица H (единственная точка входа)
"""

import gmpy2
from gmpy2 import mpfr

# Параметры фильтров по умолчанию
D0_DEFAULT = None        # частота среза: None -> min(n,m)/4
ORDER_DEFAULT = 2        # порядок фильтра
RIPPLE_DB_DEFAULT = 1.0  # неравномерность Чебышёва в полосе, дБ

def _allpass_matrix(n: int, m: int) -> list:
    """Allpass: H(u,v) = 1 — все частоты проходят без изменений."""
    one = mpfr(1)
    return [[one] * m for _ in range(n)]

def _distance_matrix(n: int, m: int) -> list:
    """
    Расстояние от частоты (u,v) до DC: D = sqrt(du^2 + dv^2),
    где du = min(u, n-u), dv = min(v, m-v) — частоты зеркальна
    относительно края матрицы
    """
    out = []
    for u in range(n):
        du2 = min(u, n - u)
        du2 *= du2
        row = []
        for v in range(m):
            dv = min(v, m - v)
            row.append(gmpy2.sqrt(mpfr(du2 + dv * dv)))
        out.append(row)
    return out


def _chebyshev_t(order: int, x) -> "mpfr":
    """
    Полином Чебышёва первого рода:
    T[0] = 1, T[1] = x, T[k] = 2*x*T[k-1] – T[k-2]
    """
    if order == 0:
        return mpfr(1)
    t_prev = mpfr(1)
    t = mpfr(x)
    for _ in range(2, order + 1):
        t_prev, t = t, 2 * x * t - t_prev
    return t


def _from_distance(n: int, m: int, h_of_d) -> list:
    """
    Матрица H(u,v) = h(D(u,v)) — формула от расстояния до DC.
    """
    return [[h_of_d(d) for d in dist_row] for dist_row in _distance_matrix(n, m)]


def _complement(lp: list) -> list:
    """
    Дополнение до 1: H = 1 − ФНЧ (получение ФВЧ из ФНЧ).

    Нужно, чтобы не терялись и не дублировались частоты
    """
    one = mpfr(1)
    return [[one - h for h in row] for row in lp]


def _cutoff(d0, n: int, m: int):
    """
    Частота среза D0: явное значение — в пикселях,
    None (по умолчанию) — четверть min(n, m).
    """
    if d0 is None:
        return mpfr(min(n, m)) / 4
    return mpfr(d0)

# ФНЧ (низкие частоты — освещение)
def _lp_butterworth_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
    order: int = ORDER_DEFAULT,
) -> list:
    """
    ФНЧ Баттерворта: H = 1 / (1 + (D/D0)^(2*order)).
    Максимально гладкая АЧХ в полосе пропускания (без пульсаций);
    """
    d0 = _cutoff(d0, n, m)
    return _from_distance(n, m, lambda d: 1 / (1 + (d / d0) ** (2 * order)))


def _lp_chebyshev1_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
    order: int = ORDER_DEFAULT,
    ripple_db: float = RIPPLE_DB_DEFAULT,
) -> list:
    """
    ФНЧ Чебышёва первого рода:
    |H| = 1 / sqrt(1 + eps^2 * T_order^2(D/D0)),
    eps = sqrt(10^(ripple_db/10) - 1)
    Равноволновая пульсация в полосе пропускания: усиление
    """
    d0 = _cutoff(d0, n, m)
    eps2 = mpfr(10) ** (mpfr(ripple_db) / 10) - 1
    return _from_distance(
        n,
        m,
        lambda d: 1 / gmpy2.sqrt(
            1 + eps2 * _chebyshev_t(order, d / d0) ** 2
        ),
    )


def _lp_gaussian_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
) -> list:
    """
    ФНЧ Гаусса: H = exp(-D^2 / (2*D0^2)), на границе D = D0 = 1/sqrt(e) = ~0.607.
    Самый гладкий спад АЧХ, без пульсаций в полосе пропускания.
    """
    d0 = _cutoff(d0, n, m)
    inv_2d0sq = 1 / (2 * d0 * d0)
    return _from_distance(n, m, lambda d: gmpy2.exp(-(d * d) * inv_2d0sq))


# ФВЧ (высокие частоты — коэффициент отражения)
def _hp_butterworth_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
    order: int = ORDER_DEFAULT,
) -> list:
    """ФВЧ Баттерворта: H = 1 − ФНЧ Баттерворта."""
    return _complement(_lp_butterworth_matrix(n, m, d0, order))


def _hp_chebyshev1_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
    order: int = ORDER_DEFAULT,
    ripple_db: float = RIPPLE_DB_DEFAULT,
) -> list:
    """ФВЧ Чебышёва первого рода: H = 1 − ФНЧ Чебышёва."""
    return _complement(_lp_chebyshev1_matrix(n, m, d0, order, ripple_db))


def _hp_gaussian_matrix(
    n: int,
    m: int,
    d0: float = D0_DEFAULT,
) -> list:
    """ФВЧ Гаусса: H = 1 − ФНЧ Гаусса."""
    return _complement(_lp_gaussian_matrix(n, m, d0))

# Типы фильтров
_filter_types = {
    "allpass": _allpass_matrix,
    "lp_butterworth": _lp_butterworth_matrix,
    "hp_butterworth": _hp_butterworth_matrix,
    "lp_chebyshev1": _lp_chebyshev1_matrix,
    "hp_chebyshev1": _hp_chebyshev1_matrix,
    "lp_gaussian": _lp_gaussian_matrix,
    "hp_gaussian": _hp_gaussian_matrix,
}

# Фильтры по умолчанию
DEFAULT_LF_FILTER = "lp_butterworth"
DEFAULT_HF_FILTER = "hp_butterworth"


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
