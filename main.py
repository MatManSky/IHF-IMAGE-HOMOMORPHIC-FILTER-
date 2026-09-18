"""
Главный модуль для тестирования гомоморфного фильтра.

Примеры:
    - test_identity_16_16
    - test_butterworth_16_16
    - test_chebyshev1_16_16
    - test_gaussian_16_16
"""

import numpy as np
from image import (
    create_image,
)
from filter import homomorphic_filter


def _print_matrix(label: str, matrix: np.ndarray) -> None:
    """Вывести матрицу с подписью"""
    print(f"{label}:")
    print(matrix)
    print()

def _run(lf_filter: str, hf_filter: str, lf_gain: float, hf_gain: float) -> None:
    """Создать изображение (E: slow_changes, R: chess) и применить фильтр"""
    np.random.seed(42)
    e, r, i = create_image("slow_changes", "chess", 16, 16)
    f = homomorphic_filter(lf_filter, hf_filter, lf_gain, hf_gain)
    filtered = f.apply(i)

    print(f"E: slow_changes, R: chess, n: 16, m: 16, "
          f"{lf_filter}, lf_gain={lf_gain}, hf_gain={hf_gain}")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)

def test_identity_16_16() -> None:
    _run("lp_butterworth", "hp_butterworth", 1.0, 1.0)


def test_butterworth_16_16() -> None:
    _run("lp_butterworth", "hp_butterworth", 0.5, 2.0)


def test_chebyshev1_16_16() -> None:
    _run("lp_chebyshev1", "hp_chebyshev1", 0.5, 2.0)


def test_gaussian_16_16() -> None:
    _run("lp_gaussian", "hp_gaussian", 0.5, 2.0)


if __name__ == "__main__":
    test_identity_16_16()
    test_butterworth_16_16()
    test_chebyshev1_16_16()
    test_gaussian_16_16()
