"""
Главный модуль для тестирования гомоморфного фильтра.

Содержит тесты с таблицами E, R, I, filtered:
    - test_const_chess_64x64
    - test_sawtooth_const_64x64
    - test_const_random_16x16
    - test_sawtooth_chess_64x64
    - test_triangular_chess_16x16
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


def test_const_chess_64_64() -> None:
    """E: const, R: chess, n: 64, m: 64"""
    np.random.seed(42)

    e, r, i = create_image("const", "chess", 64, 64)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: const, R: chess, n: 64, m: 64")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()

def test_const_const_64_64() -> None:
    """E: const, R: const, n: 64, m: 64"""
    np.random.seed(42)

    e, r, i = create_image("const", "const", 64, 64)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: const, R: const, n: 64, m: 64")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()

def test_sawtooth_const_64_64() -> None:
    """E: sawtooth, R: const, n: 64, m: 64"""
    np.random.seed(42)

    e, r, i = create_image("sawtooth", "const", 64, 64)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: sawtooth, R: const, n: 64, m: 64")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()

def test_const_random_16_16() -> None:
    """E: const, R: random, n: 16, m: 16"""
    np.random.seed(42)

    e, r, i = create_image("const", "random", 16, 16)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: const, R: random, n: 16, m: 16")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()


def test_sawtooth_chess_64_64() -> None:
    """E: sawtooth, R: chess, n: 64, m: 64"""
    np.random.seed(42)

    e, r, i = create_image("sawtooth", "chess", 64, 64)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: sawtooth, R: chess, n: 64, m: 64")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()


def test_triangular_chess_16_16() -> None:
    """E: triangular, R: chess, n: 16, m: 16"""
    np.random.seed(42)

    e, r, i = create_image("triangular", "chess", 16, 16)
    f = homomorphic_filter()
    filtered = f.apply(i)

    print(f"E: triangular, R: chess, n: 16, m: 16")
    _print_matrix("E", e)
    _print_matrix("R", r)
    _print_matrix("I", i)
    _print_matrix("filtered", filtered)
    print()

if __name__ == "__main__":
    test_sawtooth_const_64_64()
    test_const_chess_64_64()
    test_const_random_16_16()
    test_sawtooth_chess_64_64()
    test_triangular_chess_16_16()