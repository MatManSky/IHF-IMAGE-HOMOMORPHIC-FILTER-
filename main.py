"""
Главный модуль для тестирования гомоморфного фильтра.

Примеры:
    - test_identity_16_16
    - test_butterworth_16_16
    - test_chebyshev1_16_16
    - test_gaussian_16_16
    - test_color_8_8
"""

import numpy as np
from image import (
    create_image,
)
from filter import homomorphic_filter
from color_filter import color_homomorphic_filter


def _print_matrix(label: str, matrix: np.ndarray) -> None:
    """Вывести матрицу с подписью"""
    print(f"{label}:")
    print(matrix)
    print()

# Порядок каналов цветной матрицы (axis=2): slow/const/sawtooth -> R/G/B
_CHANNELS = ("R", "G", "B")

def _print_color_matrix(label: str, matrix: np.ndarray) -> None:
    """Вывести цветную матрицу (n, m, 3) по каналам с подписями"""
    for k, name in enumerate(_CHANNELS):
        _print_matrix(f"{label} (канал {name})", matrix[:, :, k])

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

def test_black_square_info_16_16() -> None:
    # тест всепропускающего фильтра
    e, r, i = create_image("const", "black_square", 16, 16)
    f = homomorphic_filter("allpass", "allpass", 1.0, 0.0)

    print(f"E: const, R: black_square, n: 16, m: 16, allpass, "
          f"lf_gain=1.0, hf_gain=0.0")
    _print_matrix("E", e)
    _print_matrix("R", r)
    f.apply_info(i)


def test_color_8_8() -> None:
    """Цветной вход (8, 8, 3): каналы E разные, R: chess, butterworth 0.5/2.0"""
    np.random.seed(42)
    e_channels, r_channels, i_channels = zip(
        *(create_image(e, "chess", 8, 8)
          for e in ("slow_changes", "const", "sawtooth")))
    color = np.stack(i_channels, axis=2)
    f = color_homomorphic_filter("lp_butterworth", "hp_butterworth", 0.5, 2.0)
    filtered = f.apply(color)

    print("Цвет: E: slow_changes/const/sawtooth (R/G/B), R: chess, "
          "n: 8, m: 8, lp_butterworth, lf_gain=0.5, hf_gain=2.0")
    _print_color_matrix("E", np.stack(e_channels, axis=2))
    _print_matrix("R (общий для всех каналов)", r_channels[0])
    _print_color_matrix("I", color)
    _print_color_matrix("filtered", filtered)


if __name__ == "__main__":
    test_identity_16_16()
    test_butterworth_16_16()
    test_chebyshev1_16_16()
    test_gaussian_16_16()
    test_color_8_8()
    test_black_square_info_16_16()
