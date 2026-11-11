"""
Обработка цветной фотографии в YCbCr-режиме.

Гомоморфная фильтрация применяется только к яркости (Y-канал),
Cb и Cr сохраняются без изменений. Результат: контраст улучшен,
цвет не искажается (в отличие от поканального RGB, где хроматика
деградирует при скалярных усилениях).

Растяжка диапазона — по яркости (output="percentile_ycbcr"):
совместная растяжка всех каналов умножила бы межканальные разности
на коэффициент растяжки и раздула исходный оттенок.

Вход:  examples/pic3_small.png
Выход: examples/pic3_small_ycbcr.png
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo import photo_filter, read_rgb, write_rgb, denoise
from color_filter import color_homomorphic_filter
from processing import balance

SRC_PATH = Path(__file__).with_name("pic3_small.png")
DST_PATH = Path(__file__).with_name("pic3_small_ycbcr.png")

LF_GAIN = 0.75     # Ослабить освещение (НЧ)
HF_GAIN = 2.0      # Усилить детали (ВЧ)
D0 = None          # частота среза — автоматически
ORDER = 2
PAD = "reflect"
DENOISE = True    # шумоподавление итогового кадра


def main() -> None:
    photo = read_rgb(SRC_PATH)
    n, m, _ = photo.shape
    print(f"Прочитан {SRC_PATH.name}: {m}x{n} px, 3 канала")
    print(f"1. вход        : {balance(photo)}")

    f = color_homomorphic_filter(
        lf_gain=LF_GAIN, hf_gain=HF_GAIN, d0=D0, order=ORDER,
        pad=PAD, mode="ycbcr",
    )
    t0 = time.time()
    filtered = f.apply(photo)
    elapsed = time.time() - t0
    print(f"ycbcr, mpfr 256 бит, pad={PAD}: {elapsed:.0f} с")
    print(f"2. фильтрован  : {balance(filtered.astype(np.float64))}")

    out = denoise(
        photo_filter.to_uint8(filtered, output="percentile_ycbcr"),
        DENOISE,
    )
    print(f"3. вывод       : {balance(out.astype(np.float64))}")

    write_rgb(DST_PATH, out)
    print(f"Записан {DST_PATH.name}: {m}x{n} px, "
          f"{DST_PATH.stat().st_size / 1000:.0f} КБ, "
          f"шумоподавление {'вкл' if DENOISE else 'выкл'}")


if __name__ == "__main__":
    main()
