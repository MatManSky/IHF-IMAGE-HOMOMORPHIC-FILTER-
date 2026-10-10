"""
Обработка цветной фотографии с целью сделать освещение холоднее.

Вход:  examples/pic3_small.png
Выход: examples/pic3_small_cold.png

Температуру меняет processing.make_colder: усиливается только НЧ-часть
спектра (освещённость E), ВЧ-часть (отражение R) не трогается.

«Холоднее» = R уменьшить, а B — увеличить.
Шаг: TEMPERATURE = 0.01 => LF_GAIN = (0.99, 1.0, 1.01).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo import photo_filter, read_rgb, write_rgb, denoise
from processing import balance, make_colder

SRC_PATH = Path(__file__).with_name("pic3_small.png")
DST_PATH = Path(__file__).with_name("pic3_small_cold.png")

TEMPERATURE = 0.01  # шаг сдвига НЧ-части
# Параметры фильтра по умолчанию
D0 = None
ORDER = 2
PAD = "reflect"
DENOISE = False  # шумоподавление итогового кадра


def main() -> None:
    photo = read_rgb(SRC_PATH)
    n, m, _ = photo.shape
    print(f"Прочитан {SRC_PATH.name}: {m}x{n} px, 3 канала")
    print(f"1. вход      : {balance(photo)}")

    t0 = time.time()
    filtered = make_colder(photo, TEMPERATURE, d0=D0, order=ORDER, pad=PAD)
    print(f"mpfr 256 бит, pad={PAD}: {time.time() - t0:.0f} с")
    print(f"2. фильтрован: {balance(filtered.astype(np.float64))}")

    out = denoise(photo_filter.to_uint8(filtered), DENOISE)  # Растяжка совместная
    print(f"3. вывод     : {balance(out.astype(np.float64))}")

    write_rgb(DST_PATH, out)
    print(f"Записан {DST_PATH.name}: {m}x{n} px, "
          f"{DST_PATH.stat().st_size / 1000:.0f} КБ, "
          f"шумоподавление {'вкл' if DENOISE else 'выкл'}")


if __name__ == "__main__":
    main()
