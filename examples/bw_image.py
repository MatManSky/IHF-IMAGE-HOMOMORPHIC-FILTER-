"""
Пример обработки ЧБ фотографии.

Вход:  examples/pic1.png
Выход: examples/pic1_processed.png
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo import photo_filter, read_gray, write_gray 
from filter import _print_dataset

SRC_PATH = Path(__file__).with_name("pic1.png")
DST_PATH = Path(__file__).with_name("pic1_processed.png")

LF_GAIN = 0.75     # Усиление низких частот (освещённость)
HF_GAIN = 2.0      # Усиление высоких частот (коэффициент отражения)
D0 = 30.0          # частота среза, пикселей
ORDER = 2
PAD = "reflect"


def main() -> None:
    photo = read_gray(SRC_PATH)
    n, m = photo.shape
    print(f"Прочитан {SRC_PATH.name}: {m}x{n} px")
    #_print_dataset("1. input", photo)

    f = photo_filter(
        lf_gain=LF_GAIN, hf_gain=HF_GAIN, d0=D0, order=ORDER, pad=PAD
    )
    t0 = time.time()
    filtered = f.apply(photo)
    print(f"mpfr 256 бит, pad={PAD}: {time.time() - t0:.0f} с")
    #_print_dataset("2. filtered", filtered)

    out = f.to_uint8(filtered)
    write_gray(DST_PATH, out)
    #_print_dataset("3. output", out)
    print(f"Записан {DST_PATH.name}: {m}x{n} px, "
          f"{DST_PATH.stat().st_size / 1000:.0f} КБ")


if __name__ == "__main__":
    main()
