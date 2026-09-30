"""
Обработка цветной фотографии с целью сделать освещение холоднее.

Вход:  examples/pic3_small.png
Выход: examples/pic3_small_cold.png

lf_gain по (R, G, B) ->  цвет освещения (E)
hf_gain = 1.0        ->  коэф. отражения (R) не меняется

Если lf_gain > 1, то канал становится светлее. 
Сделать «холоднее» = R уменьшить, а B — увеличить.
Шаг небольшой: LF_GAIN = (0.99, 1.0, 1.01).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from color_filter import color_homomorphic_filter
from photo import photo_filter, read_rgb, write_rgb

SRC_PATH = Path(__file__).with_name("pic3_small.png")
DST_PATH = Path(__file__).with_name("pic3_small_cold.png")

LF_GAIN = (0.99, 1.0, 1.01)  # по каналам R, G, B: чуть холоднее
HF_GAIN = 1.0
# Оставляем по умолчанию
D0 = None
ORDER = 2
PAD = "reflect"


def balance(values: np.ndarray) -> str:
    # Средние по каналам и холодность кадра (B - R: больше — холоднее)
    means = [float(values[:, :, k].mean()) for k in range(3)]
    return (f"R={means[0]:6.2f} G={means[1]:6.2f} B={means[2]:6.2f}  "
            f"B-R={means[2] - means[0]:+6.2f}")


def main() -> None:
    photo = read_rgb(SRC_PATH)
    n, m, _ = photo.shape
    print(f"Прочитан {SRC_PATH.name}: {m}x{n} px, 3 канала")
    print(f"1. вход      : {balance(photo)}")

    f = color_homomorphic_filter(
        lf_gain=LF_GAIN, hf_gain=HF_GAIN, d0=D0, order=ORDER, pad=PAD
    )

    t0 = time.time()
    filtered = f.apply(photo)
    print(f"mpfr 256 бит, pad={PAD}: {time.time() - t0:.0f} с")
    print(f"2. фильтрован: {balance(filtered.astype(np.float64))}")

    out = photo_filter.to_uint8(filtered) # Растяжка совместная
    print(f"3. вывод     : {balance(out.astype(np.float64))}")

    write_rgb(DST_PATH, out)
    print(f"Записан {DST_PATH.name}: {m}x{n} px, "
          f"{DST_PATH.stat().st_size / 1000:.0f} КБ")


if __name__ == "__main__":
    main()
