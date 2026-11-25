"""
Схемы применения гомоморфного фильтра к фотографиям (три картинки):

    photo_schema_bw.png       — кадр в оттенках серого (examples/bw_image.py)
    photo_schema_color.png    — поканальный цвет
                                (examples/color_image.py)
    photo_schema_ycbcr.png    — цвет по яркости YCbCr
                                (examples/color_image_ycbcr.py)

Перед генерацией — check_api(): имена, значения по умолчанию и dtype-цепочка
сверяются с photo.py, color_filter.py, processing.py, filtration.py, filter.py;
при расхождении схем со скрипт падает, а не рисует неверную картинку.

Запуск:
    python docs/photo_shema.py            # пишет все три схемы в docs/
    python docs/photo_shema.py --en       # + _en.png версии
    python docs/photo_shema.py --show     # и открывает окна просмотра
"""

import inspect
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_OUT_BW = Path(__file__).with_name("photo_schema_bw.png")
_OUT_COLOR = Path(__file__).with_name("photo_schema_color.png")
_OUT_YCBCR = Path(__file__).with_name("photo_schema_ycbcr.png")
_OUT_BW_EN = Path(__file__).with_name("photo_schema_bw_en.png")
_OUT_COLOR_EN = Path(__file__).with_name("photo_schema_color_en.png")
_OUT_YCBCR_EN = Path(__file__).with_name("photo_schema_ycbcr_en.png")

LANGS = {
    "ru": {
        "bw": [
            "pic1.png\nфото на входе\n(PNG / JPEG)",
            "read_gray (bits=8)\nчтение в float64,\nзначения ≥ 1",
            "photo_filter.apply\nфильтр mpfr 256 бит,\nвыход int64, >255",
            "to_uint8\nперевод в 0..255,\nпо перцентили 0.5/99.5",
            "write_gray (bits)\nзапись PNG/JPEG,\n10/12/16 → uint16",
        ],
        "color": [
            "pic3_small.png\nфото на входе\n(PNG / JPEG)",
            "read_rgb\nRGB или RGBA:\nfloat64 (n, m, 3/4)",
            "цветной фильтр\nпоканальный mpfr,\nусиления по (R, G, B)",
            "to_uint8\nрастяжка по RGB,\nальфа без изменений",
            "write_rgb\n8 бит, RGB или\nRGBA (n, m, 4)",
        ],
        "ycbcr": [
            "pic3_small.png\nфото на входе\n(PNG / JPEG)",
            "read_rgb\nRGB или RGBA:\nfloat64 (n, m, 3/4)",
            "BT.601 в mpfr,\nфильтр только Y,\nCb/Cr без изменений",
            "to_uint8\npercentile_ycbcr:\nрастяжка только по Y",
            "write_rgb\n8 бит, RGB или\nRGBA (n, m, 4)",
        ],
        "temp": ("make_colder / make_warmer — опция\n"
                 "lf_gain = (1−t, 1, 1+t): холоднее / теплее"),
        "ycbcr_note": ("усиления — только скаляр (поканальные — ValueError)"),
        "denoise": ('denoise "before"',
                    'denoise "after"'),
    },
    "en": {
        "bw": [
            "pic1.png\ninput photo\n(PNG / JPEG)",
            "read_gray (bits=8)\nread into float64,\nvalues ≥ 1",
            "photo_filter.apply\n256-bit mpfr filter,\noutput int64, >255",
            "to_uint8\nconvert to 0..255,\nby percentile 0.5/99.5",
            "write_gray (bits)\nwrite PNG/JPEG,\n10/12/16 → uint16",
        ],
        "color": [
            "pic3_small.png\ninput photo\n(PNG / JPEG)",
            "read_rgb\nRGB or RGBA:\nfloat64 (n, m, 3/4)",
            "color filter\nper-channel mpfr,\ngains per (R, G, B)",
            "to_uint8\nstretch on RGB,\nalpha untouched",
            "write_rgb\n8-bit, RGB or\nRGBA (n, m, 4)",
        ],
        "ycbcr": [
            "pic3_small.png\ninput photo\n(PNG / JPEG)",
            "read_rgb\nRGB or RGBA:\nfloat64 (n, m, 3/4)",
            "BT.601 conversion\nin mpfr, filter Y,\nCb/Cr untouched",
            "to_uint8\npercentile_ycbcr:\nstretch on Y only",
            "write_rgb\n8-bit, RGB or\nRGBA (n, m, 4)",
        ],
        "temp": ("make_colder / make_warmer — option\n"
                 "lf_gain = (1−t, 1, 1+t): cooler / warmer"),
        "ycbcr_note": ("scalar gains only (per-channel raises ValueError)"),
        "denoise": ('denoise "before"',
                    'denoise "after"'),
    },
}

BLACK = "black"
WHITE = "white"


def check_api() -> None:
    # Проверить, что обе схемы соответствуют коду
    import numpy as np

    import color_filter
    import filtration
    import filter as ihf_filter
    import photo
    import processing

    needed = {
        photo: ["read_gray", "write_gray", "read_rgb", "write_rgb",
                "photo_filter", "bilateral", "denoise"],
        color_filter: ["color_homomorphic_filter", "CHANNELS",
                       "rgb_to_ycbcr", "ycbcr_to_rgb"],
        processing: ["temperature_gains", "make_colder", "make_warmer"],
        filtration: ["DEFAULT_LF_FILTER", "D0_DEFAULT"],
        ihf_filter: ["homomorphic_filter", "MIN_VALUE"],
    }
    for module, names in needed.items():
        for name in names:
            if not hasattr(module, name):
                raise AssertionError(
                    f"схема ссылается на {module.__name__}.{name}, "
                    f"а в коде такого нет")

    sig = inspect.signature

    # Подписанные на схемах значения по умолчанию
    assert sig(photo.read_gray).parameters["bits"].default == 8
    assert sig(photo.write_gray).parameters["bits"].default == 8
    assert sig(photo.denoise).parameters["enabled"].default is False
    assert (sig(photo.photo_filter.__init__).parameters["pad"].default
            == "reflect")
    assert processing.temperature_gains(0.01) == (0.99, 1.0, 1.01)

    assert {"before", "after"} <= set(photo.DENOISE_PLACES)
    assert (sig(photo.photo_filter.__init__).parameters["filter_place"]
            .default == photo.DENOISE_PLACE_DEFAULT)
    assert "filter_place" in sig(photo.photo_filter.process).parameters
    for lang in LANGS:
        for text in LANGS[lang]["denoise"]:
            place = text.split('"')[1]
            assert place in photo.DENOISE_PLACES, (lang, place)

    # ycbcr-режим: параметр mode + отдельная растяжка по яркости
    assert "mode" in sig(
        color_filter.color_homomorphic_filter.__init__).parameters
    assert "ycbcr" in color_filter._COLOR_MODES
    assert "percentile_ycbcr" in photo._output_types

    # Цвет — только 8 бит: 10/12/16-битный вывод write_rgb отклоняет
    bad = Path(tempfile.gettempdir()) / "ihf_bad_dtype.png"
    try:
        photo.write_rgb(bad, np.zeros((2, 2, 3), dtype=np.int64))
        raise AssertionError("write_rgb пропустил не-uint8")
    except ValueError:
        pass

    print("API-проверка: OK — имена, дефолты и dtype-цепочка "
          "совпадают с кодом")


figs = []


def new_sheet(xlim, ylim):
    # Холст подогнан по содержимому: поля ~0.2 единицы, как в shema.py
    w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    figs.append(fig)
    return ax


def draw_box(ax, x, y, text, w=2.3, h=1.2, fs=9.5, dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.add_patch(FancyBboxPatch(
        (x - w/2 + 0.06, y - h/2 - 0.06), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc="#00000018", ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc=WHITE, ec=BLACK, lw=1.5, linestyle=ls, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            zorder=4, color=BLACK, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, lw=1.7, dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
        color=BLACK, lw=lw, linestyle=ls, zorder=1))


# Ряд из пяти обязательных стадий: блок w=2.3, зазор между краями 0.7
XS = [1.35, 4.35, 7.35, 10.35, 13.35]
HW = 2.3 / 2
Y_CHAIN = 1.2       # обязательные стадии
Y_DEN = 2.85        # пунктирные врезки denoise — над рядом
Y_NOTE = -0.75      # пунктирное примечание — под рядом
X_LIM = (0.1, XS[-1] + HW + 0.15)
# Точки вставки denoise: середины промежутков read->фильтр, to_uint8->write
DEN_X = [(XS[1] + XS[2]) / 2, (XS[3] + XS[4]) / 2]


def chain(ax, y, blocks):
    for x, text in zip(XS, blocks):
        draw_box(ax, x, y, text)
    for x1, x2 in zip(XS[:-1], XS[1:]):
        arrow(ax, x1 + HW + 0.02, y, x2 - HW - 0.02, y)


def denoise_insets(ax, labels):
    for x, text in zip(DEN_X, labels):
        draw_box(ax, x, Y_DEN, text, w=2.9, h=1.0, fs=9.0, dashed=True)
        arrow(ax, x, Y_DEN - 0.52, x, Y_CHAIN + 0.13, dashed=True)


def note_below(ax, text, w):
    draw_box(ax, XS[2], Y_NOTE, text, w=w, h=0.95, dashed=True)
    arrow(ax, XS[2], Y_NOTE + 0.48, XS[2], Y_CHAIN - 0.62, dashed=True)


def save(ax, path):
    ax.figure.savefig(path, dpi=200, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
    print(f"Записан {path.name}: {path.stat().st_size / 1000:.0f} КБ")


def render(lang: str) -> None:
    # Нарисовать схемы (RU или EN) и сохранить
    L = LANGS[lang]
    suffix = "" if lang == "ru" else "_en"
    top = Y_DEN + 0.65
    bottom = Y_NOTE - 0.65

    # examples/bw_image.py
    ax = new_sheet(X_LIM, (0.45, top))
    chain(ax, Y_CHAIN, L["bw"])
    denoise_insets(ax, L["denoise"])
    save(ax, Path(__file__).with_name(f"photo_schema_bw{suffix}.png"))

    # examples/color_image.py
    ax = new_sheet(X_LIM, (bottom, top))
    chain(ax, Y_CHAIN, L["color"])
    denoise_insets(ax, L["denoise"])
    note_below(ax, L["temp"], 4.6)
    save(ax, Path(__file__).with_name(f"photo_schema_color{suffix}.png"))

    # examples/color_image_ycbcr.py
    ax = new_sheet(X_LIM, (bottom, top))
    chain(ax, Y_CHAIN, L["ycbcr"])
    denoise_insets(ax, L["denoise"])
    note_below(ax, L["ycbcr_note"], 5.6)
    save(ax, Path(__file__).with_name(f"photo_schema_ycbcr{suffix}.png"))


check_api()

render("ru")
if "--en" in sys.argv:
    render("en")

if "--show" in sys.argv:
    plt.show()
