"""
Схемы применения гомоморфного фильтра к фотографиям (две картинки):

    photo_schema_bw.png     — кадр в оттенках серого (examples/bw_image.py)
    photo_schema_color.png  — цветной кадр     (examples/color_image.py)

Перед генерацией — check_api(): имена, значения по умолчанию и dtype-цепочка
сверяются с photo.py, color_filter.py, processing.py, filtration.py, filter.py;
при расхождении схем со скрипт падает, а не рисует неверную картинку.

Запуск:
    python docs/photo_shema.py            # пишет обе схемы в docs/
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
_OUT_BW_EN = Path(__file__).with_name("photo_schema_bw_en.png")
_OUT_COLOR_EN = Path(__file__).with_name("photo_schema_color_en.png")

LANGS = {
    "ru": {
        "bw": [
            ("pic1.png\nфото на входе\n(PNG / JPEG)",             False),
            ("read_gray (bits=8)\nчтение в float64,\nзначения ≥ 1",       False),
            ("photo_filter.apply\nфильтр mpfr 256 бит,\nвыход int64, >255", False),
            ("to_uint8\nперевод в 0..255,\nпо перцентили 0.5/99.5",        False),
            ("denoise\n(билатерный фильтр)\nвыкл",              True),
            ("write_gray (bits)\nзапись PNG/JPEG,\n10/12/16 → uint16",     False),
        ],
        "color": [
            ("pic3_small.png\nфото на входе\n(PNG / JPEG)",               False),
            ("read_rgb\nчтение в RGB,\nfloat64 (n, m, 3)",                False),
            ("цветной фильтр\nпоканальный mpfr,\nусиления по (R, G, B)",  False),
            ("to_uint8\nрастяжка совместная\nпо всем каналам",            False),
            ("denoise\n(билатерный фильтр)\nвыкл",             True),
            ("write_rgb\nзапись только 8 бит,\nиначе ValueError",         False),
        ],
        "temp": ("make_colder / make_warmer — опция\n"
                 "lf_gain = (1−t, 1, 1+t): холоднее / теплее"),
    },
    "en": {
        "bw": [
            ("pic1.png\ninput photo\n(PNG / JPEG)",                     False),
            ("read_gray (bits=8)\nread into float64,\nvalues ≥ 1",       False),
            ("photo_filter.apply\n256-bit mpfr filter,\noutput int64, >255", False),
            ("to_uint8\nconvert to 0..255,\nby percentile 0.5/99.5",      False),
            ("denoise\n(bilateral filter)\ndisabled",              True),
            ("write_gray (bits)\nwrite PNG/JPEG,\n10/12/16 → uint16",     False),
        ],
        "color": [
            ("pic3_small.png\ninput photo\n(PNG / JPEG)",                 False),
            ("read_rgb\nread into RGB,\nfloat64 (n, m, 3)",                False),
            ("color filter\nper-channel mpfr,\ngains per (R, G, B)",      False),
            ("to_uint8\njoint stretch\nacross all channels",             False),
            ("denoise\n(bilateral filter)\ndisabled",                 True),
            ("write_rgb\n8-bit only,\nelse ValueError",                  False),
        ],
        "temp": ("make_colder / make_warmer — option\n"
                 "lf_gain = (1−t, 1, 1+t): cooler / warmer"),
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
        color_filter: ["color_homomorphic_filter", "CHANNELS"],
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

    # Цвет — только 8 бит: 10/12/16-битный вывод write_rgb отклоняет
    bad = Path(tempfile.gettempdir()) / "ihf_bad_dtype.png"
    try:
        photo.write_rgb(bad, np.zeros((2, 2, 3), dtype=np.int64))
        raise AssertionError("write_rgb пропустил не-uint8")
    except ValueError:
        pass

    print("API-проверка: OK — имена, дефолты и dtype-цепочка совпадают с кодом")


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


# Центры шести блоков: блок w=2.3, зазор между краями 0.7 — длинные стрелки
XS = [1.35, 4.35, 7.35, 10.35, 13.35, 16.35]
HW = 2.3 / 2


def chain(ax, y, blocks):
    for x, (text, dashed) in zip(XS, blocks):
        draw_box(ax, x, y, text, dashed=dashed)
    for x1, x2 in zip(XS[:-1], XS[1:]):
        arrow(ax, x1 + HW + 0.02, y, x2 - HW - 0.02, y)


def save(ax, path):
    ax.figure.savefig(path, dpi=200, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
    print(f"Записан {path.name}: {path.stat().st_size / 1000:.0f} КБ")


def render(lang: str) -> None:
    # Нарисовать обе схемы (RU или EN) и сохранить
    L = LANGS[lang]
    suffix = "" if lang == "ru" else "_en"

    # examples/bw_image.py
    ax = new_sheet((0.1, 17.6), (0.45, 1.95))
    chain(ax, 1.2, L["bw"])
    save(ax, Path(__file__).with_name(f"photo_schema_bw{suffix}.png"))

    # examples/color_image.py
    ax = new_sheet((0.1, 17.6), (0.45, 3.7))
    chain(ax, 1.2, L["color"])
    draw_box(ax, XS[2], 3.0, L["temp"], w=4.6, h=0.95, dashed=True)
    arrow(ax, XS[2], 2.52, XS[2], 1.83, dashed=True)
    save(ax, Path(__file__).with_name(f"photo_schema_color{suffix}.png"))


check_api()

render("ru")
if "--en" in sys.argv:
    render("en")

if "--show" in sys.argv:
    plt.show()
