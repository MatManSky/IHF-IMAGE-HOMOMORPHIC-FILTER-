"""
Архитектура программного комплекса: модули и зависимости между ними.

Запуск:
    python docs/architecture.py            # пишет docs/architecture.png
    python docs/architecture.py --en       # + docs/architecture_en.png
    python docs/architecture.py --show     # и открывает окно просмотра
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_OUT_EN = Path(__file__).with_name("architecture_en.png")

LANGS = {
    "ru": {
        "title": "IHF — архитектура программного комплекса "
                 "гомоморфной фильтрации",
        "main": ("main.py", "синтетические сцены\n(I = E·R)"),
        "tests": ("tests/test_all.py",
                  "96 тестов инвариантов;\nsynthetic_images.py —\nсцены для тестов"),
        "processing": ("processing.py", "temperature_gains (R, G, B)\n"
                                        "shift_temperature, balance,\n"
                                        "make_colder, make_warmer"),
        "bw_example": ("examples/bw_image.py", "ЧБ: pic1.png →\n"
                                               "pic1_processed.png"),
        "color_example": ("examples/color_image.py", "цвет: pic3_small.png →\n"
                                                     "pic3_small_cold.png"),
        "image": ("image.py", "create_image() (E, R, I = E·R)\n"
                              "набор типов E и R"),
        "color_filter": ("color_filter.py", "color_homomorphic_filter\n"
                                            "(n, m, 3) — поканально\n"
                                            "усиления: скаляр или (R, G, B)"),
        "photo": ("photo.py", "read_gray/write_gray, read_rgb/write_rgb\n"
                              "photo_filter\n"
                              "to_uint8/10/12/16: no | percentile | gamma\n"
                              "bilateral, denoise — билатерный (выкл)"),
        "filter_core": ("filter.py — ядро гомоморфной фильтрации (mpfr/mpc, 256 бит)",
                        "homomorphic_filter: apply(image, pad, window), apply_info(...)\n"
                        "стадии — как в shema.py: вход → log2 → окно + дополнение → прямое БПФ\n"
                        "(Кули-Тьюки) → ФНЧ × lf_gain и ФВЧ × hf_gain → объединение (НЧ + ВЧ)\n"
                        "→ обратное БПФ → анти-дополнение → exp2 → округление → выход int64\n"
                        "pad = zero | reflect, окна hann | tukey (делятся после ОБПФ);\n"
                        "MIN_VALUE = 1 поднимает неположительную яркость"),
        "filtration": ("filtration.py — функции частотной области (mpfr)",
                       "get_filter: lp/hp Баттерворт · Чебышёв-1 · Гаусс · allpass "
                       "(ФВЧ = 1 − ФНЧ)\n"
                       "get_window: hann · tukey; частотная сетка D(u, v); "
                       "d0 = None → min(n, m)/4\n"
                       "filter вызывает get_filter/get_window; color_filter, photo, "
                       "processing берут имена и параметры по умолчанию"),
        "libs": ("Внешние библиотеки",
                 "numpy — матрицы на входе и выходе (image, filter, color_filter, "
                 "photo, processing) · gmpy2 (mpfr/mpc) — арифметика (filter, filtration)\n"
                 "Pillow — чтение и запись фото (photo) · matplotlib — только "
                 "схемы docs/"),
        "legend": "Пунктир — то, что не входит в релиз "
                  "(внешние библиотеки)",
    },
    "en": {
        "title": "IHF — software architecture for "
                 "homomorphic image filtering",
        "main": ("main.py", "synthetic scenes\n(I = E·R)"),
        "tests": ("tests/test_all.py",
                  "96 invariant tests;\nsynthetic_images.py —\nscene generators"),
        "processing": ("processing.py", "temperature_gains (R, G, B)\n"
                                        "shift_temperature, balance,\n"
                                        "make_colder, make_warmer"),
        "bw_example": ("examples/bw_image.py", "BW: pic1.png →\n"
                                               "pic1_processed.png"),
        "color_example": ("examples/color_image.py", "color: pic3_small.png →\n"
                                                     "pic3_small_cold.png"),
        "image": ("image.py", "create_image() (E, R, I = E·R)\n"
                              "E and R types set"),
        "color_filter": ("color_filter.py", "color_homomorphic_filter\n"
                                            "(n, m, 3) per-channel\n"
                                            "gains: scalar or (R, G, B)"),
        "photo": ("photo.py", "read_gray/write_gray, read_rgb/write_rgb\n"
                              "photo_filter\n"
                              "to_uint8/10/12/16: no | percentile | gamma\n"
                              "bilateral, denoise — bilateral (off)"),
        "filter_core": ("filter.py — homomorphic filtering core (mpfr/mpc, 256 bit)",
                        "homomorphic_filter: apply(image, pad, window), apply_info(...)\n"
                        "stages — as in shema.py: input → log2 → window + padding → direct DFT\n"
                        "(Cooley-Tukey) → LPF × lf_gain and HPF × hf_gain → combine (LF + HF)\n"
                        "→ inverse DFT → anti-padding → exp2 → round → output int64\n"
                        "pad = zero | reflect, windows hann | tukey (divided after IDFT);\n"
                        "MIN_VALUE = 1 lifts non-positive brightness"),
        "filtration": ("filtration.py — frequency-domain functions (mpfr)",
                       "get_filter: lp/hp Butterworth · Chebyshev-1 · Gaussian · allpass "
                       "(HPF = 1 − LPF)\n"
                       "get_window: hann · tukey; frequency grid D(u, v); "
                       "d0 = None → min(n, m)/4\n"
                       "filter calls get_filter/get_window; color_filter, photo, "
                       "processing pick names and defaults"),
        "libs": ("External libraries",
                 "numpy — input/output matrices (image, filter, color_filter, "
                 "photo, processing) · gmpy2 (mpfr/mpc) — arithmetic (filter, filtration)\n"
                 "Pillow — photo read/write (photo) · matplotlib — docs/ schemes only"),
        "legend": "Dashed — excluded from release "
                  "(external libraries)",
    },
}

OUT_PATH = Path(__file__).with_name("architecture.png")

BLACK = "black"
WHITE = "white"
GRAY = "#555555"
SHADOW = "#00000018"

TITLE_FS = 10.5
BODY_FS = 8.5


def _render(lang: str, fig, ax) -> None:
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 11.6)
    ax.set_aspect("equal")
    ax.axis("off")
    L = LANGS[lang]

    def draw_box(x, y, title, body="", w=3.4, h=1.4, fs=TITLE_FS, dashed=False,
                 body_dy=0.22):
        """
        Скруглённый блок с мягкой тенью: заголовок (модуль) и тело (что делает).
        x, y — координаты центра.
        w, h — ширина и высота.
        dashed=True — блок вне релиза (тесты, внешние библиотеки).
        body_dy — сдвиг тела вниз от центра (для низких блоков меньше).
    """
        ls = (0, (5, 3)) if dashed else 'solid'
        ax.add_patch(FancyBboxPatch(
            (x - w/2 + 0.06, y - h/2 - 0.06), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=SHADOW, ec="none", zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=WHITE, ec=BLACK, lw=1.5, linestyle=ls, zorder=3))
        if body:
            ax.text(x, y + h/2 - 0.34, title, ha='center', va='center',
                    fontsize=fs, fontweight='bold', zorder=4, color=BLACK)
            ax.text(x, y - body_dy, body, ha='center', va='center',
                    fontsize=BODY_FS, zorder=4, color=BLACK, linespacing=1.5)
        else:
            ax.text(x, y, title, ha='center', va='center', fontsize=fs,
                    fontweight='bold', zorder=4, color=BLACK)

    def arrow(x1, y1, x2, y2, rad=0.0, lw=1.6, dashed=False):
        # Стрелка зависимости
        ls = (0, (5, 3)) if dashed else 'solid'
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>', mutation_scale=14,
            color=BLACK, lw=lw, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}", zorder=1))

    def route(points, lw=1.6, dashed=False):
        ls = (0, (5, 3)) if dashed else 'solid'
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=BLACK, lw=lw, linestyle=ls,
                solid_joinstyle="miter", zorder=1)
        arrow(*points[-2], *points[-1], lw=lw, dashed=dashed)

    # Заголовок
    ax.text(10.5, 11.3, L["title"],
            ha='center', va='center', fontsize=14, fontweight='bold', color=BLACK)

    # Верхний ряд
    y0 = 9.7
    draw_box(2.7, y0, *L["main"])
    draw_box(6.7, y0, *L["tests"])
    draw_box(10.4, y0, *L["processing"], w=2.8, h=1.6)
    draw_box(14.3, y0, *L["bw_example"], w=3.8)
    draw_box(18.5, y0, *L["color_example"], w=3.7)

    # Второй ряд
    y1 = 7.4
    draw_box(2.7, y1, *L["image"], h=1.6)
    draw_box(8.6, y1, *L["color_filter"], w=4.4, h=1.6)
    draw_box(14.6, y1, *L["photo"], w=4.4, h=1.8)

    # Гомоморфный фильтр
    y2 = 4.8
    H2 = 2.8
    TOP2, BOT2 = y2 + H2 / 2, y2 - H2 / 2
    draw_box(10.0, y2, *L["filter_core"], w=14.0, h=H2, fs=12, body_dy=0.44)

    # Функции частотной области
    y3 = 2.3
    draw_box(10.0, y3, *L["filtration"], w=14.0, h=1.5, fs=11)

    # Внешние библиотеки
    y4 = 0.62
    draw_box(10.9, y4, *L["libs"], w=19.8, h=1.1, fs=10, dashed=True, body_dy=0.12)

    # Зависимости
    arrow(2.7, y1 + 0.8, 2.7, y0 - 0.7)
    arrow(7.4, y1 + 0.8, 4.0, y0 - 0.7)
    arrow(9.2, y1 + 0.8, 9.6, y0 - 0.8)
    arrow(14.2, y1 + 0.9, 14.5, y0 - 0.7)
    arrow(15.6, y1 + 0.9, 17.4, y0 - 0.7)
    route([(11.3, y0 + 0.8), (11.3, 10.75),
           (18.5, 10.75), (18.5, y0 + 0.7)])
    route([(3.0, y2), (0.5, y2), (0.5, y0), (1.0, y0)])
    route([(11.6, TOP2), (11.6, 8.55),
           (12.8, 8.55), (12.8, y0 - 0.7)])
    arrow(8.6, TOP2, 8.6, y1 - 0.8)
    route([(17.0, y2), (20.2, y2),
           (20.2, 7.0), (16.8, 7.0)])
    arrow(10.0, y4 + 0.55, 10.0, y3 - 0.75)
    arrow(10.0, y3 + 0.75, 10.0, BOT2)

    # Подписи
    ax.text(0.4, 1.3, L["legend"],
            ha='left', va='bottom', fontsize=8.5, color=GRAY)


fig, ax = plt.subplots(figsize=(18, 9.8))
_render("ru", fig, ax)

fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Записан {OUT_PATH.name}: {OUT_PATH.stat().st_size / 1000:.0f} КБ")

if "--en" in sys.argv:
    fig_en, ax_en = plt.subplots(figsize=(18, 9.8))
    _render("en", fig_en, ax_en)
    fig_en.savefig(_OUT_EN, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print(f"Записан {_OUT_EN.name}: {_OUT_EN.stat().st_size / 1000:.0f} КБ")

if "--show" in sys.argv:
    plt.show()
