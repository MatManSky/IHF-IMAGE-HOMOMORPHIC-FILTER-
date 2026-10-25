"""
Схема гомоморфного фильтра.

Обработка фотографии целиком (чтение/запись, to_uintN, denoise) — в photo_shema.py,
структура проекта — в architecture.py.

Запуск:
    python docs/shema.py            # пишет docs/homomorphic_schema.png
    python docs/shema.py --en       # + docs/homomorphic_schema_en.png
    python docs/shema.py --show     # и открывает окно просмотра
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_OUT_EN = Path(__file__).with_name("homomorphic_schema_en.png")

LANGS = {
    "ru": {
        "title_in": "Входная\nматрица",
        "title_log": "log₂",
        "title_window": "Окно +\nдополнение",
        "title_fwd": "Прямое\nБПФ",
        "title_lpf": "ФНЧ  × gain",
        "title_hpf": "ФВЧ  × gain",
        "title_merge": "Объединение\n(НЧ + ВЧ)",
        "title_bwd": "Обратное\nБПФ",
        "title_anti": "Анти-дополнение\n+ компенсация окна",
        "title_exp": "exp₂",
        "title_round": "Округление",
        "title_out": "Выходная\nматрица",
    },
    "en": {
        "title_in": "Input\nmatrix",
        "title_log": "log₂",
        "title_window": "Window +\npadding",
        "title_fwd": "Direct\nDFT",
        "title_lpf": "LPF × gain",
        "title_hpf": "HPF × gain",
        "title_merge": "Combine\n(LF + HF)",
        "title_bwd": "Inverse\nDFT",
        "title_anti": "Anti-padding\n+ window comp.",
        "title_exp": "exp₂",
        "title_round": "Round",
        "title_out": "Output\nmatrix",
    },
}

OUT_PATH = Path(__file__).with_name("homomorphic_schema.png")

fig, ax = plt.subplots(figsize=(12.5, 5.7))
ax.set_xlim(0.2, 12.65)   # поля по краям содержимого: справа — по блоку
ax.set_ylim(0.45, 6.1)    # «анти-дополнение», сверху — по верхнему ряду
ax.set_aspect('equal')
ax.axis('off')

BLACK = "black"
WHITE = "white"


def draw_box(x, y, text, w=2.0, h=0.9, fs=11, dashed=False):
    # Скруглённый блок с мягкой тенью. dashed=True — опциональный блок
    ls = (0, (5, 3)) if dashed else 'solid'

    # Небольшая тень
    shadow = FancyBboxPatch(
        (x - w/2 + 0.06, y - h/2 - 0.06), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc="#00000018", ec="none", zorder=2)
    ax.add_patch(shadow)

    # Блок
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc=WHITE, ec=BLACK, lw=1.5, zorder=3, linestyle=ls)
    ax.add_patch(box)

    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            zorder=4, color=BLACK)


def arrow(x1, y1, x2, y2, rad=0.0, lw=1.7, dashed=False):
    # Стрелка. dashed=True — обходная (опциональная) линия
    ls = (0, (5, 3)) if dashed else 'solid'
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=14,
        color=BLACK, lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


# Верхний ряд
y1 = 5.5
L = LANGS["ru"]
draw_box(1.4, y1, L["title_in"], w=2.1)
draw_box(3.6, y1, L["title_log"],               w=1.2, fs=13)
draw_box(6.1, y1, L["title_window"], w=2.0, dashed=True)
draw_box(8.5, y1, L["title_fwd"],        w=1.7)

# Основной (обязательный) путь
arrow(2.45, y1, 3.00, y1)      # Вход → log2
arrow(4.20, y1, 5.10, y1)      # log2 → Окно+дополнение
arrow(7.10, y1, 7.65, y1)      # Окно+дополнение → БПФ


# ФНЧ и ФВЧ
y2 = 4.0
draw_box(7.3, y2, L["title_lpf"], w=1.9)
draw_box(9.3, y2, L["title_hpf"], w=1.9)

arrow(8.50, 5.05, 7.30, 4.45, rad= 0.10)
arrow(8.50, 5.05, 9.30, 4.45, rad=-0.10)


# Объединение
y3 = 2.6
draw_box(8.3, y3, L["title_merge"], w=2.0)
arrow(7.30, 3.55, 7.85, 3.05, rad=-0.10)
arrow(9.30, 3.55, 8.75, 3.05, rad= 0.10)


# Обратное БПФ
draw_box(11.2, y3, L["title_bwd"], w=1.8)
arrow(9.30, y3, 10.30, y3)


# Нижний ряд
y4 = 1.1
draw_box(11.2, y4, L["title_anti"], w=2.4, fs=9.5,
         dashed=True)
draw_box( 8.2, y4, L["title_exp"],               w=1.4, fs=13)
draw_box( 6.0, y4, L["title_round"],         w=1.8)
draw_box( 3.5, y4, L["title_out"],  w=2.0)

arrow(11.20, 2.15, 11.20, 1.55)   # ОБПФ → Анти-дополнение + компенсация окна
arrow(10.00, y4,   8.90, y4)      # → exp2
arrow( 7.50, y4,   6.90, y4)      # exp2 → Округление
arrow( 5.10, y4,   4.50, y4)      # Округление → Выходная матрица


# Английская версия
fig_en, ax_en = plt.subplots(figsize=(12.5, 5.7))
ax_en.set_xlim(0.2, 12.65)
ax_en.set_ylim(0.45, 6.1)
ax_en.set_aspect('equal')
ax_en.axis('off')
L_en = LANGS["en"]


def draw_box_en(x, y, text, w=2.0, h=0.9, fs=11, dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    shadow = FancyBboxPatch(
        (x - w/2 + 0.06, y - h/2 - 0.06), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc="#00000018", ec="none", zorder=2)
    ax_en.add_patch(shadow)
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        fc=WHITE, ec=BLACK, lw=1.5, zorder=3, linestyle=ls)
    ax_en.add_patch(box)
    ax_en.text(x, y, text, ha='center', va='center', fontsize=fs,
               zorder=4, color=BLACK)


def arrow_en(x1, y1, x2, y2, rad=0.0, lw=1.7, dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=14,
        color=BLACK, lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax_en.add_patch(a)


# Top row
y1 = 5.5
draw_box_en(1.4, y1, L_en["title_in"], w=2.1)
draw_box_en(3.6, y1, L_en["title_log"],               w=1.2, fs=13)
draw_box_en(6.1, y1, L_en["title_window"], w=2.0, dashed=True)
draw_box_en(8.5, y1, L_en["title_fwd"],        w=1.7)

arrow_en(2.45, y1, 3.00, y1)
arrow_en(4.20, y1, 5.10, y1)
arrow_en(7.10, y1, 7.65, y1)

# LPF / HPF
y2 = 4.0
draw_box_en(7.3, y2, L_en["title_lpf"], w=1.9)
draw_box_en(9.3, y2, L_en["title_hpf"], w=1.9)
arrow_en(8.50, 5.05, 7.30, 4.45, rad= 0.10)
arrow_en(8.50, 5.05, 9.30, 4.45, rad=-0.10)

# Merge
y3 = 2.6
draw_box_en(8.3, y3, L_en["title_merge"], w=2.0)
arrow_en(7.30, 3.55, 7.85, 3.05, rad=-0.10)
arrow_en(9.30, 3.55, 8.75, 3.05, rad= 0.10)
draw_box_en(11.2, y3, L_en["title_bwd"], w=1.8)
arrow_en(9.30, y3, 10.30, y3)

# Bottom row
y4 = 1.1
draw_box_en(11.2, y4, L_en["title_anti"], w=2.4, fs=9.5, dashed=True)
draw_box_en( 8.2, y4, L_en["title_exp"],               w=1.4, fs=13)
draw_box_en( 6.0, y4, L_en["title_round"],         w=1.8)
draw_box_en( 3.5, y4, L_en["title_out"],  w=2.0)
arrow_en(11.20, 2.15, 11.20, 1.55)
arrow_en(10.00, y4,   8.90, y4)
arrow_en( 7.50, y4,   6.90, y4)
arrow_en( 5.10, y4,   4.50, y4)

fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Записан {OUT_PATH.name}: {OUT_PATH.stat().st_size / 1000:.0f} КБ")

if "--en" in sys.argv:
    fig_en.savefig(_OUT_EN, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print(f"Записан {_OUT_EN.name}: {_OUT_EN.stat().st_size / 1000:.0f} КБ")

if "--show" in sys.argv:
    plt.show()