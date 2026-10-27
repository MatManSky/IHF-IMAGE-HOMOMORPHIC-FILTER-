"""
Генерация синтетических изображений в оттенках серого

Типы:
  - Снеговик
  - Тень
  - Кот

Для тестов все генераторы принимают size=(h, w) — по умолчанию (1080, 1920).
Генерация:
    python tests/synthetic_images.py           # все в images/synthetic/
    python tests/synthetic_images.py --small    # тестовые 64×64

Запуск:
    python -m pytest tests/ -v


Ссылки:
 1. Рекомендация МСЭ-R BT.601-7. Параметры кодирования студийных
    сигналов цифрового телевидения для стандартных соотношений 4:3 и
    широкоэкранных 16:9 (03/2011). URL:
    https://www.itu.int/rec/R-REC-BT.601 (дата обращения: 17.10.2026);
 2. Гонсалес Р. Цифровая обработка изображений / Р. Гонсалес, Р. Вудс. М.:
    Техносфера, 2012. 1104 с.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_DEFAULT_SIZE = (1080, 1920)  # (height, width)

REC601_R = 0.299
REC601_G = 0.587
REC601_B = 0.114

# Границы градиентного фона
_BG_MIN = 16
_BG_MAX = 235

# Путь к корню проекта
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Вспомогательные функции

def _empty(size=_DEFAULT_SIZE) -> np.ndarray:
    """Белое изображение."""
    return np.full(size, 255, dtype=np.uint8)


def _gradient_background(size=_DEFAULT_SIZE) -> np.ndarray:
    """Плавный горизонтальный градиент от почти чёрного к почти белому."""
    h, w = size
    row = np.linspace(_BG_MIN, _BG_MAX, w, dtype=np.float64)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _rgb_to_gray(r: int, g: int, b: int) -> int:
    """Преобразование RGB в grayscale по Rec. 601."""
    return int(REC601_R * r + REC601_G * g + REC601_B * b)


def _fill_circle(img: np.ndarray, cx: int, cy: int, radius: int,
                 gray: int) -> None:
    """Закрашивает круг однородным оттенком серого."""
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = gray


def _fill_circle_gradient(
    img: np.ndarray,
    cx: int, cy: int, radius: int,
    gray_start: int, gray_end: int,
    direction: str = "tl_br",
) -> None:
    """
    Закрашивает круг с диагональным градиентом яркости.

    direction:
      "tl_br" — от левого-верхнего к правому-нижнему;
      "tr_bl" — от правого-верхнего к левому-нижнему.
    """
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    if direction == "tr_bl":
        dx, dy = -1.0, 1.0
    else:
        dx, dy = 1.0, 1.0

    norm = (dx * dx + dy * dy) ** 0.5
    dx /= norm
    dy /= norm

    proj = (x - cx) * dx + (y - cy) * dy
    t = np.clip((proj + radius) / (2.0 * radius), 0.0, 1.0)

    values = gray_start + (gray_end - gray_start) * t
    img[mask] = values[mask].astype(np.uint8)


def _fill_triangle(img: np.ndarray, v0: tuple, v1: tuple,
                   v2: tuple, gray: int) -> None:
    """Закрашивает треугольник оттенком серого
    (барицентрические координаты)."""
    h, w = img.shape
    y, x = np.ogrid[:h, :w]

    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2

    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(det) < 1e-6:
        return

    lam1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
    lam2 = ((y2 - y0) * (x - x0) + (x0 - x2) * (y - y0)) / det
    lam3 = 1.0 - lam1 - lam2

    mask = (lam1 >= 0) & (lam2 >= 0) & (lam3 >= 0)
    img[mask] = gray


# Снеговик (три шара с диагональным градиентом яркости)

def generate_snowman(size=_DEFAULT_SIZE) -> np.ndarray:
    """
    Снеговик из трёх шаров (снизу вверх: большой → средний → маленький).
    Каждый шар залит диагональным градиентом яркости для объёма.
    Фон — горизонтальный градиент (16→235). Глаза, нос, пуговицы.

    size: (height, width) — пропорциональная геометрия.
    """
    h, w = size
    img = _gradient_background(size)
    cx = w // 2

    # Пропорциональная геометрия (относительно ширины)
    scale = w / 1920.0
    bottom_r = int(230 * scale)
    middle_r = int(170 * scale)
    top_r = int(120 * scale)

    bottom_cy = int(850 * scale)
    middle_cy = bottom_cy - bottom_r - middle_r + int(80 * scale)
    top_cy = middle_cy - middle_r - top_r + int(60 * scale)

    # Шары с диагональным градиентом (свет сверху-слева)
    _fill_circle_gradient(img, cx, bottom_cy, bottom_r,
                          245, 150, "tl_br")
    _fill_circle_gradient(img, cx, middle_cy, middle_r,
                          248, 160, "tl_br")
    _fill_circle_gradient(img, cx, top_cy, top_r,
                          250, 165, "tl_br")

    # Глаза-угольки
    eye_dx = int(42 * scale)
    eye_dy = int(-18 * scale)
    eye_r = int(14 * scale)
    _fill_circle(img, cx - eye_dx, top_cy + eye_dy, eye_r, 20)
    _fill_circle(img, cx + eye_dx, top_cy + eye_dy, eye_r, 20)

    # Нос-морковка
    nose_tip = (cx + int(80 * scale), top_cy + int(12 * scale))
    nose_base = (cx + int(8 * scale), top_cy + int(14 * scale))
    _fill_triangle(img, nose_tip,
                   (nose_base[0], top_cy - int(2 * scale)),
                   (nose_base[0], top_cy + int(26 * scale)), 90)

    # Пуговицы
    _fill_circle(img, cx, middle_cy - int(35 * scale), int(16 * scale), 25)
    _fill_circle(img, cx, middle_cy + int(45 * scale), int(16 * scale), 25)
    _fill_circle(img, cx, bottom_cy - int(120 * scale), int(18 * scale), 25)
    _fill_circle(img, cx, bottom_cy + int(10 * scale), int(18 * scale), 25)

    return img


# -------------------------------------------------------------------------
# Тень — сцена с мультипликативным затенением
# -------------------------------------------------------------------------

def generate_shadowed(size=_DEFAULT_SIZE) -> np.ndarray:
    """
    Сцена с сильным мультипликативным затенением.

    Назначение: проверить, что IHF корректно отделяет освещённость (НЧ)
    от отражения (ВЧ). Сцена состоит из:
      1. Базового градиента (фон 120→210)
      2. Кругов разной яркости (отражение)
      3. Квадратной рамки
      4. Мультипликативной тени — гауссов затухание яркости до 40%
         в левом-нижнем углу

    Результат: сценарий «тёмный угол, яркий центр» — идеален для IHF,
    так как модель I = E × R (яркость = освещённость × отражение).

    size: (height, width).
    """
    h, w = size
    # --- Фон: горизонтальный градиент ---
    bg = _gradient_background(size)
    bg = np.clip(np.linspace(120, 210, w, dtype=np.float64),
                 120, 210).astype(np.uint8)
    bg = np.tile(bg, (h, 1))

    img = bg.astype(np.float64)

    # --- Круги разной яркости (отражение) ---
    _fill_circle(img, int(w * 0.30), int(h * 0.30), int(w * 0.15), 180)
    _fill_circle(img, int(w * 0.65), int(h * 0.25), int(w * 0.12), 220)
    _fill_circle(img, int(w * 0.50), int(h * 0.65), int(w * 0.10), 100)
    _fill_circle(img, int(w * 0.80), int(h * 0.60), int(w * 0.08), 160)

    # --- Тонкая текстура: горизонтальные полосы ---
    stripe_spacing = max(3, int(w / 384))
    y = np.arange(h) // stripe_spacing
    img += (y % 3) * 8  # +0, +8, +16

    # --- Мультипликативная тень ---
    # Гауссов "затемняющий шар" в левом-нижнем углу
    cy_shadow, cx_shadow = int(h * 0.7), int(w * 0.3)
    sigma = max(w * 0.22, 30)
    y_grid, x_grid = np.ogrid[:h, :w]
    r2 = ((y_grid - cy_shadow) ** 2 + (x_grid - cx_shadow) ** 2)
    illumination = np.exp(-r2 / (2.0 * sigma ** 2))
    # illumination: 1 вдали → exp(0) = 1; в центре тени = exp(0) = 1
    # Инвертируем: тёмное в центре, светлое на краях
    illumination = 1.0 - 0.6 * illumination  # от 0.4 до 1.0

    img *= illumination
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# -------------------------------------------------------------------------
# Кот (первый кадр cat_meow.gif) на градиентном фоне
# -------------------------------------------------------------------------

def _composite_rgba_on_gradient(rgba_arr: np.ndarray) -> np.ndarray:
    """
    Кладёт RGBA-изображение на градиентный фон по альфа-каналу
    и возвращает grayscale (Rec. 601).
    """
    alpha = rgba_arr[:, :, 3].astype(np.float64) / 255.0
    r = rgba_arr[:, :, 0].astype(np.float64)
    g = rgba_arr[:, :, 1].astype(np.float64)
    b = rgba_arr[:, :, 2].astype(np.float64)
    gray_cat = REC601_R * r + REC601_G * g + REC601_B * b

    bg = _gradient_background(rgba_arr.shape[:2]).astype(np.float64)
    result = gray_cat * alpha + bg * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _resize_if_needed(arr: np.ndarray, target_size) -> np.ndarray:
    """Если PIL есть и размер отличается — ресайз."""
    if arr.shape[:2] == target_size:
        return arr
    try:
        pil = Image.fromarray(arr, mode="L" if arr.ndim == 2 else "RGBA")
        pil = pil.resize(target_size[::-1], Image.LANCZOS)
        return np.array(pil)
    except Exception:
        return arr


def generate_cat_meow(size=_DEFAULT_SIZE) -> np.ndarray:
    """
    Берёт первый кадр cat_meow.gif и кладёт кота
    (с альфа-каналом) на градиентный фон (16→235).
    Если GIF не найден — рисует кота вручную на том же фоне.

    size: (height, width) — финальный размер кадра.
    """
    gif_path = _PROJECT_ROOT / "cat_meow.gif"

    if gif_path.exists():
        try:
            img_pil = Image.open(str(gif_path))
            img_pil.seek(0)
            rgba = np.array(img_pil.convert("RGBA"))
            result = _composite_rgba_on_gradient(rgba)
            return _resize_if_needed(result, size)
        except Exception as e:
            print(f"  [!] Ошибка '{gif_path}': {e}")

    # Fallback: статичный кот, нарисованный на градиентном фоне
    return _draw_simple_cat(size)


def _draw_simple_cat(size: tuple) -> np.ndarray:
    """
    Статичный кот (fallback), нарисованный поверх градиентного фона.
    Координаты пропорциональны размеру изображения.
    """
    h, w = size
    bg_rgb = Image.new("RGB", (w, h), (_rgb_to_gray(200, 200, 200),) * 3)
    draw = ImageDraw.Draw(bg_rgb)

    scale = w / 1920.0

    cat_x = w // 2
    cat_y = h - int(300 * scale)

    body_w = int(120 * scale)
    body_h = int(160 * scale)
    head_r = int(55 * scale)

    # Хвост
    tail_x = cat_x + body_w // 2 + int(20 * scale)
    tail_base_y = cat_y + body_h // 2
    draw.line([(tail_x, tail_base_y),
               (tail_x + int(60 * scale), tail_base_y - int(40 * scale))],
              fill=(20, 20, 20), width=int(12 * scale))

    # Тело
    draw.ellipse([(cat_x - body_w // 2, cat_y - body_h // 2),
                  (cat_x + body_w // 2, cat_y + body_h // 2)],
                 fill=(30, 30, 30))

    # Грудка
    chest_w = int(60 * scale)
    chest_h = int(80 * scale)
    draw.ellipse([(cat_x - chest_w // 2, cat_y - 20 * scale),
                  (cat_x + chest_w // 2, cat_y + 60 * scale)],
                 fill=(40, 40, 40))

    # Голова
    draw.ellipse([(cat_x - head_r, cat_y - body_h // 2 - head_r + 10),
                  (cat_x + head_r, cat_y - body_h // 2 + head_r + 10)],
                 fill=(25, 25, 25))

    # Уши
    ear_h = int(35 * scale)
    ear_w = int(25 * scale)
    for side in (-1, 1):
        lx = cat_x + side * (head_r - ear_w)
        ly = cat_y - body_h // 2 - head_r + 10 - ear_h
        draw.polygon([(lx, ly + ear_h), (lx + ear_w, ly + ear_h),
                      (lx + ear_w // 2, ly)], fill=(30, 30, 30))

    # Глаза
    eye_y = cat_y - body_h // 2 + 10
    for side in (-1, 1):
        ex = cat_x + side * 20 * scale - 9 * scale
        draw.ellipse([(ex, eye_y), (ex + 18 * scale, eye_y + 20 * scale)],
                     fill=(180, 200, 50))

    # Нос
    draw.polygon([(cat_x - 5 * scale, eye_y + 20 * scale),
                  (cat_x + 5 * scale, eye_y + 20 * scale),
                  (cat_x, eye_y + 26 * scale)], fill=(200, 100, 100))

    arr = np.array(bg_rgb, dtype=np.float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = REC601_R * r + REC601_G * g + REC601_B * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# Главный генератор

def generate_all(output_dir: str = None) -> list:
    """
    Генерирует все синтетические изображения и сохраняет в output_dir.

    output_dir: относительно корня проекта (по умолчанию images/synthetic/).
    Возвращает список кортежей (имя_файла, путь_к_файлу).
    """
    if output_dir is None:
        output_dir = str(_PROJECT_ROOT / "images" / "synthetic")
    else:
        output_dir = str(_PROJECT_ROOT / output_dir.lstrip("./"))

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Снеговик
    img = generate_snowman()
    name = "snowman"
    path = os.path.join(output_dir, f"{name}.png")
    Image.fromarray(img, mode="L").save(path)
    results.append((name, path))
    print(f"  [OK] {name} ({img.shape[1]}x{img.shape[0]}) -> {path}")

    # Тень — сценарий мультипликативного затенения
    img = generate_shadowed()
    name = "shadowed"
    path = os.path.join(output_dir, f"{name}.png")
    Image.fromarray(img, mode="L").save(path)
    results.append((name, path))
    print(f"  [OK] {name} ({img.shape[1]}x{img.shape[0]}) -> {path}")

    # Кот
    img = generate_cat_meow()
    name = "cat_meow"
    path = os.path.join(output_dir, f"{name}.png")
    Image.fromarray(img, mode="L").save(path)
    results.append((name, path))
    print(f"  [OK] {name} ({img.shape[1]}x{img.shape[0]}) -> {path}")

    print(f"\nСгенерировано {len(results)} изображений в '{output_dir}'")
    return results


if __name__ == "__main__":
    if "--small" in sys.argv:
        print("Генерация тестовых 64x64...")
        for fn in (generate_snowman, generate_shadowed, generate_cat_meow):
            img = fn((64, 64))
            print(f"  {fn.__name__}: {img.shape}, min={img.min()}, "
                  f"max={img.max()}, std={img.std():.1f}")
    else:
        generate_all()
