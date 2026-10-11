# IHF-IMAGE-HOMOMORPHIC-FILTER

**Русский** / [English](README_EN.md)

## Гомоморфная фильтрация изображений в арифметике произвольной точности

**Версия:** 0.50

**Базовая модель:** `I = E·R` → `log₂` → F → разделение спектра на НЧ и ВЧ → обработка частей → соединение → F^-1 → `exp₂` → округление.

**Реализовано:** базовая модель, обработка ЧБ- и цветных фотографий.  
**В планах:** тесты, оптимизации и расширения.

## Требования

- Python 3.10+
- numpy, gmpy2, Pillow (`requirements.txt`)
- Windows, Linux или macOS

## Установка

### Создать виртуальное окружение
```bash
python -m venv .venv          # Windows (PowerShell)
python3 -m venv .venv         # Linux / macOS
```

Активировать окружение:

```powershell
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
```

```bash
source .venv/bin/activate           # Linux / macOS
```

Установить зависимости:
```bash
pip install -r requirements.txt
```
Запуск:

```bash
python main.py                      # тесты фильтра
python examples/bw_image.py         # ЧБ-фотография
python examples/color_image.py      # цветная фотография
```

## Использование

### Матрица синтетического изображения

```python
from image import create_image
from filter import homomorphic_filter

# E — освещённость (int 1..255), R — коэффициент отражения (float (R_MIN, 1]), I = E*R
e, r, i = create_image("const", "chess", 64, 64)

# Ослабление НЧ и усиление ВЧ
f = homomorphic_filter(lf_gain=0.5, hf_gain=2.0)
filtered = f.apply(i)ы
```

### Ч/б фотография

```python
from photo import photo_filter, read_gray, write_gray

photo = read_gray("examples/pic1.png")               # float64 (n, m), значения > 0
f = photo_filter(lf_gain=0.75, hf_gain=2.0, d0=30.0) # pad="reflect" по умолчанию
write_gray("out.png", f.to_uint8(f.apply(photo)))
```

### Цветная фотография

Каналы обрабатываются раздельно.

```python
from photo import photo_filter, read_rgb, write_rgb
from color_filter import color_homomorphic_filter

photo = read_rgb("examples/pic3_small.png")  # float64 (n, m, 3), RGB, альфа отбрасывается

# Усиления можно задавать по каналам: скаляр или (R, G, B)
f = color_homomorphic_filter(lf_gain=(0.99, 1.0, 1.01), hf_gain=1.0, pad="reflect")
write_rgb("out.png", photo_filter.to_uint8(f.apply(photo)))
```

`lf_gain` по каналам меняет **освещённость** (цвет света), `hf_gain` —
**коэффициент отражения** (детали), согласно модели `I = E·R`:
в логарифмах низкие частоты — это `E`, высокие — `R`.

«Холоднее» (colder): `(0.99, 1.0, 1.01)`.

### Перевод в N бит (`to_uint8`, `to_uint10`, `to_uint12`, `to_uint16`)

Результат `apply` не ограничен диапазоном 0..255, поэтому перед записью нужен
перевод:

| `Режим` | Что делает | Параметры |
|---|---|---|
| `"percentile"` (по умолчанию / default) | линейная растяжка отсечек до верхней границы (linear stretch) | `low=0.5`, `high=99.5` — перцентили |
| `"no"` | только обрезка к диапазону (clipping only; округление уже сделал фильтр) | — |
| `"gamma"` | гамма-коррекция после нормировки к [0, 1] (gamma after normalization) | `gamma=2.2`, `low`, `high` (`None` = min/max) |

```python
photo_filter.to_uint8(filtered)                 # растяжка по перцентилям
photo_filter.to_uint8(filtered, output="no")    # сохранить значения как есть
photo_filter.to_uint8(filtered, output="gamma", gamma=1.8, low=0.5, high=99.5)
```

Режимы вывода отличаются только верхней границей `0..2^bits-1`, формулы общие.
Неизвестный режим или параметр чужого режима — `ValueError` со списком
доступных.

У цветного кадра перцентили берутся по **всем каналам сразу**.

### Шумоподавление (`denoise`, по умолчанию выключено)

Используется билатерный фильтр:

```python
from photo import photo_filter, denoise, read_gray, write_gray

out = photo_filter.to_uint8(filtered)
write_gray("out.png", denoise(out, True))                  # включено
write_gray("out.png", denoise(out, True, radius=3, sigma_range=25.0))
```

`enabled` по умолчанию `DENOISE_DEFAULT = False` — кадр возвращается как есть.
Считается в float64 — mpfr на этом шаге избыточен.

### Отладка

`apply_info` печатает все стадии обработки построчно:

```python
homomorphic_filter("allpass", "allpass", 1.0, 0.0).apply_info(i)
```

## Фильтры, дополнение, окна
Фильтры выбираются в конструкторе: lf_filter — ФНЧ, hf_filter — ФВЧ, пара в сумме даёт 1; lf_gain/hf_gain — усиления НЧ/ВЧ частей.

Типы (filtration.py): lp_butterworth, hp_butterworth (Баттерворт), lp_chebyshev1, hp_chebyshev1 (Чебышёв 1-го рода), lp_gaussian, hp_gaussian (Гаусса), allpass (всепропускающий).  

Частота среза по умолчанию адаптивна: d0 = None → min(n, m)/4; значение задаётся в пикселях. 

Параметры d0, order, ripple_db (неравномерность АЧХ Чебышёва) — в конструкторе homomorphic_filter.
БПФ по Кули-Тьюки требует длины, равной степени двойки, поэтому apply(..., pad="zero"|"reflect") дополняет кадр и отсекает дополнение после ОБПФ. 

Окна hann/tukey гасят разрыв на границе перед БПФ и компенсируются после ОБПФ: f.apply(i, pad="reflect", window="tukey").

Яркость неположительных пикселов поднимается до MIN_VALUE = 1, так как log2(x <= 0) не определён; факт подъёма печатается в консоль.

## Типы матриц

- **E** (освещённость): `const`, `sawtooth`, `triangular`,
  `exponential`, `slow_changes`
- **R** (коэффициент отражения): `chess`, `const`, `random`, `black_square`,
  `white_square`
- **I** (интенсивность) получается из **E** и **R**: `I = E·R`

## Примеры

| Скрипт | Вход → выход | Что показывает |
|---|---|---|
| `examples/bw_image.py` | `pic1.png` → `pic1_processed.png` | ЧБ-фото, `lf_gain=0.75`, `hf_gain=2.0`, растяжка по умолчанию |
| `examples/color_image.py` | `pic3_small.png` → `pic3_small_cold.png` | цвет, освещение холоднее: `lf_gain=(0.99, 1.0, 1.01)` |

В обоих примерах есть переменная `DENOISE = False` — шумоподавление готового
кадра.

## Системные зависимости

Если сборка `gmpy2` из исходников падает из-за отсутствия заголовков `libgmp`,
`libmpfr` и `libmpc`, их устанавливают команды ниже:

**Debian / Ubuntu:**
```bash
sudo apt update
sudo apt install -y libgmp-dev libmpfr-dev libmpc-dev
```
**Fedora:**
```bash
sudo dnf install -y gmp-devel mpfr-devel libmpc-devel
```
**Arch Linux:**
```bash
sudo pacman -S --needed gmp mpfr libmpc
```
**macOS:**
```bash
brew install gmp mpfr libmpc
```

## Благодарности

Благодарю Константина Францевича Глассмана за консультации по гомоморфной
фильтрации и обработке изображений.
Благодарю glasgio за возможность использовать в качестве валидации результатов
https://github.com/glasgio/homomorphic-filter

## Лицензия

MIT

## Контакты

matmansky@yandex.ru (Владимир Яковлев)
