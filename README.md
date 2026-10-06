# IHF-IMAGE-HOMOMORPHIC-FILTER-

Гомоморфная фильтрация изображений в арифметике произвольной точности  
(Homomorphic image filtering using arbitrary-precision arithmetic).

**Версия: 0.45**

Базовая модель (basic model): `I = E·R` → `log2` → F → разделение спектра на НЧ и ВЧ
(spectrum split into LF and HF parts) → обработка частей (part processing) →
соединение → F^-1 → `exp2` → округление (rounding).

Реализовано (implemented): базовая модель, обработка ЧБ- и цветных фотографий
(grayscale and color photo processing). В планах (planned): тесты (tests),
оптимизациии и расширения (optimizations and extension).

## Требования (Requirements)

- Python **3.10+**
- numpy, gmpy2, Pillow (`requirements.txt`)
- Windows, Linux или macOS

## Установка (Installation)

Создать виртуальное окружение (create virtual environment)
```bash
python -m venv .venv # Windows (PowerShell)
```
```bash
python3 -m venv .venv # Linux / macOS
```

Активировать окружение (activate the environment):

```powershell
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
```

```bash
source .venv/bin/activate           # Linux / macOS
```

Установить зависимости (install dependencies):
```bash
pip install -r requirements.txt
```
Запуск (run):

```bash
python main.py                      # тесты фильтра (filter tests)
python examples/bw_image.py         # ЧБ-фотография (grayscale photo)
python examples/color_image.py      # цветная фотография (color photo)
```

## Использование (Usage)

### Матрица синтетического изображения (Synthetic image matrix)

```python
from image import create_image
from filter import homomorphic_filter

# E — освещение (int 1..255), R — отражение (float (R_MIN, 1]), I = E*R
# (E — illumination, R — reflection, I = E*R)
e, r, i = create_image("const", "chess", 64, 64)

# Ослабление НЧ и усиление ВЧ (LF suppression, HF emphasis)
f = homomorphic_filter(lf_gain=0.5, hf_gain=2.0)
filtered = f.apply(i)
```

### Ч/б фотография (Grayscale photo)

```python
from photo import photo_filter, read_gray, write_gray

photo = read_gray("examples/pic1.png")               # float64 (n, m), значения (values) > 0
f = photo_filter(lf_gain=0.75, hf_gain=2.0, d0=30.0) # pad="reflect" по умолчанию (default)
write_gray("out.png", f.to_uint8(f.apply(photo)))
```

### Цветная фотография (Сolor photo)

Каналы обрабатываются раздельно (channels are processed separately).

```python
from photo import photo_filter, read_rgb, write_rgb
from color_filter import color_homomorphic_filter

photo = read_rgb("examples/pic3_small.png")  # float64 (n, m, 3), RGB, альфа отбрасывается

# Усиления можно задавать по каналам: скаляр или (R, G, B)
# (gains: a scalar or one value per channel)
f = color_homomorphic_filter(lf_gain=(0.99, 1.0, 1.01), hf_gain=1.0, pad="reflect")
write_rgb("out.png", photo_filter.to_uint8(f.apply(photo)))
```

`lf_gain` по каналам меняет **освещённость** (цвет света), `hf_gain` —
**коэффициент отражения** (детали), согласно модели `I = E·R`:
в логарифмах низкие частоты — это `E`, высокие — `R`.

«Холоднее» (colder): `(0.99, 1.0, 1.01)`.

### Перевод в N бит (`to_uint8`, `to_uint12`, `to_uint16`)

Результат `apply` не ограничен диапазоном 0..255, поэтому перед записью нужен
перевод (the result is not limited to 0..255, so conversion is required):

| `output` | что делает (what it does) | параметры (params) |
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
доступных (unknown mode or parameter raises `ValueError`).

У цветного кадра перцентили берутся по **всем каналам сразу** (The color frame has percentiles across all channels at once.).

### Отладка (debug)

`apply_info` печатает все стадии обработки построчно (prints every pipeline stage):

```python
homomorphic_filter("allpass", "allpass", 1.0, 0.0).apply_info(i)
```

## Фильтры, дополнение, окна (Filters, Padding, Windows)

Фильтры выбираются в конструкторе (selected in the constructor): `lf_filter` —
ФНЧ (LPF), `hf_filter` — ФВЧ (HPF), пара в сумме даёт 1 (the pair sums to 1);
`lf_gain`/`hf_gain` — усиления НЧ/ВЧ частей (LF/HF gains).

Типы (types, `filtration.py`): `lp_butterworth`, `hp_butterworth` (Баттерворт /
Butterworth), `lp_chebyshev1`, `hp_chebyshev1` (Чебышёв 1-го рода / Chebyshev
type I), `lp_gaussian`, `hp_gaussian` (Гаусса / Gaussian), `allpass`
(всепропускающий / all-pass).

Частота среза по умолчанию адаптивна (adaptive cutoff): `d0 = None` →
`min(n, m)/4`; значение задаётся **в пикселях**. Параметры `d0`, `order`,
`ripple_db` (неравномерность АЧХ Чебышёва) — в конструкторе `homomorphic_filter`.

БПФ по Кули-Тьюки требует длины, равной степени двойки, поэтому
`apply(..., pad="zero"|"reflect")` дополняет кадр и отсекает дополнение после
ОБПФ. Окна `hann`/`tukey` гасят разрыв на границе перед БПФ и компенсируются
после ОБПФ: `f.apply(i, pad="reflect", window="tukey")`.

Яркость неположительных пикселов поднимается до `MIN_VALUE = 1`, так как
`log2(x <= 0)` не определён; факт подъёма печатается в консоль (non-positive
pixels are lifted to MIN_VALUE and reported).

## Типы матриц (Matrix Types)

- **E** (освещение / illumination): `const`, `sawtooth`, `triangular`,
  `exponential`, `slow_changes`
- **R** (отражение / reflection): `chess`, `const`, `random`, `black_square`,
  `white_square`
- **I** (интенсивность / intensity) получается из (is obtained from) **E** и (and) **R**: `I = E·R`

## Примеры (Examples)

| Скрипт (script) | Вход → выход (in → out) | Что показывает (what it shows) |
|---|---|---|
| `examples/bw_image.py` | `pic1.png` → `pic1_processed.png` | ЧБ-фото, `lf_gain=0.75`, `hf_gain=2.0`, растяжка по умолчанию |
| `examples/color_image.py` | `pic3_small.png` → `pic3_small_cold.png` | цвет, освещение холоднее: `lf_gain=(0.99, 1.0, 1.01)` |

## Системные зависимости (System Dependencies)

Если сборка `gmpy2` из исходников падает из-за отсутствия заголовков `libgmp`,
`libmpfr` и `libmpc`, их устанавливают команды ниже (if building gmpy2 from
sources fails, install the headers of GMP, MPFR and MPC):

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

## Благодарности (Acknowledgements)

Благодарю Константина Францевича Глассмана за консультации по гомоморфной
фильтрации и обработке изображений.
Благодарю glasgio за возможность использовать в качестве валидации результатов
https://github.com/glasgio/homomorphic-filter
(I thank Konstantin Frantsevich Glassman for his consultations on homomorphic
filtering and image processing. I thank glasgio for the opportunity to use
https://github.com/glasgio/homomorphic-filter as a validation of the results)

## Лицензия (License)

MIT

## Контакты (Contacts)

matmansky@yandex.ru (Vladimir Yakovlev)
