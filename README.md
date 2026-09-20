# IHF-IMAGE-HOMOMORPHIC-FILTER-
Homomorphic image filtering using arbitrary-precision arithmetic (MPFR, 256-bit).

**Версия: 0.26** — реализована базовая модель (a basic model has been implemented):
`I = E·R` → `log2` → F → разделение спектра на НЧ- и ВЧ-компоненты
(spectrum split into LF and HF parts) → обработка частей
(part processing) → соединение → F^-1 → `exp2` → округление (rounding).  
Цветной фильтр (color filter, `color_filter.py`): поканальная обработка R, G, B
(each channel processed independently).  
Код будет расширяться (The code will expand.).

## Требования (Requirements)

- Python **3.10+**
- Windows, Linux или macOS

## Установка и запуск (Installation and Launch)

### Windows

```powershell
# Создать и активировать виртуальное окружение (create and activate virtual environment)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Установить зависимости (install dependencies)
pip install -r requirements.txt

# Запустить тесты (run tests)
python main.py
```

### Linux

```bash
# Создать и активировать виртуальное окружение (create and activate virtual environment)
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости (install dependencies)
pip install -r requirements.txt

# Запустить тесты (run tests)
python main.py
```

### macOS

```bash
# Создать и активировать виртуальное окружение (create and activate virtual environment)
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости (install dependencies)
pip install -r requirements.txt

# Запустить тесты (run tests)
python main.py
```

## Использование (Usage)

```python
import numpy as np
from image import create_image
from filter import homomorphic_filter

# E — освещение (int 1..255), R — отражение (float (1/255, 1]), I = E*R
# (E — illumination (int 1..255), R — reflection (float (1/255, 1]), I = E*R)
e, r, i = create_image("const", "chess", 64, 64)

# Усиление ВЧ и ослабление НЧ (HF emphasis, LF suppression)
f = homomorphic_filter(lf_gain=0.5, hf_gain=2.0)
filtered = f.apply(i) # результат: int64 (result: int64)

# Цветное изображение (n, m, 3): фильтр применяется к каждому каналу
# (Color image (n, m, 3): the filter is applied to each channel)
from color_filter import color_homomorphic_filter

# Создаём три канала изображения (Create three image channels):
i_r = create_image("const", "chess", 64, 64)[2]      # канал R (R channel)
i_g = create_image("sawtooth", "chess", 64, 64)[2]   # канал G (G channel)
i_b = create_image("triangular", "chess", 64, 64)[2] # канал B (B channel)

color = np.stack([i_r, i_g, i_b], axis=2)  # shape (64, 64, 3)
filtered_color = color_homomorphic_filter(lf_gain=0.5, hf_gain=2.0).apply(color)
```

Фильтры выбираются в конструкторе (Filters are selected in the constructor):
`lf_filter` — ФНЧ (LPF), `hf_filter` — ФВЧ (HPF), пара в сумме даёт 1 (the pair sums to 1);
`lf_gain`/`hf_gain` — усиления НЧ/ВЧ частей (LF/HF part gains).
Типы фильтров (filter types, `filtration.py`): `lp_butterworth`, `hp_butterworth`
(Баттерворт / Butterworth), `lp_chebyshev1`, `hp_chebyshev1` (Чебышёв 1-го рода /
Chebyshev type I), `lp_gaussian`, `hp_gaussian` (Гаусса / Gaussian),
`allpass` (всепропускающий).
Частота среза по умолчанию адаптивна (adaptive cutoff): `D0 = min(n,m)/4`

Типы матриц (types of matrices): E — `const`, `sawtooth`, `triangular`, `exponential`,
`slow_changes`;
R — `chess`, `const`, `random`.

## Лицензия (License)

MIT

## Контакты (Contacts)

matmansky@yandex.ru (Vladimir Yakovlev)
