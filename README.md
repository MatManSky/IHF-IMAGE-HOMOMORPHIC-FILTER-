# IHF-IMAGE-HOMOMORPHIC-FILTER-
Homomorphic image filtering using arbitrary-precision arithmetic (MPFR, 256-bit).

**Версия: 0.19** — реализована базовая модель (a basic model has been implemented):
`I = E·R` → `log2` → F → фильтр (filter) → `log2` → `exp2` → округление (rounding) → F^-1 → `exp2` → округление (rounding).  
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
from image import create_image
from filter import homomorphic_filter

# E — освещение (int 1..255), R — отражение (float (1/255, 1]), I = E*R
# (E — illumination (int 1..255), R — reflection (float (1/255, 1]), I = E*R)
e, r, i = create_image("const", "chess", 64, 64)

f = homomorphic_filter()
filtered = f.apply(i)   # результат: int64 (result: int64)
```

Фильтр выбирается в конструкторе (The filter is selected in the constructor). По умолчанию (default) `allpass` — всепропускающий (allpass);

Типы матриц (types of matrices): E — `const`, `sawtooth`, `triangular`, `exponential`;
R — `chess`, `const`, `random`.

## Лицензия (License)

MIT

## Контакты (Contacts)

matmansky@yandex.ru (Vladimir Yakovlev)
