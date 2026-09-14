# IHF-IMAGE-HOMOMORPHIC-FILTER-
Homomorphic image filtering using arbitrary-precision arithmetic (MPFR, 256-bit).

**Версия: 0.17** — реализована базовая модель:
`I = E·R` → `log2` → ДПФ → *(фильтр — заглушка)* → обратное ДПФ → `exp2` → округление.
В дальнейшем код будет расширяться.

## Требования

- Python **3.10+**
- Windows, Linux или macOS

## Установка и запуск

### Windows (PowerShell)

```powershell
# Создать и активировать виртуальное окружение
python -m venv .venv
.venv\Scripts\Activate.ps1

# Установить зависимости
pip install -r requirements.txt

# Запустить тесты
python main.py
```

### Linux (bash)

```bash
# Создать и активировать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Запустить тесты
python main.py
```

## Использование

```python
from image import create_image
from filter import homomorphic_filter

# E — освещение (int 1..255), R — отражение (float (1/255, 1]), I = E*R
e, r, i = create_image("const", "chess", 64, 64)

f = homomorphic_filter()
filtered = f.apply(i)   # результат: int64
```

Типы матриц: E — `const`, `sawtooth`, `triangular`, `exponential`;
R — `chess`, `random`.

## Лицензия

MIT

## Контакты

matmansky@yandex.ru (Vladimir Yakovlev)

