# Установка и системные зависимости / Installation & System Dependencies

**Русский** / [English](#installation)

---

## Установка

### Требования

- Python 3.10+
- `numpy`, `gmpy2`, `Pillow` (см. `requirements.txt`)
- Windows, Linux или macOS

### Виртуальное окружение

Создать окружение:

```bash
python -m venv .venv          # Windows (PowerShell)
python3 -m venv .venv         # Linux / macOS
```

Активировать:

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

### Системные зависимости

Если сборка `gmpy2` из исходников падает из-за отсутствия заголовков
`libgmp`, `libmpfr` и `libmpc`, установите их одной из команд ниже.

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

### Запуск

```bash
python main.py                      # тесты фильтра / filter tests
python examples/bw_image.py         # Фотография в оттенках серого
python examples/color_image.py      # Цветная фотография
```

### Тесты

```bash
python -m pytest tests/ -q          # 96 тестов / 96 tests
python tests/test_all.py            # то же без pytest
```

---

## Installation

### Requirements

- Python 3.10+
- `numpy`, `gmpy2`, `Pillow` (see `requirements.txt`)
- Windows, Linux or macOS

### Virtual environment

Create:

```bash
python -m venv .venv          # Windows (PowerShell)
python3 -m venv .venv         # Linux / macOS
```

Activate:

```powershell
.venv\Scripts\Activate.ps1    # Windows (PowerShell)
```

```bash
source .venv/bin/activate     # Linux / macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### System dependencies

If building `gmpy2` from source fails due to missing headers for `libgmp`,
`libmpfr` and `libmpc`, install them with one of the commands below.

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

### Run

```bash
python main.py                  # filter tests
python examples/bw_image.py     # Grayscale photo
python examples/color_image.py  # Color photo
```

### Tests

```bash
python -m pytest tests/ -q          # 96 tests
python tests/test_all.py            # same without pytest
```
