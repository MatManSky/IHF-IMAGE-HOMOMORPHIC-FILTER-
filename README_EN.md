# IHF-IMAGE-HOMOMORPHIC-FILTER  

**English** / [Русский](README.md) 

## Homomorphic image filtering using arbitrary-precision arithmetic  

**Version**: 0.50

## Model
`I = E·R` → `log₂` → F → spectrum split into LF and HF components → per-component processing → recombination → F⁻¹ → `exp₂` → rounding.

**Implemented**: the basic pipeline, grayscale and color photo processing.  
**Planned**: tests, optimizations and extensions.

## Requirements

- Python 3.10+
- `numpy`, `gmpy2`, `Pillow` (see `requirements.txt`)
- Windows, Linux or macOS

## Installation

Create a virtual environment:
```
python -m venv .venv          # Windows (PowerShell)
python3 -m venv .venv         # Linux / macOS
```

Activate the environment:
```
.venv\Scripts\Activate.ps1    # Windows (PowerShell)
source .venv/bin/activate     # Linux / macOS
```

Install dependencies:
```
pip install -r requirements.txt
```

## Running
```
python main.py                  # filter tests
python examples/bw_image.py     # grayscale photo
python examples/color_image.py  # color photo
```

## Usage

### Synthetic image matrix
```python
from image import create_image
from filter import homomorphic_filter

# E — illumination (int 1..255), R — reflectance (float (R_MIN, 1]), I = E·R
e, r, i = create_image("const", "chess", 64, 64)

# LF suppression, HF emphasis
f = homomorphic_filter(lf_gain=0.5, hf_gain=2.0)
filtered = f.apply(i)
```

### Grayscale photo
```python
from photo import photo_filter, read_gray, write_gray

photo = read_gray("examples/pic1.png")           # float64 (n, m), values > 0
f = photo_filter(lf_gain=0.75, hf_gain=2.0, d0=30.0)  # pad="reflect" by default
write_gray("out.png", f.to_uint8(f.apply(photo)))
```

### Color photo
Channels are processed independently.
```python
from photo import photo_filter, read_rgb, write_rgb
from color_filter import color_homomorphic_filter

photo = read_rgb("examples/pic3_small.png")  # float64 (n, m, 3), RGB; alpha is discarded

# Gains: scalar or one value per channel (R, G, B)
f = color_homomorphic_filter(lf_gain=(0.99, 1.0, 1.01), hf_gain=1.0, pad="reflect")
write_rgb("out.png", photo_filter.to_uint8(f.apply(photo)))
```

Per-channel `lf_gain` modulates illumination (the color of the light), while `hf_gain` modulates reflectance (fine detail), in accordance with the model `I = E·R`: in the logarithmic domain, low frequencies correspond to `E` and high frequencies to `R`.

"Cooler" tone: `(0.99, 1.0, 1.01)`.

### Conversion to N-bit integer (`to_uint8`, `to_uint10`, `to_uint12`, `to_uint16`)
The result of `apply` is not bounded to `0..255`, so conversion is required before saving:

| Output mode | What it does | Parameters |
| --- | --- | --- |
| `"percentile"` (default) | Linear stretch of percentiles to the upper bound | `low=0.5`, `high=99.5` — percentiles |
| `"no"` | Clipping only (filter already performs rounding) | — |
| `"gamma"` | Gamma correction after normalization to `[0, 1]` | `gamma=2.2`, `low`, `high` (`None` = min/max) |

```python
photo_filter.to_uint8(filtered)                                    # percentile stretch
photo_filter.to_uint8(filtered, output="no")                       # keep values as-is
photo_filter.to_uint8(filtered, output="gamma", gamma=1.8, low=0.5, high=99.5)
```

Output modes differ only in the upper bound `0..2^bits-1`; the formulas are shared. An unknown mode, or a parameter belonging to another mode, raises `ValueError` with a list of available options.

For a color frame, percentiles are computed jointly across all channels.

### Denoising (`denoise`, disabled by default)
A bilateral filter is used:
```python
from photo import photo_filter, denoise, read_gray, write_gray

out = photo_filter.to_uint8(filtered)
write_gray("out.png", denoise(out, True))                          # enabled
write_gray("out.png", denoise(out, True, radius=3, sigma_range=25.0))
```
`enabled` defaults to `DENOISE_DEFAULT = False` — the frame is returned unchanged.
Denoising is computed in `float64`; MPFR precision is redundant at this stage.

### Debugging
`apply_info` prints every pipeline stage line by line:
```python
homomorphic_filter("allpass", "allpass", 1.0, 0.0).apply_info(i)
```

## Filters, Padding, Windows
Filters are selected in the constructor: `lf_filter` — LPF, `hf_filter` — HPF; the pair sums to unity. `lf_gain` / `hf_gain` are the LF/HF gains.

Types (`filtration.py`): `lp_butterworth`, `hp_butterworth` (Butterworth), `lp_chebyshev1`, `hp_chebyshev1` (Chebyshev type I), `lp_gaussian`, `hp_gaussian` (Gaussian), `allpass` (all-pass).

The cutoff frequency is adaptive by default: `d0 = None` → `min(n, m)/4`; the value is specified in pixels. Parameters `d0`, `order`, `ripple_db` (Chebyshev passband ripple) are set in the `homomorphic_filter` constructor.

The Cooley–Tukey FFT requires a power-of-two length, so `apply(..., pad="zero"|"reflect")` pads the frame and trims the padding after the IFFT. `hann` / `tukey` windows suppress boundary discontinuities before the FFT and are compensated after the IFFT: `f.apply(i, pad="reflect", window="tukey")`.

Pixel values that are non-positive are clamped to `MIN_VALUE = 1`,
since `log₂(x ≤ 0)` is undefined; a warning is printed to the console.

## Matrix Types
- `E` (illumination): `const`, `sawtooth`, `triangular`, `exponential`, `slow_changes`
- `R` (reflectance): `chess`, `const`, `random`, `black_square`, `white_square`
- `I` (intensity) is obtained from `E` and `R` as `I = E·R`

## Examples

| Script | Input → Output | What it demonstrates |
| --- | --- | --- |
| `examples/bw_image.py` | `pic1.png` → `pic1_processed.png` | Grayscale photo, `lf_gain=0.75`, `hf_gain=2.0`, default stretch |
| `examples/color_image.py` | `pic3_small.png` → `pic3_small_cold.png` | Color, cooler illumination: `lf_gain=(0.99, 1.0, 1.01)` |

Both examples contain a `DENOISE = False` flag controlling post-filter denoising of the resulting frame.

## System Dependencies
If building `gmpy2` from source fails due to missing headers for `libgmp`, `libmpfr` and `libmpc`, install them with one of the commands below:

Debian / Ubuntu:
```
sudo apt update
sudo apt install -y libgmp-dev libmpfr-dev libmpc-dev
```

Fedora:
```
sudo dnf install -y gmp-devel mpfr-devel libmpc-devel
```

Arch Linux:
```
sudo pacman -S --needed gmp mpfr libmpc
```

macOS:
```
brew install gmp mpfr libmpc
```

## Acknowledgments
I thank Konstantin Frantsevich Glassman for consultations on homomorphic filtering and image processing.
I thank glasgio for providing https://github.com/glasgio/homomorphic-filter as a validation reference.

## License
MIT

## Contact
matmansky@yandex.ru (Vladimir Yakovlev)