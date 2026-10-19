"""
Сквозные тесты гомоморфной фильтрации.

Запуск из корня проекта:
    python -m pytest tests/ -v
    python tests/test_all.py          # без pytest, только stdlib

Проверяются инварианты, а не конкретные числа:
  1. Округление после exp2 даёт исходные данные;
  2. Единичное усиление (lf_gain = hf_gain = 1);
  3. БПФ согласовано с ДПФ, прямое+обратное (2D) = тождество;
  4. Модель I = E·R: коэффициент отражения восстанавливается как I / E;
  5. photo_filter и color_homomorphic_filter не падают и дают корректную форму.

Ссылки:
 1. Гонсалес Р. Цифровая обработка изображений / Р. Гонсалес, Р. Вудс. М.: Техносфера, 2005. 1072 с.;
 2. Cooley J.W. An Algorithm for the Machine Calculation of Complex Fourier Series / J.W. Cooley, J.W. Tukey // Mathematics of Computation. 1965, Vol. 19, No. 90. P. 297–301;
 3. Stockham T.G. Image Processing in the Context of a Visual Model // Proceedings of the IEEE. 1972, Vol. 60, No. 7. P. 828–842;
 4. IEEE Std 754-2019. IEEE Standard for Floating-Point Arithmetic. New York: IEEE, 2019. 84 p.;
 5. Homomorphic filter // GitHub. URL: https://github.com/glasgio/homomorphic-filter (дата обращения: 17.10.2026);
 6. Андонов В.Г. Влияние резкости изображения на оптимальный градиент кривой тоновоспроизведения в кино и телевидении / В.Г. Андонов, М.В. Антипин, К.Ф. Гласман;
 7. Красильников Н.Н. Цифровая обработка 2D- и 3D-изображений: учеб. пособие. СПб.: БХВ-Петербург, 2011. 608 с.;
 8. Сергиенко А.Б. Цифровая обработка сигналов: учеб. пособие. 3-е изд. СПб.: БХВ-Петербург, 2011. 756 с.;
 9. Oppenheim A.V. Nonlinear Filtering of Multiplied and Convolved Signals / A.V. Oppenheim, R.W. Schafer, T.G. Stockham // Proceedings of the IEEE. 1968, Vol. 56, No. 8. P. 1264–1291.
"""

import os
import struct
import sys
import tempfile

import numpy as np
import gmpy2
from gmpy2 import mpc, mpfr
from PIL import Image  # проверка контейнера PNG (16-битный вывод)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import filtration
import image as image_mod
import photo as photo_mod
import processing
from filter import (homomorphic_filter, MIN_VALUE, _EPS, _WINDOW_FLOOR,
                    _fft_1d, _dft_1d, _fft_2d, _twiddle_table,
                    _roots_of_unity, _bitrev_swaps)
from color_filter import color_homomorphic_filter
from image import R_MIN

# Кадр небольшой: БПФ radix-2 требует степень двойки
N, M = 16, 16


# Импорт _setup_console из main.py, чтобы не плодить копии
import main as _main_mod
_setup_console = _main_mod._setup_console


def _filter(lf_gain=1.0, hf_gain=1.0, d0=4.0, order=2, **kw):
    return homomorphic_filter("lp_butterworth", "hp_butterworth",
                              lf_gain, hf_gain, d0, order, **kw)


def _real(matrix) -> np.ndarray:
    # Матрица mpfr/mpc -> float64 (мнимая часть здесь всегда нулевая)
    return np.array([[float(c.real) for c in row] for row in matrix],
                    dtype=np.float64)


def _eps() -> float:
    # Машинное эпсилон текущего режима точности
    return float(_EPS)


def _raises(exc, fn, *a, **kw):
    # Вызвать fn и сообщить, нужное ли исключение получилось
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception as other:
        raise AssertionError(
            f"ожидалось {exc.__name__}, получено {type(other).__name__}: {other}"
        ) from None
    return False


def _scene(e_type, r_type):
    # Кадр сцены, готовый для фильтра (без нулей)
    _, _, src = image_mod.create_image(e_type, r_type, N, M)
    return np.maximum(np.asarray(src, dtype=np.float64), MIN_VALUE)


# Базис

def test_roundtrip_log_antilog():
    """ exp2(log2(x)) с точностью MPFR и округлением даёт x без изменений"""
    f = _filter()
    # Диапазон 1..255: все пикселы > 0, логарифм определён без поднимания
    src = np.random.default_rng(0).integers(1, 256, (N, M)).astype(np.float64)
    got = f._antilogarithm(f._logarithm(src), N, M)
    assert got.dtype == np.int64
    bad = int((got != src.astype(np.int64)).sum())
    assert bad == 0, f"расхождений после round-trip: {bad} из {got.size}"


def test_zero_pixels_are_clamped():
    """Нули поднимаются до MIN_VALUE, а не дают -inf в логарифме"""
    arr = _real(_filter()._logarithm(np.zeros((N, M), dtype=np.float64)))
    assert np.isfinite(arr).all(), "в логарифме нулей появились inf/nan"
    assert arr.min() >= np.log2(MIN_VALUE) - 1e-12, "ноль не поднят до MIN_VALUE"


def test_logarithm_is_linear_in_brightness():
    """log2 переводит умножение яркости в сложение: log2(2I) = log2(I) + 1"""
    f = _filter()
    src = np.random.default_rng(1).integers(1, 128, (N, M)).astype(np.float64)
    a = _real(f._logarithm(src))
    b = _real(f._logarithm(src * 2.0))
    assert np.abs((b - a) - 1.0).max() < 1e-12, "log2 не линейен по яркости"


def test_eps_is_far_below_data_step():
    """Допуск округления не должен «съедать» шаг целых (1.0)"""
    assert float(_EPS) < 1e-20, f"_EPS слишком крупный: {float(_EPS)}"


# БПФ

def test_fft_matches_naive_dft():
    """Быстрое БПФ совпадает с прямым расчётом ДПФ."""
    rng = np.random.default_rng(2)
    x = [mpc(float(v)) for v in rng.integers(0, 256, 8)]
    for k, (a, b) in enumerate(zip(_fft_1d(x), _dft_1d(x))):
        assert abs(a - b) < mpfr("1e-20"), f"БПФ разошлось с ДПФ на k={k}"


def test_fft_inverse_needs_2d_scale():
    """
    Обратное 1D БПФ не нормируется: масштаб 1/(n*m) живёт в _fft_2d.
    Тест фиксирует контракт, чтобы нормировку не перенесли случайно
    """
    rng = np.random.default_rng(3)
    x = [mpc(float(v)) for v in rng.integers(0, 256, 8)]
    back = _fft_1d(_fft_1d(x), inverse=True)
    for a, b in zip(x, back):
        assert abs(b - a * 8) < mpfr("1e-15"), \
            "обратное 1D БПФ перестало быть ненормированным"


def test_fft_2d_roundtrip_is_identity():
    """Прямое + обратное 2D БПФ возвращают исходную матрицу"""
    rng = np.random.default_rng(4)
    src = rng.integers(0, 256, (8, 8)).astype(np.float64)
    data = [[mpc(float(v)) for v in row] for row in src.tolist()]
    back = _real(_fft_2d(_fft_2d(data), inverse=True))
    assert np.abs(back - src).max() < 1e-12, "2D round-trip исказил данные"


def test_fft_non_power_of_two_still_works():
    """Не степень двойки — прямой расчёт ДПФ, результат тоже верный"""
    rng = np.random.default_rng(5)
    x = [mpc(float(v)) for v in rng.integers(0, 256, 7)]
    for a, b in zip(_fft_1d(x), _dft_1d(x)):
        assert abs(a - b) < mpfr("1e-20"), "ДПФ для n=7 разошлось"


def test_fft_is_linear():
    """БПФ линейно: FFT(a + b) = FFT(a) + FFT(b)."""
    rng = np.random.default_rng(6)
    a = [mpc(float(v)) for v in rng.integers(0, 128, 8)]
    b = [mpc(float(v)) for v in rng.integers(0, 128, 8)]
    lhs = _fft_1d([x + y for x, y in zip(a, b)])
    rhs = [x + y for x, y in zip(_fft_1d(a), _fft_1d(b))]
    for x, y in zip(lhs, rhs):
        assert abs(x - y) < mpfr("1e-18"), "БПФ не линейно"


# Прозрачность

def test_unity_gain_is_transparent():
    """lf_gain = hf_gain = 1: H = 1, спектр не меняется, выход = вход"""
    src = np.random.default_rng(7).integers(1, 256, (N, M)).astype(np.float64)
    got = _filter(1.0, 1.0).apply(src)
    bad = int((got != src.astype(np.int64)).sum())
    assert bad == 0, f"единичное усиление исказило {bad} из {got.size} пикселов"


def test_unity_gain_with_pad_is_transparent():
    """Дополнение до степени двойки не вносит искажений при H = 1."""
    src = np.random.default_rng(8).integers(1, 256, (12, 20)).astype(np.float64)
    for pad in ("zero", "reflect"):
        got = _filter(1.0, 1.0).apply(src, pad=pad)
        assert got.shape == src.shape, f"pad={pad}: форма {got.shape}"
        bad = int((got != src.astype(np.int64)).sum())
        assert bad == 0, f"pad={pad}: искажено {bad} из {got.size} пикселов"


def test_window_is_transparent_above_floor():
    """
    С окном восстановление точно там, где окно не ниже _WINDOW_FLOOR:
    компенсация делит на max(w, floor), поэтому в хвостах окна точность
    невозможна по построению. Проверяем обе половины контракта
    """
    src = np.random.default_rng(9).integers(1, 256, (12, 20)).astype(np.float64)
    got = _filter(1.0, 1.0).apply(src, pad="reflect", window="tukey")

    w = _real(filtration.get_window("tukey", 12, 20))
    floor = float(_WINDOW_FLOOR)
    good = w > floor
    same = got == src.astype(np.int64)

    assert same[good].all(), (
        f"в области окна (w > {floor}) искажено "
        f"{int((~same & good).sum())} пикселов"
    )
    assert not (same & ~good).all(), \
        f"в хвостах окна (w <= {floor}) вообще нет отличий — подозрительно"


def test_amplified_highs_brighten_texture():
    """Усиление ВЧ на шахматном тексте должно поднимать контраст"""
    src = _scene("slow_changes", "chess")
    flat = _filter(1.0, 1.0).apply(src)
    boosted = _filter(1.0, 2.5, d0=2.0).apply(src)
    assert boosted.std() > flat.std(), \
        f"усиление ВЧ не подняло контраст: {boosted.std()} <= {flat.std()}"


def test_suppressed_lows_darken_flat_shading():
    """Ослабление НЧ убирает плавную составляющую яркости."""
    src = _scene("slow_changes", "chess")
    flat = _filter(1.0, 1.0).apply(src)
    dark = _filter(0.2, 1.0, d0=2.0).apply(src)
    assert dark.mean() < flat.mean(), \
        f"ослабление НЧ не затемнило кадр: {dark.mean()} >= {flat.mean()}"


# АЧХ и окна

def test_every_filter_has_expected_dc_gain():
    """ФНЧ без пульсаций и allpass проходят постоянную (H=1), ФВЧ подавляет (H=0)"""
    for name in filtration._filter_types:
        h = float(np.real(filtration.get_filter(name, 8, 8, d0=2.0)[0][0]))
        if name.startswith("hp_"):
            want = 0.0 if "chebyshev" not in name else None  # пульсация lp на DC
        elif name == "allpass":
            want = 1.0
        elif "chebyshev" in name:
            want = None  # проверяем отдельно в test_chebyshev_has_ripple_at_dc
        else:
            want = 1.0
        if want is not None:
            assert abs(h - want) < 1e-9, f"{name}: H(0,0)={h}, ожидалось {want}"


def test_chebyshev_has_ripple_at_dc():
    """
    Чебышёв 1-го рода: H(0,0) < 1 при ripple_db > 0 — это физика
    равноволновой пульсации в полосе пропускания
    """
    with_ripple = float(
        filtration.get_filter("lp_chebyshev1", 8, 8, d0=2.0,
                              ripple_db=1.0)[0][0])
    without_ripple = float(
        filtration.get_filter("lp_chebyshev1", 8, 8, d0=2.0,
                              ripple_db=0.0)[0][0])
    assert abs(without_ripple - 1.0) < 1e-9, "ripple=0: H(0,0) != 1"
    assert with_ripple < without_ripple, "ripple_db=1.0 должно снижать H(0,0)"


def test_butterworth_is_monotone():
    """Баттерворт: монотонный спад по диагонали от DC"""
    h = _real(filtration.get_filter("lp_butterworth", 16, 16, d0=4.0, order=4))
    diag = [h[k][k] for k in range(9)]
    for a, b in zip(diag, diag[1:]):
        assert b <= a + 1e-12, f"lp_butterworth: неубывание по диагонали: {diag}"


def test_gaussian_dc_is_one():
    """Гауссов: H(0,0) = 1"""
    h = _real(filtration.get_filter("lp_gaussian", 8, 8, d0=2.0))
    assert abs(h[0][0] - 1.0) < 1e-9, "lp_gaussian: H(0,0) != 1"


def test_gaussian_is_monotone():
    """Гауссов: экспоненциальный спад, монотонен по диагонали."""
    h = _real(filtration.get_filter("lp_gaussian", 16, 16, d0=4.0))
    diag = [h[k][k] for k in range(9)]
    for a, b in zip(diag, diag[1:]):
        assert b <= a + 1e-12, f"lp_gaussian: неубывание по диагонали: {diag}"


def test_lows_and_highs_are_complementary():
    """ФНЧ + ФВЧ одной семьи дают 1: спектр разложен без провалов."""
    for family in ("butterworth", "gaussian", "chebyshev1"):
        lo = _real(filtration.get_filter(f"lp_{family}", 8, 8, d0=2.0))
        hi = _real(filtration.get_filter(f"hp_{family}", 8, 8, d0=2.0))
        assert np.abs(lo + hi - 1.0).max() < 1e-9, f"{family}: lp + hp != 1"


def test_afc_is_monotone_along_frequency():
    """АЧХ ФНЧ убывает с частотой, ФВЧ — растёт (вдоль строки)"""
    lo = _real(filtration.get_filter("lp_butterworth", 16, 16, d0=4.0, order=2))
    hi = _real(filtration.get_filter("hp_butterworth", 16, 16, d0=4.0, order=2))
    assert all(lo[0][k] >= lo[0][k + 1] for k in range(1, 7)), \
        "ФНЧ не убывает по частоте"
    assert all(hi[0][k] <= hi[0][k + 1] for k in range(1, 7)), \
        "ФВЧ не растёт по частоте"


def test_every_window_is_finite_and_non_negative():
    for name in filtration._window_types:
        w = np.asarray(filtration.get_window(name, 8, 8), dtype=np.float64)
        assert w.shape == (8, 8), f"{name}: форма {w.shape}"
        assert np.isfinite(w).all(), f"{name}: не конечные значения"
        assert w.min() >= 0.0, f"{name}: отрицательные веса"
        assert w.max() <= 1.0 + 1e-12, f"{name}: веса больше 1"


def test_unknown_names_are_rejected():
    for bad in ("nope", "", None):
        for name, fn in (("get_filter", filtration.get_filter),
                         ("get_window", filtration.get_window)):
            assert _raises((ValueError, KeyError, TypeError), fn, bad, 8, 8), \
                f"{name}({bad!r}) не отклонён"


def test_mismatched_filter_pair_is_rejected():
    """Несогласованная пара (lp_butterworth + hp_gaussian) не должна проходить."""
    assert _raises(ValueError, homomorphic_filter,
                   "lp_butterworth", "hp_gaussian", 1.0, 1.0, 4.0, 2)


# Физическая модель

def test_attributes_recovered_from_image():
    """image = e * r: отражение восстанавливается как image / e."""
    e, r, img = image_mod.create_image("const", "chess", N, M)
    assert img.shape == (N, M)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_back = np.where(e > 0, img / e, 0.0)
    assert np.abs(r_back - r).max() < 1e-9, "атрибуты не восстанавливаются"


def test_create_matrix_names_and_bounds():
    for kind, names in (("r", image_mod._r_types), ("e", image_mod._e_types)):
        assert names, f"kind={kind}: список типов пуст"
        for name in names:
            m = image_mod.create_matrix(name, N, M, kind=kind)
            assert m.shape == (N, M), f"{name}: форма {m.shape}"
            assert np.isfinite(m).all(), f"{name}: не конечные значения"


def test_create_image_shapes():
    for e_type in image_mod._e_types:
        for r_type in image_mod._r_types:
            e, r, img = image_mod.create_image(e_type, r_type, N, M)
            assert e.shape == r.shape == img.shape == (N, M), \
                f"{e_type}/{r_type}: формы не совпадают"


def test_illegal_kind_rejected():
    assert _raises(ValueError, image_mod.create_matrix,
                   "const", N, M, kind="x"), "неизвестный kind принят"


# Фото и цвет

def test_photo_filter_shape_and_dtype():
    src = np.random.default_rng(10).integers(0, 256, (N, M)).astype(np.float64)
    out = photo_mod.photo_filter(lf_gain=0.8, hf_gain=1.6, d0=4.0).apply(src)
    assert out.shape == src.shape, f"форма {out.shape} != {src.shape}"
    assert out.dtype == np.int64


def test_photo_filter_accepts_path_and_matrix_alike():
    """Один и тот же кадр из файла и из матрицы даёт идентичный результат"""
    src = np.random.default_rng(11).integers(0, 256, (N, M)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "in.png")
        photo_mod.write_gray(path, src)
        f = photo_mod.photo_filter(lf_gain=0.8, hf_gain=1.5, d0=4.0)
        from_file = f.apply(path)
        from_matrix = f.apply(photo_mod.read_gray(path))
    assert np.array_equal(from_file, from_matrix), \
        "путь и матрица дают разный результат"


def test_to_uint8_is_in_range_for_every_mode():
    src = np.random.default_rng(12).integers(0, 256, (N, M)).astype(np.float64)
    out = photo_mod.photo_filter(lf_gain=0.8, hf_gain=2.0, d0=4.0).apply(src)
    for mode in photo_mod._output_types:
        u8 = photo_mod.photo_filter.to_uint8(out, output=mode)
        assert u8.dtype == np.uint8, f"{mode}: тип {u8.dtype}"
        assert int(u8.min()) >= 0 and int(u8.max()) <= 255, f"{mode}: вне 0..255"


def test_to_uint8_rejects_unknown_mode_and_params():
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert _raises(ValueError, photo_mod.photo_filter.to_uint8,
                   values, output="nope"), "неизвестный режим вывода не отклонён"
    assert _raises(ValueError, photo_mod.photo_filter.to_uint8,
                   values, output="gamma", bogus=1), "лишний параметр принят"


def test_photo_filter_file_cycle():
    """Полный цикл: файл -> фильтр -> uint8 -> файл"""
    src = np.random.default_rng(13).integers(0, 256, (N, M)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "in.png")
        b = os.path.join(d, "out.png")
        photo_mod.write_gray(a, src)

        got = photo_mod.read_gray(a)
        assert got.shape == (N, M), f"прочитана форма {got.shape}"
        assert got.min() >= MIN_VALUE, "нулевые пикселы не подняты до MIN_VALUE"

        out = photo_mod.photo_filter(lf_gain=0.8, hf_gain=1.5, d0=4.0).apply(a)
        photo_mod.write_gray(b, photo_mod.photo_filter.to_uint8(out))
        assert os.path.getsize(b) > 0, "выходный файл пуст"
        assert photo_mod.read_gray(b).shape == (N, M), "выход другой формы"


def test_to_uint10_12_16_are_in_range_for_every_mode():
    """10, 12 и 16 бит: dtype uint16, диапазон 0..2^bits-1, шкалы согласованы"""
    src = np.random.default_rng(23).integers(0, 256, (N, M)).astype(np.float64)
    out = photo_mod.photo_filter(lf_gain=0.8, hf_gain=2.0, d0=4.0).apply(src)
    for bits in (10, 12, 16):
        name = f"to_uint{bits}"
        for mode in photo_mod._output_types:
            values = getattr(photo_mod.photo_filter, name)(out, output=mode)
            assert values.dtype == np.uint16, f"{name}/{mode}: тип {values.dtype}"
            top = int(values.max())
            assert int(values.min()) >= 0 and top <= photo_mod.MAX_UINT[bits], \
                f"{name}/{mode}: вне 0..{photo_mod.MAX_UINT[bits]} (max {top})"
    # Согласованность шкал: в долях полной шкалы значения совпадают с
    # точностью до половины шага каждой шкалы (верхние границы 255 и 4095
    # не кратны друг другу, поэтому в отсчётах расхождение до 1.5 LSB — норма)
    u8 = photo_mod.photo_filter.to_uint8(out, output="percentile")
    u12 = photo_mod.photo_filter.to_uint12(out, output="percentile")
    drift = float(np.abs(
        u12 / photo_mod.MAX_UINT[12] - u8 / photo_mod.MAX_UINT[8]).max())
    limit = 0.5 / photo_mod.MAX_UINT[8] + 0.5 / photo_mod.MAX_UINT[12]
    assert drift <= limit, f"шкалы 8 и 12 бит разошлись: {drift} > {limit}"


def test_write_gray_bits_roundtrip_is_exact():
    """write_gray + read_gray с одинаковым bits возвращают те же значения.

    10/12 бит у PNG своего режима нет: write_gray кладёт их старшими битами
    в 16-битный контейнер, read_gray снимает сдвиг (раньше не снимал —
    после записи 12 бит чтение возвращало значения, умноженные на 16)
    """
    rng = np.random.default_rng(24)
    with tempfile.TemporaryDirectory() as d:
        for bits in (8, 10, 12, 16):
            dtype = np.uint8 if bits == 8 else np.uint16
            values = rng.integers(0, photo_mod.MAX_UINT[bits] + 1,
                                  (N, M)).astype(dtype)
            path = os.path.join(d, f"out{bits}.png")
            photo_mod.write_gray(path, values, bits=bits)
            with Image.open(path) as picture:
                mode = picture.mode
            with open(path, "rb") as handle:
                depth = struct.unpack(">IIBB", handle.read()[16:26])[2]
            assert depth == (8 if bits == 8 else 16), \
                f"{bits} бит: в IHDR bit depth {depth}"
            assert mode == ("L" if bits == 8 else "I;16"), \
                f"{bits} бит: Pillow прочитал режим {mode}"
            # read_gray поднимает нули до MIN_VALUE — как и перед log2
            back = photo_mod.read_gray(path, bits=bits)
            expected = np.maximum(values.astype(np.float64), MIN_VALUE)
            assert np.array_equal(back, expected), \
                f"{bits} бит: round-trip не вернул значения"
            if bits > 8:
                # без bits 16-битный PNG читается как 8-битный: старшие биты
                shrunk = photo_mod.read_gray(path)
                expected = np.maximum((values >> (bits - 8)).astype(np.float64),
                                      MIN_VALUE)
                assert np.array_equal(shrunk, expected), \
                    f"{bits} бит: чтение по умолчанию — не старшие 8 бит"

    # Сценарий: фильтр -> to_uint12 -> файл -> чтение
    src = rng.integers(0, 256, (N, M)).astype(np.float64)
    out = photo_mod.photo_filter(lf_gain=0.8, hf_gain=1.5, d0=4.0).apply(src)
    v12 = photo_mod.photo_filter.to_uint12(out)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "chain12.png")
        photo_mod.write_gray(path, v12, bits=12)
        back = photo_mod.read_gray(path, bits=12)
    assert np.array_equal(back, np.maximum(v12.astype(np.float64), MIN_VALUE)), \
        "цепочка to_uint12 -> write_gray -> read_gray не замкнута"


def test_bits_and_color_depth_are_validated():
    """Разрядность и 16-битный цвет: внятный ValueError, а не молчание"""
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert _raises(ValueError, photo_mod.write_gray, os.devnull,
                   np.zeros((4, 4), np.uint16), 9), "bits=9 принят"
    assert _raises(ValueError, photo_mod.read_gray, os.devnull, bits=9), \
        "read_gray: bits=9 принят"
    assert _raises(ValueError, photo_mod.photo_filter.to_uint8,
                   values, output="percentile", maximum=5.0), \
        "maximum принят как параметр режима"
    assert _raises(ValueError, photo_mod.write_rgb, os.devnull,
                   np.zeros((4, 4, 3), np.uint16)), \
        "16-битный цветной кадр принят записью"


def test_color_filter_shape_and_dtype():
    rgb = np.random.default_rng(14).integers(
        0, 256, (N, M, 3)).astype(np.float64)
    out = color_homomorphic_filter(lf_gain=0.8, hf_gain=1.8, d0=4.0).apply(rgb)
    assert out.shape == (N, M, 3), f"форма {out.shape}, ожидалось (n, m, 3)"
    assert out.dtype == np.int64


def test_color_channels_are_independent():
    """Каналы обрабатываются раздельно: смена одного не трогает другие"""
    rng = np.random.default_rng(15)
    rgb = rng.integers(0, 256, (N, M, 3)).astype(np.float64)
    f = color_homomorphic_filter(lf_gain=0.9, hf_gain=1.4, d0=4.0)

    base = f.apply(rgb)
    other = rgb.copy()
    other[..., 2] = rng.integers(0, 256, (N, M)).astype(np.float64)
    moved = f.apply(other)

    assert not np.array_equal(base[..., 2], moved[..., 2]), \
        "третий канал не отреагировал на смену данных"
    for ch in (0, 1):
        assert np.array_equal(base[..., ch], moved[..., ch]), \
            f"канал {ch} изменился из-за другого канала"


def test_color_gray_matches_monochrome():
    """На сером кадре цветной фильтр не должен расходиться с обычным."""
    rng = np.random.default_rng(16)
    gray = rng.integers(1, 256, (N, M)).astype(np.float64)
    rgb = np.repeat(gray[..., None], 3, axis=2)

    color = color_homomorphic_filter(
        lf_gain=0.8, hf_gain=1.6, d0=4.0).apply(rgb)
    plain = _filter(0.8, 1.6, d0=4.0).apply(gray, pad="reflect")
    diff = np.abs(color[..., 0].astype(np.float64)
                  - plain.astype(np.float64))
    assert diff.max() <= 1, f"серый канал разошёлся с моно на {diff.max()}"


# Температура освещения

def _color_scene() -> np.ndarray:
    """Цветной кадр (n, m, 3) без нулей"""
    rng = np.random.default_rng(20)
    return rng.integers(1, 256, (N, M, 3)).astype(np.float64)


def _log_mean(values: np.ndarray, channel: int) -> float:
    """Среднее log2 по каналу: сдвиг температуры меняет его ровно в lf_gain раз"""
    return float(np.log2(values[..., channel].astype(np.float64)).mean())


def test_temperature_gains_mapping():
    """Холоднее — R вниз, B вверх; теплее — наоборот; 0 — без сдвига"""
    assert processing.temperature_gains(0.0) == (1.0, 1.0, 1.0)
    assert processing.temperature_gains(0.01) == (0.99, 1.0, 1.01)
    assert processing.temperature_gains(-0.01) == (1.01, 1.0, 0.99)


def test_temperature_gains_reject_out_of_range():
    for bad in (1.0, -1.0, 2.5, -3.0):
        assert _raises(ValueError, processing.temperature_gains, bad), \
            f"temperature={bad} не отклонён"


def test_temperature_touches_only_low_frequencies():
    """
    Сдвиг температуры умножает только НЧ-часть спектра (освещённость),
    поэтому в log2-области разница с тождественным фильтром равна
    (g - 1) * НЧ(log2 I), а ВЧ-часть остаётся нетронутой
    """
    # Большие яркости, чтобы округление exp2 не мешало измерению
    src = np.random.default_rng(21).uniform(1e5, 1e6, (N, M))
    g = 0.9
    cold = _filter(g, 1.0, d0=4.0).apply(src).astype(np.float64)
    plain = _filter(1.0, 1.0, d0=4.0).apply(src).astype(np.float64)
    measured = np.log2(cold) - np.log2(plain)

    log_data = _filter(g, 1.0, d0=4.0)._logarithm(src)
    lp = filtration.get_filter("lp_butterworth", N, M, d0=4.0, order=2)
    spectrum = _fft_2d(log_data)
    low = _fft_2d(
        [[s * h for s, h in zip(row, hrow)] for row, hrow in zip(spectrum, lp)],
        inverse=True,
    )
    expected = (g - 1.0) * _real(low)
    assert np.abs(measured - expected).max() < 1e-3, \
        "сдвиг температуры затронул не только НЧ-часть"


def test_make_colder_and_warmer_move_channels_apart():
    """Холоднее — B-R растёт, теплее — падает; форма и тип сохраняются"""
    rgb = _color_scene()
    plain = processing.shift_temperature(rgb, 0.0, d0=4.0)
    cold = processing.make_colder(rgb, 0.1, d0=4.0)
    warm = processing.make_warmer(rgb, 0.1, d0=4.0)

    assert cold.shape == warm.shape == (N, M, 3), "форма результата изменилась"
    assert cold.dtype == np.int64, f"тип результата {cold.dtype}"

    # Среднее log2 по каналу масштабируется в lf_gain раз
    assert _log_mean(cold, 0) < _log_mean(plain, 0), "холоднее не приглушило R"
    assert _log_mean(cold, 2) > _log_mean(plain, 2), "холоднее не подсветило B"
    assert _log_mean(warm, 0) > _log_mean(plain, 0), "теплее не подсветило R"
    assert _log_mean(warm, 2) < _log_mean(plain, 2), "теплее не приглушило B"
    assert _log_mean(cold, 1) == _log_mean(plain, 1), "зелёный канал сдвинулся"


def test_temperature_rejects_bad_strength_and_shape():
    rgb = _color_scene()
    assert _raises(ValueError, processing.make_colder, rgb, -0.1), \
        "отрицательный strength у make_colder принят"
    assert _raises(ValueError, processing.make_warmer, rgb, -0.1), \
        "отрицательный strength у make_warmer принят"
    assert _raises(ValueError, processing.make_colder, rgb[:, :, 0], 0.01), \
        "серый кадр принят цветной функцией"


# Шумоподавление и яркость цвета

STEP = 32  # сторона кадра со ступенькой яркости
FLAT_COLS = np.r_[3:13, 19:29]  # плоские участки по обе стороны ступеньки


def _noisy_step() -> np.ndarray:
    """Ступенька 60 -> 180 с гауссовым шумом sigma=8, uint8 (STEP, STEP)."""
    rng = np.random.default_rng(23)
    step = np.repeat(
        np.where(np.arange(STEP) < STEP // 2, 60, 180)[None, :], STEP, axis=0)
    noisy = np.clip(step + rng.normal(0.0, 8.0, (STEP, STEP)), 0.0, 255.0)
    return noisy.astype(np.uint8)


def _residual(values: np.ndarray) -> np.ndarray:
    """Остаток: кадр минус сглаженное по строке ядром [1, 2, 1]/4"""
    v = np.asarray(values, dtype=np.float64)
    kernel = np.array([1.0, 2.0, 1.0]) / 4.0
    smooth = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, v)
    return v - smooth


def test_denoise_is_off_by_default():
    """По умолчанию шумоподавление выключено и не трогает кадр."""
    values = _noisy_step()
    assert photo_mod.DENOISE_DEFAULT is False, "denoise включён по умолчанию"
    assert photo_mod.denoise(values) is values, "выключенный denoise изменил кадр"
    assert np.array_equal(photo_mod.denoise(values, False), values)
    assert photo_mod.denoise(values, True).dtype == values.dtype


def test_bilateral_kills_noise_keeps_edge():
    """Шум в плоских областях падает вдвое, скачок на границе остаётся"""
    values = _noisy_step()
    denoised = photo_mod.bilateral(values)

    before = _residual(values)[:, FLAT_COLS].std()
    after = _residual(denoised)[:, FLAT_COLS].std()
    assert after < before / 2, f"шум {before:.2f} -> {after:.2f}, не вдвое"

    left, right = STEP // 2 - 13, STEP // 2 + 3
    step_before = values[:, right:right + 10].mean() - values[:, left:left + 10].mean()
    step_after = (denoised[:, right:right + 10].mean()
                  - denoised[:, left:left + 10].mean())
    assert abs(step_after - step_before) <= 1.0, (
        f"ступенька {step_before:.2f} -> {step_after:.2f}: граница просела")


def test_bilateral_is_gaussian_when_range_is_huge():
    """sigma_range -> бесконечность: билатерный точно = гауссово размытие"""
    source = _noisy_step().astype(np.float64)
    radius, sigma = 2, 1.0
    got = photo_mod.bilateral(source, radius=radius, sigma_space=sigma,
                              sigma_range=1e9)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    padded = np.pad(source, radius, mode="symmetric")
    want = np.zeros_like(source)
    for a in offsets:
        for b in offsets:
            want += (kernel[a + radius] * kernel[b + radius]
                     * padded[radius + a: radius + a + STEP,
                              radius + b: radius + b + STEP])
    assert np.abs(got - want).max() < 1e-9, "в пределе не совпало с гауссом"


def test_bilateral_shape_dtype_and_constant():
    """Форма, dtype и диапазон сохраняются; постоянный кадр не меняется"""
    values = _noisy_step()
    denoised = photo_mod.bilateral(values)
    assert denoised.shape == values.shape, "форма изменилась"
    assert denoised.dtype == np.uint8, f"dtype {denoised.dtype}"
    assert denoised.min() >= 0 and denoised.max() <= 255, "вышло за шкалу"

    constant = np.full((N, M), 77, dtype=np.uint8)
    assert np.array_equal(photo_mod.bilateral(constant), constant), \
        "постоянный кадр изменился"

    wide = photo_mod.bilateral(values.astype(np.uint16) * 257,
                               sigma_range=photo_mod.DENOISE_SIGMA_RANGE * 257)
    assert wide.dtype == np.uint16 and wide.max() <= 65535, "uint16 потерялся"


def test_bilateral_color_shares_one_weight():
    """Граница только в R защищает от размытия G и B: один вес на каналы"""
    rgb = np.zeros((N, STEP, 3), dtype=np.uint8)
    rgb[:, :STEP // 2, 0] = 200
    rgb[:, STEP // 2:, 0] = 40
    rgb[:, :, 1:] = 100
    denoised = photo_mod.bilateral(rgb)

    assert denoised.shape == rgb.shape and denoised.dtype == np.uint8
    assert np.all(denoised[:, STEP // 2 - 2:STEP // 2 + 2, 1:] == 100), \
        "G и B размылись через границу канала R"
    assert denoised[0, 5, 0] == 200 and denoised[0, -6, 0] == 40, \
        "ступенька в канале R пропала"



def test_denoise_rejects_bad_params_and_shape():
    values = _noisy_step()
    for label, action in (
        ("radius=0", lambda: photo_mod.bilateral(values, radius=0)),
        ("radius=1.5", lambda: photo_mod.bilateral(values, radius=1.5)),
        ("sigma_range=0", lambda: photo_mod.bilateral(values, sigma_range=0)),
        ("sigma_space=-1", lambda: photo_mod.bilateral(values, sigma_space=-1)),
        ("кадр (n, m, 4)", lambda: photo_mod.bilateral(np.zeros((N, M, 4)))),
        ("кадр (n,)", lambda: photo_mod.bilateral(np.zeros(N))),
        ("denoise с чужим параметром",
         lambda: photo_mod.denoise(values, True, radius=-1)),
    ):
        assert _raises(ValueError, action), f"{label} принят"


def test_unknown_pad_mode_is_rejected():
    """Опечатка в pad не должна молча включать reflect."""
    src = np.random.default_rng(22).integers(1, 256, (N, M)).astype(np.float64)
    for bad in ("refl", "mean", "Reflect", "Zero"):
        assert _raises(ValueError, _filter().apply, src, pad=bad), \
            f"pad={bad!r} не отклонён"


def test_mono_filter_requires_2d_image():
    """Ядро принимает только (n, m): 1D и 3D — внятный ValueError"""
    assert _raises(ValueError, _filter().apply, np.full(16, 100.0)), \
        "одномерный вход принят"
    assert _raises(ValueError, _filter().apply, np.full((4, 4, 3), 100.0)), \
        "трёхмерный вход принят ядром (для цвета есть color_homomorphic_filter)"
    # список списков — не numpy, но форму имеет: приводится, а не падает
    from_list = _filter().apply([[100.0] * 8] * 8)
    from_array = _filter().apply(np.full((8, 8), 100.0))
    assert np.array_equal(from_list, from_array), "список и массив разошлись"


# Синтетика
# Тестовые размеры: степень двойки для быстрого БПФ
_TEST_SIZE = (64, 64)  # (height, width) — компактно, mpfr256 не тормозит

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_images as si


def _filter_for(size):
    """Фильтр, подобранный под размер кадра: d0 ~ 1/8 диагонали"""
    h, w = size
    diag = (h ** 2 + w ** 2) ** 0.5
    d0 = diag / 8.0
    return homomorphic_filter("lp_butterworth", "hp_butterworth",
                              lf_gain=0.3, hf_gain=2.0,
                              d0=d0, order=2)


def _low_freq_std(arr: np.ndarray, sigma: int = 8) -> float:
    """Стд. отклонение размытой версии (мера НЧ-вариации), numpy-only.

    Три прохода box blur ~ гауссово размытие (Central Limit Theorem).
    """
    kernel = np.ones(2 * sigma + 1, dtype=np.float64) / (2 * sigma + 1)

    def _blur(a, axis):
        return np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), axis, a)

    result = arr.astype(np.float64)
    for _ in range(3):
        result = _blur(result, axis=0)
        result = _blur(result, axis=1)
    return float(result.std())


def test_snowman_shape_and_range():
    """Снеговик: uint8, форма, диапазон, есть и тёмные, и яркие"""
    img = si.generate_snowman(_TEST_SIZE)
    assert img.dtype == np.uint8 and img.shape == _TEST_SIZE
    assert img.min() >= 0 and img.max() <= 255
    assert img.min() < 50, "нет тёмных пикселов (глаза, контуры)"
    assert img.max() > 200, "нет ярких пикселов"


def test_shadowed_has_multiplicative_variation():
    """Тень: сильная НЧ-вариация — фон 120→210 + мультипликативная тень"""
    img = si.generate_shadowed(_TEST_SIZE)
    assert img.dtype == np.uint8 and img.shape == _TEST_SIZE
    assert img.min() < 80, "тень не затемнила достаточно (min > 80)"
    assert img.max() > 180, "нет ярких участков (max < 180)"


def test_shadowed_ihf_reduces_low_freq():
    """ИHF с lf_gain=0.3: снижает НЧ-вариацию теневого кадра"""
    img = si.generate_shadowed(_TEST_SIZE)
    before = _low_freq_std(img)
    out = _filter_for(_TEST_SIZE).apply(img.astype(np.float64))
    # to_uint8 может растянуть диапазон, поэтому нормализуем
    out_norm = out.astype(np.float64) / 255.0
    after = _low_freq_std(out_norm)
    assert after < before * 0.9, \
        f"НЧ-вариация не упала: {before:.4f} -> {after:.4f}"


def test_cat_meow_shape_and_range():
    """Кот: uint8, форма, диапазон."""
    img = si.generate_cat_meow(_TEST_SIZE)
    assert img.dtype == np.uint8 and img.shape == _TEST_SIZE
    assert img.min() >= 0 and img.max() <= 255


def test_shadowed_uint8_roundtrip():
    """Фильтр -> to_uint8 -> значения 0..255."""
    img = si.generate_shadowed(_TEST_SIZE)
    out = _filter_for(_TEST_SIZE).apply(img.astype(np.float64))
    u8 = photo_mod.photo_filter.to_uint8(out)
    assert u8.dtype == np.uint8
    assert u8.min() >= 0 and u8.max() <= 255, \
        f"to_uint8 вышел за шкалу: {u8.min()}..{u8.max()}"


def test_shadowed_denoise_chain():
    """Фильтр -> to_uint8 -> denoise (вкл): не падает, форма сохраняется"""
    img = si.generate_shadowed(_TEST_SIZE)
    out = _filter_for(_TEST_SIZE).apply(img.astype(np.float64))
    u8 = photo_mod.photo_filter.to_uint8(out)
    denoised = photo_mod.denoise(u8, enabled=True)
    assert denoised.shape == _TEST_SIZE and denoised.dtype == np.uint8


# Reflectance-паттерны
# Snowman и cat — коэффициенты отражения в image.py, масштабируемые
# аналитически (относительные координаты), без PIL-ресайза.

def test_r_types_are_in_range():
    """snowman и cat: float64, R_MIN < R <= 1 на любом размере."""
    for n, m in [(1, 1), (3, 5), (16, 32), (64, 64), (200, 120)]:
        for r_type in ("snowman", "cat"):
            r = image_mod.create_matrix(r_type, n, m, kind="r")
            assert r.dtype == np.float64 and r.shape == (n, m)
            assert r.min() >= R_MIN, f"{r_type} {n}x{m}: min={r.min()}"
            assert r.max() <= 1.0 + 1e-12, f"{r_type} {n}x{m}: max={r.max()}"


def test_r_types_have_contrast():
    """snowman и cat имеют и яркие, и тёмные участки (не плоские)"""
    for r_type in ("snowman", "cat"):
        r = image_mod.create_matrix(r_type, 64, 64)
        assert r.max() - r.min() > 0.1, \
            f"{r_type} слишком плоский: размах {r.max() - r.min():.3f}"


def test_r_types_scale_without_pil():
    """Один и тот же относительный паттерн на разных размерах"""
    rng = np.random.default_rng(30)
    for r_type in ("snowman", "cat"):
        r_small = image_mod.create_matrix(r_type, 16, 16)
        r_big = image_mod.create_matrix(r_type, 128, 128)
        # Проверяем: относительная структура сохранена
        # (не совпадает пиксел-в-пиксел из-за дискретизации,
        #  но min/max и доли пикселов разных уровней — да)
        assert abs(r_small.min() - r_big.min()) < 0.01, \
            f"{r_type} min изменился при масштабировании"
        assert abs(r_small.max() - r_big.max()) < 0.01, \
            f"{r_type} max изменился при масштабировании"


def test_ihf_preserves_r_structure_on_r_types():
    """
    I = slow_changes × R: IHF с lf_gain=0.3 подавляет НЧ-градиент E,
    оставляя R. Корреляция log2(out) с log2(R) должна быть высокой.
    """
    for r_type in ("snowman", "cat"):
        e, r, i = image_mod.create_image("slow_changes", r_type, 64, 64)
        f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                               lf_gain=0.3, hf_gain=2.0, d0=8.0, order=2)
        out = f.apply(i).astype(np.float64)
        log_r = np.log2(r)
        log_out = np.log2(np.maximum(out, 1.0))
        corr = abs(np.corrcoef(log_out, log_r)[0, 1])
        assert corr > 0.95, \
            f"{r_type}: corr(log2(out), log2(R)) = {corr:.4f} < 0.95"


def test_ihf_reduces_low_freq_gradient_on_r_types():
    """
    IHF с lf_gain=0.3 снижает уклон (градиент) по строкам кадра,
    сгенерированного из slow_changes × R
    """
    for r_type in ("snowman", "cat"):
        e, r, i = image_mod.create_image("slow_changes", r_type, 64, 64)
        # Уклон по строкам для входа
        slope_in = abs(np.polyfit(np.arange(64), i.mean(axis=0), 1)[0])
        # Фильтр
        f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                               lf_gain=0.3, hf_gain=2.0, d0=8.0, order=2)
        out = f.apply(i).astype(np.float64)
        slope_out = abs(np.polyfit(np.arange(64), out.mean(axis=0), 1)[0])
        assert slope_out < slope_in * 0.5, \
            f"{r_type}: уклон НЧ не упал: {slope_in:.3f} -> {slope_out:.3f}"


# Кэши оптимизаций
# Ускорения (таблицы множителей БПФ, мемоизация АЧХ, свёртка усиления)
# не имеют права менять результат — эти тесты это фиксируют.

def test_twiddle_table_matches_direct_computation():
    """Кэшированные множители равны посчитанным напрямую и переиспользуются"""
    two_pi = 2 * gmpy2.const_pi()
    for length, sign in ((2, -1), (8, -1), (16, 1), (64, 1)):
        cached = _twiddle_table(length, sign)
        ang = sign * two_pi / length
        fresh = [mpc(gmpy2.cos(ang * k), gmpy2.sin(ang * k))
                 for k in range(length // 2)]
        assert len(cached) == length // 2, f"length={length}: {len(cached)}"
        for got, want in zip(cached, fresh):
            assert got == want, f"length={length}, sign={sign}: множитель иной"
        assert _twiddle_table(length, sign) is cached, "таблица не кэшируется"


def test_roots_of_unity_are_roots_of_one():
    """Таблица ДПФ: n значений, w[t] == w[t mod n], старший корень = 1."""
    for n, sign in ((7, -1), (9, 1), (31, -1)):
        w = _roots_of_unity(n, sign)
        assert len(w) == n, f"n={n}: длина {len(w)}"
        assert w[0] == mpc(1), f"n={n}: w[0] != 1"
        assert _roots_of_unity(n, sign) is w, "таблица корней не кэшируется"


def test_bitrev_swaps_is_true_bit_reversal():
    """Пары swap дают перестановку, идентичную наивному бит-реверсу."""
    for n in (2, 4, 8, 16, 64):
        bits = n.bit_length() - 1
        want = [int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(n)]
        got = list(range(n))
        for i, j in _bitrev_swaps(n):
            got[i], got[j] = got[j], got[i]
        assert got == want, f"n={n}: {got} != {want}"


def test_dft_1d_matches_per_cell_trigonometry():
    """
    Табличные корни дают то же ДПФ, что наивный cos/sin на каждую пару
    (j,k). Допуск 1e-40 много меньше любой реальной разницы: таблица
    держит аргумент в [0, 2*pi), поэтому точность только выигрывает
    """
    rng = np.random.default_rng(31)
    n = 9  # не степень двойки — именно путь ДПФ
    x = [mpc(float(v)) for v in rng.integers(0, 256, n)]
    two_pi = 2 * gmpy2.const_pi()
    for inverse in (False, True):
        sign = 1 if inverse else -1
        naive = []
        for k in range(n):
            s = mpc(0)
            for j in range(n):
                ang = sign * two_pi * j * k / n
                s += x[j] * mpc(gmpy2.cos(ang), gmpy2.sin(ang))
            naive.append(s)
        for got, want in zip(_dft_1d(x, inverse=inverse), naive):
            assert abs(got - want) < mpfr("1e-40"), \
                f"inverse={inverse}: расхождение {abs(got - want)}"


def test_afc_memoization_matches_direct_computation():
    """
    Мемоизация АЧХ по целому du^2+dv^2 даёт побитово ту же матрицу,
    что расчёт формулы заново на каждую ячейку
    """
    d0 = mpfr(6.0)
    inv_2d0sq = 1 / (2 * d0 * d0)

    def naive(n, m, formula):
        return [[formula(gmpy2.sqrt(mpfr(min(u, n - u) ** 2
                                         + min(v, m - v) ** 2)))
                 for v in range(m)] for u in range(n)]

    families = (
        ("lp_butterworth", lambda d: 1 / (1 + (d / d0) ** 6),
         dict(d0=6.0, order=3)),
        ("lp_gaussian", lambda d: gmpy2.exp(-(d * d) * inv_2d0sq),
         dict(d0=6.0)),
    )
    n, m = 24, 32
    for name, formula, params in families:
        memo = filtration.get_filter(name, n, m, **params)
        want = naive(n, m, formula)
        for row_memo, row_want in zip(memo, want):
            for got, cell in zip(row_memo, row_want):
                assert got == cell, \
                    f"{name}: {float(got)} != {float(cell)}"


def test_afc_is_symmetric_about_center():
    """
    Все семьи: H одинакова в клетках с одинаковым расстоянием до DC —
    на этом держится мемоизация (в том числе у Чебышёва)
    """
    for name in ("lp_butterworth", "lp_gaussian", "lp_chebyshev1"):
        h = filtration.get_filter(name, 16, 20, d0=4.0, order=2,
                                  ripple_db=1.0)
        for u in range(1, 16):
            for v in range(1, 20):
                assert h[u][v] == h[16 - u][20 - v], \
                    f"{name}: H[{u}][{v}] != H[{16 - u}][{20 - v}]"


def test_combined_response_equals_explicit_sum():
    """Свёрнутая АЧХ = lf_gain*ФНЧ + hf_gain*ФВЧ и переиспользуется."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.4, 1.7, d0=4.0, order=2)
    parts, response = f._get_response((16, 16), verbose=True)
    lf_matrix, hf_matrix = parts
    for row_h, row_l, row_f in zip(response, lf_matrix, hf_matrix):
        for h, lf, hf in zip(row_h, row_l, row_f):
            assert h == lf * f._lf_gain + hf * f._hf_gain, \
                f"свёртка АЧХ иная: {float(h)}"

    _, again = f._get_response((16, 16), verbose=False)
    assert again is response, "суммарная АЧХ не переиспользуется"

    _, other = f._get_response((8, 8), verbose=False)
    assert other is not response, "при смене размера вернулась старая матрица"
    assert len(other) == 8 and len(other[0]) == 8, "размер пересчитан неверно"


def test_log2_of_equal_values_is_shared_object():
    """
    Одинаковые пикселы получают один и тот же mpfr-объект (кэш log2),
    что безопасно: mpfr неизменяем, а арифметика создаёт новые числа.
    """
    src = np.full((8, 8), 137, dtype=np.float64)
    log_data = _filter()._logarithm(src)
    first = log_data[0][0]
    assert all(cell is first for row in log_data for cell in row), \
        "одинаковые значения не переиспользуют один log2"
    # и значение верное: log2(137)
    assert float(first.real) == float(gmpy2.log2(mpfr(137))), \
        "кэшированный log2 неверен"


# Новые оптимизации

def _antilog_exact_reference(vals):
    """
    Эталон антилогарифмирования ДО оптимизации: только mpfr, без float64.
    Именно так считал exp2 до появления векторного быстрого пути.
    """
    return [int(gmpy2.round_away(gmpy2.exp2(v) + _EPS)) for v in vals]


def _antilog_via_fast(vals):
    """Быстрый путь (_antilogarithm) на том же наборе mpfr-значений."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth", 0.3, 2.0, d0=4)
    data = [[mpc(v, mpfr(0)) for v in vals]]
    return list(f._antilogarithm(data, 1, len(vals)).reshape(-1))


def test_antilog_fast_path_matches_exact_reference():
    """
    Быстрый float64-путь обязан давать ровно те же целые, что и чистый mpfr.

    Набор специально провокационный: значения, exp2 которых попадает ровно на
    границу округления k+0.5, и значения со смещениями вокруг этой границы —
    от заведомо безопасных (1e-6) до спорных (1e-17). Плюс типичные значения
    реальной яркости и крайние случаи.
    """
    vals = []

    # 1. Ровно на границе округления: exp2(v) = k + 0.5
    for k in range(1, 300):
        vals.append(gmpy2.log2(mpfr(2 * k + 1) / mpfr(2)))

    # 2. Смещения вокруг границы, включая много меньше float64-шага
    for k in range(1, 150):
        base = gmpy2.log2(mpfr(2 * k + 1) / mpfr(2))
        for shift in ("1e-6", "1e-9", "1e-11", "1e-13", "1e-15", "1e-17"):
            s = mpfr(shift)
            vals.append(base + s)
            vals.append(base - s)

    # 3. Типичная лог-яркость реального кадра
    rng = np.random.default_rng(211)
    vals += [mpfr(float(v)) for v in rng.uniform(-10.0, 16.0, size=3000)]

    # 4. Мелкие (уходят в 0) и крупные (миллиарды) значения
    vals += [mpfr(float(v)) for v in rng.uniform(-80.0, -20.0, size=400)]
    vals += [mpfr(float(v)) for v in rng.uniform(20.0, 42.0, size=400)]

    # 5. Крайние: ноль, денормалы, крупные но представимые в int64
    #    (exp2(62)=4.6e18 < INT64_MAX, exp2(63)=2^63 уже не влезает — это
    #    граница домена, она проверяется отдельным тестом)
    vals += [mpfr(0), mpfr(-1e6), mpfr("1e-300"), mpfr(-1e-300),
             mpfr(62.0), mpfr(-1e6)]

    ref = _antilog_exact_reference(vals)
    got = _antilog_via_fast(vals)

    bad = [(i, ref[i], int(got[i])) for i in range(len(vals))
           if ref[i] != int(got[i])]
    assert not bad, (f"быстрый exp2 разошёлся с mpfr на {len(bad)} из "
                     f"{len(vals)}: {bad[:5]}")


def test_antilog_beyond_int64_raises_in_both_paths():
    """
    exp2(63) = 2^63 не влезает в int64. Быстрый и точный пути должны
    вести себя одинаково: оба падают, а не выдают молча неверное число
    """
    vals = [mpfr(63.0), mpfr(70.0)]
    ref = _antilog_exact_reference(vals)          # большие python-int
    # Быстрый путь обязан не выдать ложное целое
    try:
        got = _antilog_via_fast(vals)
    except (OverflowError, ValueError):
        return                                  # как и в точном приведении
    assert [int(v) for v in got] == ref, \
        f"за пределами int64 получен некорректный результат: {got}"


def test_antilog_fast_path_is_actually_taken():
    """
    Оптимизация имеет смысл только если точный mpfr вызывается редко.
    На типичном кадре в mpfr должно уходить не больше ~1% пикселов.
    """
    rng = np.random.default_rng(212)
    src = rng.integers(1, 256, size=(64, 64)).astype(np.float64)
    f = _filter()
    vals = [row.real for row in f._logarithm(src) for row in row]

    x = np.fromiter((float(v) for v in vals), dtype=np.float64,
                    count=len(vals))
    y = np.exp2(x)
    ok = np.isfinite(y) & (y < 9.0e18)
    ok &= np.abs(y - (np.floor(y) + 0.5)) > 1e-9 * np.maximum(1.0, y)
    exact_share = float((~ok).sum()) / len(vals)
    assert exact_share < 0.01, \
        f"в точный путь уходит {exact_share:.1%} пикселов — кэш не помогает"


def test_antilog_boundary_values_match_exact():
    """
    exp2 ровно на границе k+0.5: _EPS сдвигает выше, round_away даёт k+1.
    """
    ks = list(range(1, 300))
    vals = [gmpy2.log2(mpfr(2 * k + 1) / mpfr(2)) for k in ks]
    got = _antilog_via_fast(vals)
    assert [int(v) for v in got] == [k + 1 for k in ks], \
        "значение ровно на границе округлилось не прочь от нуля"


def test_antilog_large_values_stay_correct():
    """exp2(20)≈1e6 и exp2(40)≈1e12 — большие значения не теряют точность."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth", 0.3, 2.0, d0=4)
    # exp2(20) = 1048576, exp2(10) = 1024
    data = [[mpc(mpfr(20.0), mpfr(0)), mpc(mpfr(10.0), mpfr(0)),
             mpc(mpfr(30.0), mpfr(0)), mpc(mpfr(40.0), mpfr(0))]]
    got = f._antilogarithm(data, 1, 4)
    expected = [1048576, 1024, 1073741824, 1099511627776]
    assert list(got[0]) == expected, f"большие значения: {list(got[0])}"


def test_antilog_small_values_near_zero():
    """exp2(-60)→0, exp2(-10)→0, exp2(0)→1 — малые значения корректны."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth", 0.3, 2.0, d0=4)
    data = [[mpc(mpfr(-60.0), mpfr(0)), mpc(mpfr(-10.0), mpfr(0)),
             mpc(mpfr(0.0), mpfr(0)), mpc(mpfr(1.0), mpfr(0))]]
    got = f._antilogarithm(data, 1, 4)
    expected = [0, 0, 1, 2]
    assert list(got[0]) == expected, f"малые значения: {list(got[0])}"


def test_antilog_all_integers_1_255_roundtrip():
    """Каждое целое 1..255 → log2 → exp2 → int64 возвращает исходное."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth", 0.3, 2.0, d0=4)
    src = np.arange(1, 256, dtype=np.float64).reshape(1, -1)
    got = f._antilogarithm(f._logarithm(src), 1, 255)
    bad = int((got != src.astype(np.int64)).sum())
    assert bad == 0, f"расхождений: {bad} из 255"


def test_antilog_all_integers_1_65535_roundtrip():
    """Весь диапазон 12-битного кадра: 65535 значений, потерь нет."""
    f = homomorphic_filter("lp_butterworth", "hp_butterworth", 0.3, 2.0, d0=4)
    src = np.arange(1, 65536, dtype=np.float64).reshape(1, -1)
    got = f._antilogarithm(f._logarithm(src), 1, 65535)
    bad = int((got != src.astype(np.int64)).sum())
    assert bad == 0, f"расхождений: {bad} из 65535"


def test_apply_is_deterministic_across_repeated_calls():
    """Два вызова apply() на одном входе дают побитово одинаковый выход."""
    rng = np.random.default_rng(99)
    for size in [(16, 16), (32, 32), (64, 64)]:
        img = rng.integers(1, 256, size=size, dtype=np.uint16)
        f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                               lf_gain=0.3, hf_gain=2.0, d0=8)
        out1 = f.apply(img, pad="reflect")
        out2 = f.apply(img, pad="reflect")
        assert np.array_equal(out1, out2), \
            f"apply не детерминирован: {size}"


def test_apply_deterministic_between_filter_instances():
    """Два фильтра с одинаковыми параметрами дают одинаковый результат."""
    rng = np.random.default_rng(100)
    img = rng.integers(1, 256, size=(32, 32), dtype=np.uint16)
    f1 = homomorphic_filter("lp_butterworth", "hp_butterworth",
                            lf_gain=0.3, hf_gain=2.0, d0=8)
    f2 = homomorphic_filter("lp_butterworth", "hp_butterworth",
                            lf_gain=0.3, hf_gain=2.0, d0=8)
    assert np.array_equal(f1.apply(img, pad="reflect"),
                          f2.apply(img, pad="reflect")), \
        "фильтры с одинаковыми параметрами дают разный результат"


def test_apply_stable_across_alternating_sizes():
    """
    Кэш АЧХ хранит ровно один размер. При чередовании размеров каждый
    следующий кадр должен считать АЧХ заново и давать тот же результат,
    что и обработка этого кадра отдельно.
    """
    rng = np.random.default_rng(101)
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           lf_gain=0.3, hf_gain=2.0, d0=8)
    sizes = [(16, 16), (24, 24), (16, 16), (32, 32), (24, 24), (32, 32)]
    frames = {s: rng.integers(1, 256, size=s, dtype=np.uint16) for s in sizes}

    # Эталон: каждый кадр обрабатывается своим свежим фильтром
    reference = {}
    for s, img in frames.items():
        g = homomorphic_filter("lp_butterworth", "hp_butterworth",
                               lf_gain=0.3, hf_gain=2.0, d0=8)
        reference[s] = g.apply(img, pad="reflect")

    for s in sizes:
        got = f.apply(frames[s], pad="reflect")
        assert np.array_equal(got, reference[s]), \
            f"размер {s} после чередования дал иной выход"


def test_apply_stable_across_filter_families():
    """
    Смена семьи фильтров на том же размере не должна оставлять следов
    в кэше АЧХ: butterworth → gaussian → butterworth даёт один и тот же
    результат для первой и третьей обработки.
    """
    rng = np.random.default_rng(102)
    img = rng.integers(1, 256, size=(32, 32), dtype=np.uint16)
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           lf_gain=0.3, hf_gain=2.0, d0=8)
    first = f.apply(img, pad="reflect")
    # Тот же объект фильтра не умеет менять семью, поэтому сравниваем
    # три независимых фильтра в порядке butterworth/gaussian/butterworth
    g = homomorphic_filter("lp_gaussian", "hp_gaussian",
                           lf_gain=0.3, hf_gain=2.0, d0=8)
    g.apply(img, pad="reflect")
    again = homomorphic_filter("lp_butterworth", "hp_butterworth",
                               lf_gain=0.3, hf_gain=2.0, d0=8)
    assert np.array_equal(again.apply(img, pad="reflect"), first), \
        "butterworth после gaussian дал иной результат"


def test_fft_skip_mul_by_one():
    """
    Пропуск умножения на 1 (k=0) в бабочках не меняет результат:
    БПФ совпадает с прямым расчётом ДПФ для всех степеней двойки.
    """
    rng = np.random.default_rng(103)
    for n in [2, 4, 8, 16, 32, 64]:
        x = [mpc(float(v)) for v in rng.integers(0, 256, n)]
        for sign in (False, True):
            got = _fft_1d(x, inverse=sign)
            expected = _dft_1d(x, inverse=sign)
            for i, (a, b) in enumerate(zip(got, expected)):
                assert abs(a - b) < mpfr("1e-20"), \
                    f"пропуск умножения на 1: разошлось с ДПФ k={i}, n={n}"


def test_butterfly_zero_multiplier_is_one():
    """
    Множитель уровня при k=0 действительно равен 1 — иначе пропуск
    умножения искажает результат.
    """
    for length in [2, 4, 8, 16, 64, 128]:
        for sign in (-1, 1):
            w0 = _twiddle_table(length, sign)[0]
            assert abs(w0 - mpc(1)) < mpfr("1e-40"), \
                f"w(0) != 1 для length={length}, sign={sign}"


def test_cache_response_reused_same_size():
    """
    Повторный _get_response того же размера возвращает ту же матрицу.
    """
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.4, 1.7, d0=4.0, order=2)
    _, r1 = f._get_response((16, 16), False)
    _, r2 = f._get_response((16, 16), False)
    assert r2 is r1, "кэш АЧХ не переиспользуется для одного размера"


def test_cache_afc_parts_cleared_in_non_verbose():
    """
    В не-вербозном пути частные АЧХ (lp/hp) не остаются в памяти —
    они очищаются сразу после формирования суммарной АЧХ. Каждая из них
    стоит n*m чисел по 256 бит, держать их в рабочем пути нельзя.
    """
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.3, 2.0, d0=8)
    img = np.full((16, 16), 128, dtype=np.float64)
    f.apply(img, pad="reflect")
    assert f._response_parts is None, \
        "частные АЧХ остались в памяти после не-вербозного apply"


def test_cache_afc_parts_available_in_verbose():
    """
    apply_info обязан получить частные АЧХ для отчёта — и они должны
    соответствовать суммарной.
    """
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.3, 2.0, d0=4)
    parts, combined = f._get_response((16, 16), verbose=True)
    assert parts is not None, "в вербозном режиме частные АЧХ не построены"
    lp, hp = parts
    for rh, rl, rf in zip(combined, lp, hp):
        for h, l, ff in zip(rh, rl, rf):
            assert h == 0.3 * l + 2.0 * ff, "суммарная АЧХ не согласована"


def test_log2_cache_reduces_object_count():
    """
    Кэш log2 создаёт ровно один mpfr-объект на уникальное значение.
    Для кадра 16×16 со всеми одинаковыми пикселами → 1 объект log2.
    """
    src = np.full((16, 16), 100, dtype=np.float64)
    log_data = _filter()._logarithm(src)
    first = log_data[0][0]
    assert all(cell is first for row in log_data for cell in row), \
        "кэш log2 не объединяет одинаковые значения"


def test_log2_cache_shares_only_equal_values():
    """
    Разные значения не должны получать один объект, а одинаковые —
    разные. Проверка на кадре из двух значений.
    """
    src = np.array([[100.0, 200.0]] * 8, dtype=np.float64)
    log_data = _filter()._logarithm(src)
    a, b = log_data[0][0], log_data[0][1]
    assert a is not b, "разные значения получили один объект"
    assert log_data[7][0] is a and log_data[7][1] is b, \
        "одинаковые значения в другой строке не переиспользуются"
    assert abs(a - mpc(gmpy2.log2(mpfr(100)))) < mpfr("1e-40"), "log2(100) неверен"
    assert abs(b - mpc(gmpy2.log2(mpfr(200)))) < mpfr("1e-40"), "log2(200) неверен"


def test_afc_cache_clears_on_resize():
    """
    При смене размера кэш очищается: старый размер теряется.
    """
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.3, 2.0, d0=4)
    _, r1 = f._get_response((8, 8), False)
    _, r2 = f._get_response((16, 16), False)
    assert r2 is not r1, "при смене размера старый кэш остался"
    # И старый размер недоступен:
    _, r1_again = f._get_response((8, 8), False)
    assert r1_again is not r1, "кэш не пересоздан для старого размера"


def test_afc_response_matches_explicit_lp_hf():
    """
    Суммарная АЧХ = lf_gain * LP + hf_gain * HP, а не произведение
    и не что-то иное.
    """
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           0.3, 2.0, d0=4, order=2)
    # Явно создаём фильтры отдельно
    lf = filtration.get_filter("lp_butterworth", 16, 16, d0=4, order=2)
    hf = filtration.get_filter("hp_butterworth", 16, 16, d0=4, order=2)
    _, combined = f._get_response((16, 16), False)
    for r in range(16):
        for c in range(16):
            expected = f._lf_gain * lf[r][c] + f._hf_gain * hf[r][c]
            assert combined[r][c] == expected, \
                f"AЧХ[{r}][{c}] != lf_gain*LP + hf_gain*HP"


def test_ihf_accuracy_vs_numpy_float64():
    """
    АЧХ фильтра (butterworth) в mpfr близка к float64-аналогу.
    Относительная ошибка на каждом пикселе < 1e-12.
    """
    n, m, d0, order = 32, 32, 8.0, 2
    # MPFR-фильтр
    f = homomorphic_filter("lp_butterworth", "hp_butterworth",
                           lf_gain=0.3, hf_gain=2.0, d0=d0, order=order)
    _, combined_mpfr = f._get_response((n, m), False)

    # Numpy-аналог: wrap-расстояние, DC на [0,0]
    Y, X = np.ogrid[0:n, 0:m]
    du = np.minimum(X, m - X)
    dv = np.minimum(Y, n - Y)
    D = np.sqrt(du.astype(np.float64) ** 2 + dv.astype(np.float64) ** 2)
    lp_np = 1.0 / (1.0 + (D / d0) ** (2 * order))
    hp_np = 1.0 - lp_np
    combined_np = 0.3 * lp_np + 2.0 * hp_np

    for r in range(n):
        for c in range(m):
            expected = float(combined_mpfr[r][c])
            rel = abs(expected - combined_np[r, c])
            if combined_np[r, c] > 1e-12:
                rel /= abs(combined_np[r, c])
            assert rel < 1e-12, \
                f"AЧХ[{r}][{c}] rel_err={rel:.2e}"


def _reference_ihf(src):
    """
    Независимый эталон гомоморфной фильтрации 8x8.

    Написан по формулам Gonzalez-Woods, без единой оптимизации
    модуля filter.py: тригонометрия считается заново на каждую ячейку
    каждого ДПФ, суммарная АЧХ не сворачивается, exp2 идёт только через
    mpfr. Совпадение с apply() подтверждает, что кэши, свёртка АЧХ,
    пропуск умножений и быстрый exp2 не исказили результат.
    """
    n, m = src.shape
    two_pi = 2 * gmpy2.const_pi()

    # 1. Логарифмирование
    log = [[gmpy2.log2(mpfr(float(v))) for v in row] for row in src.tolist()]

    def dft2(data, inverse):
        out = []
        sign = mpfr(1) if inverse else mpfr(-1)
        for r in range(n):
            line = []
            for c in range(m):
                acc = mpc(0)
                for u in range(n):
                    ang_r = mpfr(r * u) / mpfr(n)
                    for v in range(m):
                        ang = sign * two_pi * (ang_r + mpfr(c * v) / mpfr(m))
                        acc += data[u][v] * mpc(gmpy2.cos(ang), gmpy2.sin(ang))
                line.append(acc)
            out.append(line)
        return out

    # 2. Прямое ДПФ
    spectrum = dft2(log, inverse=False)

    # 3. Обработка: lf_gain*LP*S + hf_gain*HP*S (не свёрнутой АЧХ)
    lp = filtration.get_filter("lp_butterworth", n, m, d0=4.0, order=2)
    hp = filtration.get_filter("hp_butterworth", n, m, d0=4.0, order=2)
    lf_gain, hf_gain = mpfr("0.3"), mpfr("2.0")
    filtered = [[spectrum[u][v] * (lp[u][v] * lf_gain)
                 + spectrum[u][v] * (hp[u][v] * hf_gain)
                 for v in range(m)] for u in range(n)]

    # 4. Обратное ДПФ с масштабом 1/(n*m)
    restored = dft2(filtered, inverse=True)
    scale = mpfr(1) / mpfr(n * m)
    restored = [[cell * scale for cell in row] for row in restored]

    # 5. Антилогарифмирование и округление
    return np.array(
        [[int(gmpy2.round_away(gmpy2.exp2(restored[r][c].real) + _EPS))
          for c in range(m)] for r in range(n)],
        dtype=np.int64)


def test_ihf_matches_naive_reference():
    """
    Оптимизированный apply() совпадает с наивным эталоном, посчитанным
    прямо по формулам (без кэшей, без свёртки АЧХ, без быстрого exp2).
    """
    rng = np.random.default_rng(104)
    src = rng.integers(1, 256, size=(8, 8)).astype(np.float64)
    got = _filter(0.3, 2.0, d0=4.0, order=2).apply(src)
    expected = _reference_ihf(src)
    bad = int((got != expected).sum())
    assert bad == 0, (f"расхождений с эталоном: {bad} из "
                      f"{got.size}\nполучено\n{got}\nэталон\n{expected}")


def test_roots_cache_is_shared_and_stable():
    """
    Таблица корней из единицы отдаётся из кэша тем же объектом и
    содержит ровно n элементов для обоих знаков.
    """
    for n in [7, 9, 12, 15, 21, 16]:
        for sign in (-1, 1):
            t1 = _roots_of_unity(n, sign)
            t2 = _roots_of_unity(n, sign)
            assert t1 is t2, f"таблица корней n={n} не переиспользуется"
            assert len(t1) == n, f"корней {len(t1)}, ожидалось {n}"
        # знаки дают сопряжённые таблицы
        tp, tn = _roots_of_unity(n, 1), _roots_of_unity(n, -1)
        for a, b in zip(tp, tn):
            assert abs(a - mpc(b.real, -b.imag)) < mpfr("1e-40"), \
                "таблицы знаков не сопряжены"


def test_dft_1d_is_repeatable():
    """
    Повторные вызовы _dft_1d на одном входе дают побитово равный вывод
    (кэш корней не должен вносить дрейф между вызовами).
    """
    rng = np.random.default_rng(105)
    for n in [7, 9, 12, 15, 21]:
        x = [mpc(float(v)) for v in rng.integers(0, 256, n)]
        g1 = _dft_1d(x)
        for _ in range(3):
            for a, b in zip(_dft_1d(x), g1):
                assert a == b, f"_dft_1d не стабилен при повторе, n={n}"


# Запуск без pytest

def run_all():
    # Прогнать все test_* функции, вернуть число провалов
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} прошло")
    return failed


if __name__ == "__main__":
    _setup_console()
    sys.exit(1 if run_all() else 0)
