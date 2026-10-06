import pandas as pd
import pytest

from ml.utm_sheet import cell_id_for_lonlat, decode, quadrant_bounds

# Rows from land_14_phra-nakhon-si-ayutthaya.csv, hand-checked against the quadrant extents.
AYUTTHAYA_ROWS = [
    # UTMMAP1, UTMMAP2, UTMMAP3, expected (e_km, n_km)
    (5137, 4, 6888, (668, 1588)),
    (5037, 2, 3668, (636, 1568)),
    (5037, 1, 4682, (646, 1582)),
    (5137, 4, 8676, (686, 1576)),
]


def test_quadrant_bounds_ayutthaya():
    # 5137 IV (NW quadrant) contains Ayutthaya town (100.57E, 14.35N)
    assert quadrant_bounds(5137, 4) == (100.5, 14.25, 100.75, 14.5)


@pytest.mark.parametrize("m1,m2,m3,expected", AYUTTHAYA_ROWS)
def test_decode_known_rows(m1, m2, m3, expected):
    out = decode(pd.DataFrame({"UTMMAP1": [m1], "UTMMAP2": [m2], "UTMMAP3": [m3]})).iloc[0]
    assert (out.e_km, out.n_km) == expected
    assert out.zone == 47
    assert out.consistent


def test_decode_lat_lon_is_ayutthaya_town():
    out = decode(pd.DataFrame({"UTMMAP1": [5137], "UTMMAP2": [4], "UTMMAP3": [6888]})).iloc[0]
    assert out.lat == pytest.approx(14.36, abs=0.03)
    assert out.lon == pytest.approx(100.57, abs=0.03)


def test_sheet_without_4000_number_is_rejected():
    out = decode(pd.DataFrame({"UTMMAP1": [5023], "UTMMAP2": [1], "UTMMAP3": [0]})).iloc[0]
    assert not out.consistent


def test_neighbouring_quadrant_label_is_tolerated():
    # Songkhla row labelled 5023 I but decoding to Hat Yai (7.0N, 100.47E), just south of it
    out = decode(pd.DataFrame({"UTMMAP1": [5023], "UTMMAP2": [1], "UTMMAP3": [6272]})).iloc[0]
    assert (out.e_km, out.n_km) == (662, 772)
    assert 0 < out.offset_km <= 30
    assert out.consistent
    assert out.lat == pytest.approx(7.0, abs=0.05)


def test_cell_id_roundtrip():
    out = decode(pd.DataFrame({"UTMMAP1": [5137], "UTMMAP2": [4], "UTMMAP3": [6888]})).iloc[0]
    assert cell_id_for_lonlat(out.lon, out.lat) == out.cell_id == "47_668_1588"
