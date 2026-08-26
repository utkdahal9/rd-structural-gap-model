import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regions import get_primary_state, assign_region, add_region_column


def test_get_primary_state_simple():
    assert get_primary_state("Boston-Cambridge-Newton, MA-NH") == "MA"


def test_get_primary_state_multi_word_city():
    assert get_primary_state("New York-Newark-Jersey City, NY-NJ-PA") == "NY"


def test_get_primary_state_no_comma():
    assert get_primary_state("Not An MSA Name") is None


def test_assign_region_northeast():
    assert assign_region("Boston-Cambridge-Newton, MA-NH") == "Northeast"


def test_assign_region_west():
    assert assign_region("Seattle-Tacoma-Bellevue, WA") == "West"


def test_assign_region_unknown():
    assert assign_region("Nowhere, ZZ") is None


def test_add_region_column():
    df = pd.DataFrame({"MSA_Name": ["Boston-Cambridge-Newton, MA-NH", "Houston-Pasadena-The Woodlands, TX"]})
    out = add_region_column(df)
    assert list(out["Region"]) == ["Northeast", "South"]
