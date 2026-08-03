import json
from pathlib import Path

from saferestore.datasets import load_carinthia_s
from scripts.prepare_splits import stable_fraction


def test_stable_fraction_is_deterministic():
    assert stable_fraction("abc", 7) == stable_fraction("abc", 7)
    assert stable_fraction("abc", 7) != stable_fraction("abc", 8)
