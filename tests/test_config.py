import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
from spacesense import config
import pytest


def test_ConfigEntry():
    """
        Test the Config_Entry function
    """

    temp_entry = config.ConfigEntry("group_value", "key", "default_value")
    temp_entry.set("a")
    assert temp_entry.get() == "a"

    temp_entry_withset = config.ConfigEntry("group_value", "key", "default_value")

    temp_entry_withset.set("value_set")

    assert temp_entry.get() == "value_set"


def test_fname():


    fname = config.get_config_file()

    assert os.path.isfile(fname)
