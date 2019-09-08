from spacesense import config
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
import pytest


def test_ConfigEntry():
    """
        Test the Config_Entry function
    """

    temp_entry = config.ConfigEntry("group_value", "key", "default_value")

    got_value = temp_entry.get()
    assert got_value == "default_value"

    temp_entry.set("value_set")
    got_value = temp_entry.get()

    assert got_value == "value_set"


def test_fname():


    fname = config.get_config_file()

    assert os.path.isfile(fname)
