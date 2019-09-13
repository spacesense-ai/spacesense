import configparser
import os
from appdirs import user_config_dir
from ..utils import mkdir


"""Test first if there is a .config.ini file here, else take the default"""
local_config = os.getcwd() + "/.config.ini"

if os.path.isfile(local_config):
    _CONFIG_FNAME = local_config
else:
    _CONFIG_FNAME = str(user_config_dir(appname="spacesence", appauthor="spacesence")) + "/config.ini"

mkdir(os.path.dirname(_CONFIG_FNAME))
_config = configparser.ConfigParser()
_config.read(_CONFIG_FNAME)


def get_config_file():
    """return the name of the configuration file"""
    return _CONFIG_FNAME


class ConfigEntry:
    """One entry of the config, allows to get the key in a group with a default value"""

    def __init__(self, group, key2, default=""):
        self.group = group
        self.key2 = key2
        self.default = default

    def get(self):
        """Get the key if it is in the config file, else return the default"""
        try:
            return _config[self.group][self.key2]
        except KeyError:
            return self.default

    def set(self, value):
        """Set the value in both the object and the configuration file"""
        if self.group not in _config:
            _config.add_section(self.group)
        _config[self.group][self.key2] = value
        with open(_CONFIG_FNAME, 'w') as f:
            _config.write(f)


global_workdir = ConfigEntry("Global", "workdir", "/tmp")


def log_dir():
    return global_workdir.get() + "/logs"


data_dir = ConfigEntry("Global", "datadir", "/tmp/data")

copernicus_login = ConfigEntry("Copernicus", "login")
copernicus_password = ConfigEntry("Copernicus", "password")

earthdata_login = ConfigEntry("Earthdata", "login")
earthdata_password = ConfigEntry("Earthdata", "password")
