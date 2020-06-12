#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Configuration parsing and loading (credit to pylint authors). """
import io
import os
import collections
import configparser
from typing import Any, Dict, Optional, Generator

import toml
from oxentiel import Oxentiel

from asta import _internal


def _toml_has_config(path: str) -> bool:
    with open(path, "r") as toml_handle:
        content = toml.load(toml_handle)
        try:
            content["tool"]["asta"]
        except KeyError:
            return False

    return True


def _cfg_has_config(path: str) -> bool:
    parser = configparser.ConfigParser()
    parser.read(path)
    return any(section.startswith("asta.") for section in parser.sections())


def find_default_config_files() -> Generator[str, None, None]:
    """ Find all possible config files. """
    rc_names = ("astarc", ".astarc")
    config_names = rc_names + ("pyproject.toml", "setup.cfg")
    for config_name in config_names:
        if os.path.isfile(config_name):
            if config_name.endswith(".toml") and not _toml_has_config(config_name):
                continue
            if config_name.endswith(".cfg") and not _cfg_has_config(config_name):
                continue

            yield os.path.abspath(config_name)

    if os.path.isfile("__init__.py"):
        curdir = os.path.abspath(os.getcwd())
        while os.path.isfile(os.path.join(curdir, "__init__.py")):
            curdir = os.path.abspath(os.path.join(curdir, ".."))
            for rc_name in rc_names:
                rc_path = os.path.join(curdir, rc_name)
                if os.path.isfile(rc_path):
                    yield rc_path

    if "ASTARC" in os.environ and os.path.exists(os.environ["ASTARC"]):
        if os.path.isfile(os.environ["ASTARC"]):
            yield os.environ["ASTARC"]
    else:
        user_home = os.path.expanduser("~")
        if user_home not in ("~", "/root"):
            home_rc = os.path.join(user_home, ".astarc")
            if os.path.isfile(home_rc):
                yield home_rc
            home_rc = os.path.join(user_home, ".config", "astarc")
            if os.path.isfile(home_rc):
                yield home_rc

    if os.path.isfile("/etc/astarc"):
        yield "/etc/astarc"


def find_astarc() -> Optional[str]:
    """ Look for an ``astarc`` file. Return it if found, else return ``None``. """
    for config_file in find_default_config_files():
        if config_file.endswith("astarc"):
            return config_file
    return None


# pylint: disable=protected-access
def read_config(config_path: str) -> configparser.ConfigParser:
    """ Parse a config file from ``config_path``, given a file exists there. """
    parser = configparser.ConfigParser()
    if config_path.endswith(".toml"):
        with open(config_path, "r") as toml_file:
            content = toml.load(toml_file)

        try:
            sections_values = content["tool"]["asta"]
        except KeyError:
            pass
        else:
            for section, values in sections_values.items():
                parser._sections[section.upper()] = values  # type: ignore[attr-defined]
    else:
        # Use this encoding in order to strip the BOM marker, if any.
        with io.open(config_path, "r", encoding="utf_8_sig") as config_file:
            parser.read_file(config_file)

        # Normalize sections' titles.
        for sect, values in list(parser._sections.items()):  # type: ignore
            if sect.startswith("asta."):
                sect = sect[len("asta.") :]
            if not sect.isupper() and values:
                parser._sections[sect.upper()] = values  # type: ignore[attr-defined]
    return parser


def as_dict(config: configparser.ConfigParser) -> Dict[str, Dict[str, str]]:
    """ Returns a ``ConfigParser`` object as a dictionary. """
    dictionary: Dict[str, Dict[str, str]] = collections.OrderedDict()
    for section in config.sections():
        dictionary[section] = {}
        for key, val in config.items(section):
            dictionary[section][key] = val
    return dictionary


def get_ox() -> Oxentiel:
    """ Returns a configuration file. """
    ox = _internal.ox
    if not ox:
        path = find_astarc()
        parse = path and os.path.exists(path)
        if parse and path:
            parser = configparser.ConfigParser()
            parser = read_config(path)
            if "MASTER" in parser:
                settings = as_dict(parser)["MASTER"]
            else:
                settings = collections.OrderedDict()
        else:
            settings = collections.OrderedDict()

        # Read defaults.
        asta_dir = os.path.dirname(os.path.realpath(__file__))
        default_path = os.path.join(asta_dir, "defaults/astarc")
        assert os.path.isfile(default_path)
        defaultparser = configparser.ConfigParser()
        defaultparser = read_config(default_path)
        assert "MASTER" in defaultparser
        defaults = as_dict(defaultparser)["MASTER"]

        # Set defaults.
        for key, val in defaults.items():
            if key not in settings:
                settings[key] = val

        new_settings = collections.OrderedDict()
        for key, val in settings.items():
            new_val: Any = val
            if val in ("yes", "no"):
                # pylint: disable=simplifiable-if-expression
                new_val = True if val == "yes" else False
            new_settings[key.replace("-", "_")] = new_val
        settings = new_settings

        ox = Oxentiel(settings, mutable=True)
        _internal.ox = ox
    return ox
