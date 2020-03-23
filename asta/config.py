#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Configuration parsing and loading (credit to pylint authors). """
import os
import configparser
from typing import Generator, Optional

import toml


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
    """search the asta rc file and return its path if it find it, else None
    """
    for config_file in find_default_config_files():
        if config_file.endswith("astarc"):
            return config_file

    return None


def sample_read():
    """
    use_config_file = config_file and os.path.exists(config_file)
    if use_config_file:
        parser = self.cfgfile_parser

        if config_file.endswith(".toml"):
            with open(config_file, "r") as fp:
                content = toml.load(fp)

            try:
                sections_values = content["tool"]["pylint"]
            except KeyError:
                pass
            else:
                for section, values in sections_values.items():
                    parser._sections[section.upper()] = values
        else:
            # Use this encoding in order to strip the BOM marker, if any.
            with io.open(config_file, "r", encoding="utf_8_sig") as fp:
                parser.read_file(fp)

            # normalize sections'title
            for sect, values in list(parser._sections.items()):
                if sect.startswith("pylint."):
                    sect = sect[len("pylint.") :]
                if not sect.isupper() and values:
                    parser._sections[sect.upper()] = values
    """
    raise NotImplementedError
