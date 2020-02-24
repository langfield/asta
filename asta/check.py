""" Module for setting the typecheck environment variable. """
import os


def on() -> None:
    """ Turn typechecking on. """
    os.environ["ASTA_TYPECHECK"] = "1"


def off() -> None:
    """ Turn typechecking off. """
    os.environ["ASTA_TYPECHECK"] = "0"
