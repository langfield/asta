import sys
import inspect
import importlib

MODULES = []


def bind(*args) -> None:
    """ Save modules/classes/functions to refresh with updated dims. """
    for arg in args:
        MODULES.append(arg)


def refresh() -> None:
    """ Force import everything in ``MODULES``. """
    frame = inspect.stack()[1]
    caller_module = inspect.getmodule(frame[0])
    members = list(get_members(sys.modules[caller_module.__name__]))
    for mem in members:
        exec(f"global {mem.alias}")

        # If its not a module, it has a parent module.
        if mem.module:
            module = sys.modules[mem.module]
            importlib.reload(module)
            exec(f"from {mem.module} import {mem.name} as {mem.alias}")
        else:
            print(f"Reloading: {mem.name}")
            module = sys.modules[mem.name]
            importlib.reload(module)
            exec(f"import {mem.name} as {mem.alias}")


class _Import:
    """ A struct for holding import information. """

    def __init__(self, alias: str, name: str, module: str = "") -> None:
        self.alias = alias
        self.name = name
        self.module = module

    def __repr__(self) -> str:
        rep = f"<_Import | alias: '{self.alias}' | name: '{self.name}' "
        rep += f"| module: '{self.module}'>"
        return rep


def get_members(module):
    """ Return a generator of module information maps. """
    # Only get members which are modules or functions.
    predicate = lambda x: inspect.ismodule(x) or inspect.isfunction(x)
    for name, member in inspect.getmembers(module, predicate):
        try:
            modulename = member.__module__
            if modulename != "__main__":
                yield _Import(name, member.__name__, member.__module__)
        except:
            if inspect.ismodule(member):
                yield _Import(name, member.__name__)
