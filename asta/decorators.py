"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
import inspect
from typing import Any, Tuple, Dict, Set, List

from sympy import solvers
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from asta.origins import check_annotation
from asta.display import (
    fail_system,
    get_header,
)


def validate_annotations(  # type: ignore[no-untyped-def]
    decorated, annotations: Dict[str, Any], args: Tuple[Any], kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make sure there is an annotation for each parameter, return arguments in
    args other than class references (self, cls, mcs), and keyword arguments in
    kwargs with defaults filled in.
    """
    num_annots = len(annotations)
    num_non_return_annots = num_annots
    if "return" in annotations:
        num_non_return_annots -= 1

    # Get the parameter list from the function signature.
    sig = inspect.signature(decorated)
    paramlist = list(sig.parameters)

    # Get a list of kwargs with defaults filled-in.
    defaults: Dict[str, Any] = {}
    for k, v in sig.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default

    checkable_args: Dict[str, Any] = defaults.copy()
    checkable_args.update(kwargs)

    # Remove unannotated instance/class/metaclass reference.
    # TODO: Consider checking this more idiomatically with ``__func__`` or ``ismethod``.
    pure_args = args
    refs = ("self", "cls", "mcs")
    if len(sig.parameters) == num_non_return_annots + 1 and paramlist[0] in refs:
        pure_args = pure_args[1:]  # type: ignore[assignment]
    for i, arg in enumerate(pure_args):
        name = list(annotations.keys())[i]
        checkable_args[name] = arg

    # Check for mismatch between lengths of arguments/annotations.
    if num_non_return_annots != len(checkable_args):
        num_annot_err = f"Mismatch between number of annotated "
        num_annot_err += f"non-(self / cls / mcs) parameters "
        num_annot_err += f"'({num_non_return_annots})' and number of arguments "
        num_annot_err += f"'({len(checkable_args)})'. "
        num_annot_err += f"There may be a type annotation missing."
        raise TypeError(num_annot_err)

    return checkable_args


def typechecked(decorated):  # type: ignore[no-untyped-def]
    """
    Typecheck a function annotated with ``asta`` type objects. This decorator
    will only check the shape and datatype of parameters annotated with
    ``asta`` type variables.  Mypy should be used for everything else. Note
    that the argument-parameter assignment problem is trivial because the only
    arguments which can be out of order are keyword arguments.

    Parameters
    ----------
    decorated : ``Callable[[Any], Any]``.
        The function to be typechecked.

    Returns
    -------
    _wrapper : ``Callable[[Any], Any]``.
        The decorated version of ``decorated``.
    """
    if "ASTA_TYPECHECK" not in os.environ or os.environ["ASTA_TYPECHECK"] == "0":
        return decorated

    # Treat classes.
    if inspect.isclass(decorated):

        # Grab the module name.
        prefix = decorated.__qualname__ + "."

        # DEBUG
        print("Prefix:", prefix)

        # Iterate over attributes.
        for key, attr in decorated.__dict__.items():

            # If it's decoratable.
            if (
                inspect.isfunction(attr)
                or inspect.ismethod(attr)
                or inspect.isclass(attr)
            ):
                # If the name prefix matches and it has annotations.
                if attr.__qualname__.startswith(prefix) and getattr(
                    attr, "__annotations__", None
                ):

                    # Decorate the method/function/class.
                    setattr(decorated, key, typechecked(attr))

            # Only for class and staticmethods; instance methods are caught above.
            elif isinstance(attr, (classmethod, staticmethod)):

                # If the underlying function has annotations.
                if getattr(attr.__func__, "__annotations__", None):
                    wrapped = typechecked(attr.__func__)

                    # Re-wrap with ``classmethod`` or ``staticmethod`` and put back.
                    setattr(decorated, key, type(attr)(wrapped))

        return decorated

    def _wrapper(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        """ Decorated/typechecked function. """

        # Print header for ``decorated``.
        header: str = get_header(decorated)
        print(header)

        equations: Set[Expr] = set()
        annotations: Dict[str, Any] = decorated.__annotations__
        checkable_args: Dict[str, Any] = validate_annotations(
            decorated,
            annotations,
            args,  # type: ignore
            kwargs,
        )

        # Check arguments.
        for name, arg in checkable_args.items():
            annotation = annotations[name]
            equations = check_annotation(name, arg, annotation, equations)
        del annotation

        # TODO: Treat lists, sequences recursively.
        # Check return.
        ret = decorated(*args, **kwargs)
        annotation = annotations["return"]
        equations = check_annotation("return", ret, annotation, equations)
        del annotation

        # TODO: Consider putting this in its own function.
        # Solve our system of equations if it is nonempty.
        if equations:
            symbols: Set[Symbol] = set()
            for equation in equations:
                symbols = symbols.union(equation.free_symbols)
            solutions: List[Dict[Symbol, int]] = solvers.solve(
                equations, symbols, dict=True
            )

            # If we don't get a unique solution, it's not a match.
            if len(solutions) != 1:
                halt = os.environ["ASTA_TYPECHECK"] == "2"
                fail_system(equations, symbols, solutions, halt)

        return ret

    _wrapper.__module__ = decorated.__module__
    _wrapper.__name__ = decorated.__name__

    return _wrapper
