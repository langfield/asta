#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions for checking type annotations and their origin types. """
import inspect
import collections
from io import TextIOBase, RawIOBase, IOBase, BufferedIOBase
from typing import (
    Any,
    Tuple,
    Set,
    Dict,
    List,
    AbstractSet,
    Callable,
    Union,
    IO,
    Sequence,
    Type,
    TextIO,
    BinaryIO,
)

from oxentiel import Oxentiel
from sympy.core.expr import Expr

import asta.dims
from asta.dims import Placeholder
from asta.utils import shapecheck, attrcheck
from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.display import (
    get_type_name,
    qualified_name,
    type_representation,
    pass_argument,
    fail_argument,
    fail_tuple,
    fail_tuple_length,
    fail_namedtuple,
    fail_empty_tuple,
    fail_uninitialized,
    fail_list,
    fail_sequence,
    fail_dict,
    fail_set,
    fail_keys,
    fail_callable,
    fail_union,
    fail_class,
    fail_subclass,
    fail_io,
    fail_text_io,
    fail_binary_io,
    fail_complex,
    fail_float,
    fail_protocol,
    fail_literal,
    fail_too_many_args,
    fail_too_few_args,
    fail_fallback,
)
from asta.constants import torch, _TORCH_IMPORTED, _TENSORFLOW_IMPORTED

METAMAP: Dict[type, SubscriptableMeta] = {_ArrayMeta: Array}

if _TORCH_IMPORTED:
    from asta.tensor import Tensor
    from asta._tensor import _TensorMeta

    METAMAP[_TensorMeta] = Tensor

if _TENSORFLOW_IMPORTED:
    from asta.tftensor import TFTensor
    from asta._tftensor import _TFTensorMeta

    METAMAP[_TFTensorMeta] = TFTensor

try:
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    Literal = None

# pylint: disable=too-many-lines

def refresh(
    annotation: SubscriptableMeta, ox: Oxentiel
) -> Tuple[SubscriptableMeta, bool]:
    """ Load an asta type annotation containing classical placeholders. """
    dtype = annotation.dtype
    shape = annotation.shape
    dimvars = []
    initialized = True
    uninitialized_placeholder_names: Set[str] = set()

    if annotation.shape is not None:
        for item in annotation.shape:

            # Handle fixed placeholders.
            if isinstance(item, Placeholder):
                placeholder = item
                dimvar = getattr(asta.dims, placeholder.name)

                # Catch uninitialized placeholders.
                if isinstance(dimvar, Placeholder):
                    initialized = False
                    name = placeholder.name
                    if name not in uninitialized_placeholder_names:
                        fail_uninitialized(name, ox)
                    uninitialized_placeholder_names.add(name)

                # Handle case where placeholder is unpacked in annotation.
                if placeholder.unpacked:
                    for elem in dimvar:
                        dimvars.append(elem)
                else:
                    dimvars.append(dimvar)
            else:
                dimvars.append(item)
        assert len(dimvars) == len(shape)
        shape = tuple(dimvars)

    # Note we're guaranteed that ``annotation`` has type ``SubscriptableMeta``.
    subscriptable_class = METAMAP[type(annotation)]

    refreshed_annotation: SubscriptableMeta
    if shape is not None:
        refreshed_annotation = subscriptable_class[dtype, shape]  # type: ignore
    else:
        refreshed_annotation = subscriptable_class[dtype]

    return refreshed_annotation, initialized


def check_asta(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check asta subscriptable class types. """

    annotation, initialized = refresh(annotation, ox)
    if not initialized:
        return equations

    # Check if the literal ``value`` matches the annotation.
    rep: str = type_representation(value)

    # If the isinstance check fails, print/raise an error.
    if not isinstance(value, annotation):
        fail_argument(name, annotation, rep, ox)

    # Otherwise, print a pass.
    else:
        pass_argument(name, annotation, rep, ox)

        # Update equation set.
        shape_equations: Set[Expr] = set()
        attr_equations: Set[Expr] = set()

        if annotation.shape is not None:

            # HARDCODE
            # Handle case where type(shape) != tuple, e.g. ``torch.Size``.
            value_shape = tuple(value.shape)
            assert not isinstance(value_shape, torch.Size)

            # Grab equations from shapecheck call.
            shape_match, shape_equations = shapecheck(value_shape, annotation.shape)
            attr_match, attr_equations = attrcheck(value, annotation.kwattrs)
            assert shape_match and attr_match

        equations = equations.union(shape_equations, attr_equations)

    return equations


def check_typed_dict(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Typecheck a typed dict. """

    # If the argument is not a dict, we're aleady in trouble.
    if not isinstance(value, dict):
        fail_dict(name, annotation, qualified_name(value), ox)
        return equations

    expected_keys = frozenset(annotation.__annotations__)
    existing_keys = frozenset(value)

    extra_keys = existing_keys - expected_keys
    if extra_keys:
        fail_keys(name, expected_keys, existing_keys, ox)

    if annotation.__total__:
        missing_keys = expected_keys - existing_keys
        if missing_keys:
            fail_keys(name, expected_keys, existing_keys, ox)

    # pylint: disable=invalid-name
    MISSING_VALUE = object()
    for key, argtype in annotation.__annotations__.items():
        argvalue = value.get(key, MISSING_VALUE)
        if argvalue is not MISSING_VALUE:
            equations = check_annotation(
                f"{name}[{key}]", argvalue, argtype, equations, ox
            )
    return equations


def check_callable(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Check an argument with annotation ``callable`` or ``Callable[]``. """
    if not callable(value):
        fail_callable(name, annotation, qualified_name(value), ox)

    if annotation.__args__:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return

        if hasattr(annotation, "__result__"):
            # Python 3.5
            argument_types = annotation.__args__
            check_args = argument_types is not Ellipsis
        else:
            # Python 3.6+
            argument_types = annotation.__args__[:-1]
            check_args = argument_types != (Ellipsis,)

        if check_args:
            # The callable must not have keyword-only arguments without defaults.
            unfulfilled_kwonlyargs = [
                param.name
                for param in signature.parameters.values()
                if param.kind == inspect.Parameter.KEYWORD_ONLY
                and param.default == inspect.Parameter.empty
            ]
            if unfulfilled_kwonlyargs:
                # NOTE: Should this output a warning?
                return

            num_mandatory_args = len(
                [
                    param.name
                    for param in signature.parameters.values()
                    if param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and param.default is inspect.Parameter.empty
                ]
            )
            has_varargs = any(
                param
                for param in signature.parameters.values()
                if param.kind == inspect.Parameter.VAR_POSITIONAL
            )

            # NOTE: Consider adding true recursive checking.
            if num_mandatory_args > len(argument_types):
                fail_too_many_args(name, len(argument_types), num_mandatory_args, ox)
            elif not has_varargs and num_mandatory_args < len(argument_types):
                fail_too_few_args(name, len(argument_types), num_mandatory_args, ox)
    return


def check_tuple(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check an argument with annotation ``tuple`` or ``Tuple[]``. """

    # Specialized check for NamedTuples.
    if hasattr(annotation, "_field_types"):

        if not isinstance(value, annotation):
            fail_namedtuple(name, qualified_name(annotation), qualified_name(value), ox)
            return equations

        # pylint: disable=protected-access
        for subname, field_type in annotation._field_types.items():
            equations = check_annotation(
                f"{name}.{subname}", getattr(value, subname), field_type, equations, ox
            )

    elif not isinstance(value, tuple):
        fail_tuple(name, qualified_name(value), ox)

    elif not getattr(annotation, "__args__", None):
        # Unparametrized Tuple or plain tuple.
        pass

    else:
        # Python 3.6+
        use_ellipsis = annotation.__args__[-1] is Ellipsis

        # Remove the ``Ellipsis`` from the end if it exists.
        if use_ellipsis:
            tuple_params = annotation.__args__[:-1]
        else:
            tuple_params = annotation.__args__

        if use_ellipsis:
            elem_type = tuple_params[0]
            for i, elem in enumerate(value):
                equations = check_annotation(
                    f"{name}[{i}]", elem, elem_type, equations, ox
                )
        elif tuple_params == ((),):
            if value != ():
                fail_empty_tuple(name, qualified_name(value), ox)
        else:
            if len(value) != len(tuple_params):
                ann_rep = qualified_name(annotation)
                fail_tuple_length(name, len(tuple_params), ann_rep, len(value), ox)
                return equations

            for i, (elem, elem_type) in enumerate(zip(value, tuple_params)):
                equations = check_annotation(
                    f"{name}[{i}]", elem, elem_type, equations, ox
                )

    return equations


def check_list(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check an argument with annotation ``list`` or ``List[]``. """
    if not isinstance(value, list):
        fail_list(name, annotation, qualified_name(value), ox)

    # If annotation is a subscriptable generic (``List[...]``).
    if annotation is not list:
        if annotation.__args__ not in (None, annotation.__parameters__):
            value_type = annotation.__args__[0]
            if value_type is not Any:

                value_iterable = value
                if not ox.check_all_sequence_elements:
                    value_iterable = value_iterable[:1]
                for i, v in enumerate(value_iterable):
                    element_equations = check_annotation(
                        f"{name}[{i}]", v, value_type, equations, ox
                    )
                    equations = equations.union(element_equations)

    return equations


def check_sequence(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check an argument with annotation ``Sequence[*]``. """
    if not isinstance(value, collections.abc.Sequence):
        fail_sequence(name, annotation, qualified_name(value), ox)

    # Consider removing this test or even just figuring out what it does?
    if annotation.__args__ not in (None, annotation.__parameters__):
        value_type = annotation.__args__[0]
        if value_type is not Any:

            value_iterable = value
            if not ox.check_all_sequence_elements:
                value_iterable = value_iterable[:1]
            for i, v in enumerate(value_iterable):
                element_equations = check_annotation(
                    f"{name}[{i}]", v, value_type, equations, ox
                )
                equations = equations.union(element_equations)

    return equations


def check_dict(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check an argument with annotation ``Dict[*]``. """
    if not isinstance(value, dict):
        fail_dict(name, annotation, qualified_name(value), ox)

    # Equivalently: if annotation is ``Dict[*]``.
    elif annotation is not dict:
        if annotation.__args__ not in (None, annotation.__parameters__):
            key_type, value_type = annotation.__args__
            if key_type is not Any or value_type is not Any:
                for k, v in value.items():
                    key_equations = check_annotation(
                        f"{name}.<key>", k, key_type, equations, ox
                    )
                    val_equations = check_annotation(
                        f"{name}[{k}]", v, value_type, equations, ox
                    )

                    equations = equations.union(key_equations, val_equations)

    return equations


def check_set(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check an argument with annotation ``Set[*]``. """
    if not isinstance(value, AbstractSet):
        fail_set(name, annotation, qualified_name(value), ox)

    # Equivalently: if annotation is ``Set[*]``.
    if annotation is not set:
        if annotation.__args__ not in (None, annotation.__parameters__):
            value_type = annotation.__args__[0]
            if value_type is not Any:
                for v in value:
                    set_equations = check_annotation(
                        f"{name}.<set_element>", v, value_type, equations, ox
                    )
                    equations = equations.union(set_equations)

    return equations


def check_union(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Typecheck an argument annotated with ``Union[]``. """
    if hasattr(annotation, "__union_params__"):
        # Python 3.5
        union_params = annotation.__union_params__
    else:
        # Python 3.6+
        union_params = annotation.__args__

    for type_ in union_params:
        try:
            equations = check_annotation(name, value, type_, equations, ox)
            return equations
        except TypeError:
            pass

    typelist = ", ".join(get_type_name(t) for t in union_params)
    fail_union(name, typelist, qualified_name(value), ox)
    return equations


def check_class(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Check an arbitrary class. """
    if not inspect.isclass(value):
        fail_class(name, annotation, qualified_name(value), ox)

    # Needed on Python 3.7+
    if annotation is Type:
        return

    expected_class = annotation.__args__[0] if annotation.__args__ else None
    if expected_class and not issubclass(value, expected_class):
        supercls = qualified_name(expected_class)
        fail_subclass(name, supercls, qualified_name(value), ox)

    return


def check_number(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Check for when ``value`` is in ``(complex, float, int)``. """
    if annotation is complex and not isinstance(value, complex):
        fail_complex(name, qualified_name(value.__class__), ox)
    elif annotation is float and not isinstance(value, float):
        fail_float(name, qualified_name(value.__class__), ox)


def check_io(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Typecheck arguments with *IO annotations. """
    if annotation is TextIO:
        if not isinstance(value, TextIOBase):
            fail_text_io(name, qualified_name(value.__class__), ox)
    elif annotation is BinaryIO:
        if not isinstance(value, (RawIOBase, BufferedIOBase)):
            fail_binary_io(name, qualified_name(value.__class__), ox)
    elif not isinstance(value, IOBase):
        fail_io(name, qualified_name(value.__class__), ox)


def check_protocol(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Typecheck arguments with protocol annotations. """
    if not issubclass(type(value), annotation):
        fail_protocol(name, type(value).__qualname__, annotation.__qualname__, ox)


def check_literal(name: str, value: Any, annotation: Any, ox: Oxentiel) -> None:
    """ Typecheck a value annotated with ``Literal[*]``. """
    if value not in annotation.__args__:
        fail_literal(name, annotation.__args__, value, ox)


# Equality checks are applied to these.
ORIGIN_TYPE_CHECKERS = {
    AbstractSet: check_set,
    Callable: check_callable,
    collections.abc.Callable: check_callable,
    dict: check_dict,
    Dict: check_dict,
    list: check_list,
    List: check_list,
    Sequence: check_sequence,
    collections.abc.Sequence: check_sequence,
    collections.abc.Set: check_set,
    set: check_set,
    Set: check_set,
    tuple: check_tuple,
    Tuple: check_tuple,
    type: check_class,
    Union: check_union,
}

# We ought to be above version 3.5.2.
assert Type is not None
NON_ASTA_INITIAL_CHECKERS = {
    Callable: check_callable,
    collections.abc.Callable: check_callable,
    type: check_class,
    Type: check_class,
}
if Literal is not None:
    NON_ASTA_CHECKERS = {**NON_ASTA_INITIAL_CHECKERS, **{Literal: check_literal}}
else:
    NON_ASTA_CHECKERS = NON_ASTA_INITIAL_CHECKERS


def check_annotation(
    name: str, value: Any, annotation: Any, equations: Set[Expr], ox: Oxentiel
) -> Set[Expr]:
    """ Check if ``value`` is of type ``annotation`` for asta types only. """

    # The solution to the set of equations for each individual annotation
    # should be unique. If there is no solution, print/raise an error. If there
    # are multiple solutions, raise an error; someone is trying to do inference
    # on way too many variables.

    # Thus there should be no need to iterate over the function signature more
    # than once. A single isinstance check should always be sufficient to
    # determine if there is a solution. However, that does not mean that single
    # isinstance checks are self-contained. In addition to specifying a unique
    # solution, all the annotations' solutions must agree for a given function
    # signature.

    # Get origin type.
    origin: Any = getattr(annotation, "__origin__", None)

    # Treat asta types.
    if isinstance(annotation, SubscriptableMeta):
        equations = check_asta(name, value, annotation, equations, ox)

    elif origin is not None:
        checker_function_map = ORIGIN_TYPE_CHECKERS
        if ox.check_non_asta_types:
            checker_function_map.update(NON_ASTA_CHECKERS)
        checker_fn: Callable = checker_function_map[origin]  # type: ignore
        if checker_fn:
            equations = checker_fn(name, value, annotation, equations, ox)
        else:
            equations = check_annotation(name, value, origin, equations, ox)
    elif inspect.isclass(annotation):

        subclass_callable: bool = issubclass(annotation, Callable)  # type: ignore
        has_args: bool = hasattr(annotation, "__args__")

        if issubclass(annotation, Tuple):  # type: ignore[arg-type]
            equations = check_tuple(name, value, annotation, equations, ox)
        elif issubclass(annotation, dict) and hasattr(annotation, "__annotations__"):
            equations = check_typed_dict(name, value, annotation, equations, ox)
        elif ox.check_non_asta_types:
            if issubclass(annotation, (float, complex)):
                check_number(name, value, annotation, ox)
            elif subclass_callable and has_args:
                # Needed on Python 3.5.0 to 3.5.2
                check_callable(name, value, annotation, ox)
            elif issubclass(annotation, IO):
                check_io(name, value, annotation, ox)
            elif getattr(annotation, "_is_protocol", False):
                check_protocol(name, value, annotation, ox)
            else:
                annotation = (
                    getattr(annotation, "__extra__", None) or origin or annotation
                )

                # As per https://github.com/python/typing/issues/552
                if annotation is bytes:
                    annotation = (bytearray, bytes)

                # Handles ``str`` and ``int``.
                if not isinstance(value, annotation):
                    fail_fallback(
                        name, qualified_name(annotation), qualified_name(value), ox
                    )

    return equations
