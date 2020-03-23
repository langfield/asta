#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions for checking type annotations and their origin types. """
import inspect
import collections
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
    TypeVar,
    Sequence,
)

from sympy.core.expr import Expr

import asta.dims
from asta.dims import Placeholder
from asta.utils import shapecheck, attrcheck
from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.display import (
    type_representation,
    pass_argument,
    pass_return,
    fail_argument,
    fail_return,
    fail_tuple,
    fail_tuple_length,
    fail_namedtuple,
    fail_empty_tuple,
    fail_uninitialized,
    fail_list,
    fail_sequence,
    fail_dict,
    fail_set,
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


def qualified_name(obj: Any) -> str:
    """
    Return the qualified name (e.g. package.module.Type) for the given object.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    type_ = obj if inspect.isclass(obj) else type(obj)
    module = type_.__module__
    qualname: str = type_.__qualname__
    return qualname if module in ("typing", "builtins") else f"{module}.{qualname}"


def check_tuple(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
) -> Set[Expr]:
    """ Check an argument with annotation ``tuple`` or ``Tuple[]``. """

    # Specialized check for NamedTuples.
    if hasattr(annotation, "_field_types"):
        if not isinstance(value, annotation):
            fail_namedtuple(name, qualified_name(annotation), qualified_name(value))

        # pylint: disable=protected-access
        for subname, field_type in annotation._field_types.items():
            equations = check_annotation(
                f"{name}.{subname}", getattr(value, subname), field_type, equations
            )

    elif not isinstance(value, tuple):
        fail_tuple(name, qualified_name(value))

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
            element_type = tuple_params[0]
            for i, element in enumerate(value):
                equations = check_annotation(
                    f"{name}[{i}]", element, element_type, equations
                )
        elif tuple_params == ((),):
            if value != ():
                fail_empty_tuple(name, qualified_name(value))
        else:
            if len(value) != len(tuple_params):
                ann_rep = qualified_name(annotation)
                fail_tuple_length(name, len(tuple_params), ann_rep, len(value))

            for i, (elem, elem_type) in enumerate(zip(value, tuple_params)):
                equations = check_annotation(f"{name}[{i}]", elem, elem_type, equations)

    return equations


def check_list(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
) -> Set[Expr]:
    """ Check an argument with annotation ``list`` or ``List[]``. """
    if not isinstance(value, list):
        fail_list(name, annotation, qualified_name(value))

    # If annotation is a subscriptable generic (``List[...]``).
    if annotation is not list:
        if annotation.__args__ not in (None, annotation.__parameters__):
            value_type = annotation.__args__[0]
            if value_type is not Any:

                # TODO: Consider adding a configuration option to avoid
                # checking the entire list, maybe just do the first and last
                # elements, or an intelligent sample.
                for i, v in enumerate(value):
                    element_equations = check_annotation(
                        f"{name}[{i}]", v, value_type, equations
                    )
                    equations = equations.union(element_equations)

    return equations


def check_sequence(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
) -> Set[Expr]:
    """ Check an argument with annotation ``Sequence[*]``. """
    if not isinstance(value, collections.abc.Sequence):
        fail_sequence(name, annotation, qualified_name(value))

    # Consider removing this test or even just figuring out what it does?
    if annotation.__args__ not in (None, annotation.__parameters__):
        value_type = annotation.__args__[0]
        if value_type is not Any:

            # TODO: Consider not iterating over entire sequence.
            for i, v in enumerate(value):
                element_equations = check_annotation(
                    f"{name}[{i}]", v, value_type, equations
                )
                equations = equations.union(element_equations)

    return equations


def check_dict(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
) -> Set[Expr]:
    """ Check an argument with annotation ``Dict[*]``. """
    if not isinstance(value, dict):
        fail_dict(name, annotation, qualified_name(value))

    # Equivalently: if annotation is ``Dict[*]``.
    if annotation is not dict:
        if annotation.__args__ not in (None, annotation.__parameters__):
            key_type, value_type = annotation.__args__
            if key_type is not Any or value_type is not Any:
                for k, v in value.items():
                    key_equations = check_annotation(
                        f"{name}.<key>", k, key_type, equations
                    )
                    val_equations = check_annotation(
                        f"{name}[{k}]", v, value_type, equations
                    )

                    equations = equations.union(key_equations, val_equations)

    return equations


def check_set(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
) -> Set[Expr]:
    """ Check an argument with annotation ``Set[*]``. """
    if not isinstance(value, AbstractSet):
        fail_set(name, annotation, qualified_name(value))

    # Equivalently: if annotation is ``Set[*]``.
    if annotation is not set:
        if annotation.__args__ not in (None, annotation.__parameters__):
            value_type = annotation.__args__[0]
            if value_type is not Any:
                for v in value:
                    set_equations = check_annotation(
                        f"{name}.<set_element>", v, value_type, equations
                    )
                    equations = equations.union(set_equations)

    return equations


def refresh(annotation: SubscriptableMeta) -> Tuple[SubscriptableMeta, bool]:
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
                        fail_uninitialized(name)
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


def get_equations(
    equations: Set[Expr], annotation: SubscriptableMeta, arg: Any,
) -> Set[Expr]:
    """ Computes subscript argument equations via shapecheck/attrcheck calls. """
    shape_equations: Set[Expr] = set()
    attr_equations: Set[Expr] = set()

    if annotation.shape is not None:

        # HARDCODE
        # Handle case where type(shape) != tuple, e.g. ``torch.Size``.
        arg_shape = tuple(arg.shape)
        assert not isinstance(arg_shape, torch.Size)

        # Grab equations from shapecheck call.
        shape_match, shape_equations = shapecheck(arg_shape, annotation.shape)
        attr_match, attr_equations = attrcheck(arg, annotation.kwattrs)
        assert shape_match and attr_match

    equations = equations.union(shape_equations, attr_equations)

    return equations


# Equality checks are applied to these.
ORIGIN_TYPE_CHECKERS = {
    AbstractSet: check_set,
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
}


def check_annotation(
    name: str, value: Any, annotation: Any, equations: Set[Expr]
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

    # Determine which display functions to use.
    if name == "return":
        fail_fn = fail_return
        pass_fn = pass_return
    else:
        fail_fn = fail_argument
        pass_fn = pass_argument

    # Get origin type.
    origin: Any = getattr(annotation, "__origin__", None)
    _subclass_check_unions = hasattr(Union, "__union_set_params__")

    # Only check if the annotation is an asta subscriptable class.
    # TODO: Consider moving this into a ``check_asta`` function.
    if isinstance(annotation, SubscriptableMeta):
        annotation, initialized = refresh(annotation)
        if not initialized:
            return equations

        # Check if the literal ``value`` matches the annotation.
        rep: str = type_representation(value)
        # HARDCODE
        identifier = "0" if name == "return" else name

        # If the isinstance check fails, print/raise an error.
        if not isinstance(value, annotation):
            fail_fn(identifier, annotation, rep)

        # Otherwise, print a pass.
        else:
            pass_fn(identifier, annotation, rep)

            # Update variable dimension map.
            equations = get_equations(equations, annotation, value)

    elif origin is not None:
        checker_fn = ORIGIN_TYPE_CHECKERS[origin]
        if checker_fn:
            equations = checker_fn(name, value, annotation, equations)
        else:
            equations = check_annotation(name, value, origin, equations)
    elif inspect.isclass(annotation):

        subclass_callable: bool = issubclass(annotation, Callable)  # type: ignore
        subclass_union: bool = issubclass(annotation, Union)  # type: ignore
        has_args: bool = hasattr(annotation, "__args__")

        if issubclass(annotation, Tuple):  # type: ignore[arg-type]
            # TODO: Merge equations.
            tuple_equations = check_tuple(name, value, annotation, equations)
        elif subclass_callable and has_args:
            # Needed on Python 3.5.0 to 3.5.2
            # check_callable(name, value, annotation, memo)
            pass
        elif issubclass(annotation, (float, complex)):
            # check_number(name, value, annotation)
            pass
        elif _subclass_check_unions and subclass_union:
            # check_union(name, value, annotation, memo)
            pass
        elif isinstance(annotation, TypeVar):  # type: ignore[arg-type]
            # check_typevar(name, value, annotation, memo)
            pass
        elif issubclass(annotation, IO):
            # check_io(name, value, annotation)
            pass
        elif issubclass(annotation, dict) and hasattr(annotation, "__annotations__"):
            # check_typed_dict(name, value, annotation, memo)
            pass
        elif getattr(annotation, "_is_protocol", False):
            # check_protocol(name, value, annotation)
            pass
        else:
            annotation = getattr(annotation, "__extra__", None) or origin or annotation

            if annotation is bytes:
                # As per https://github.com/python/typing/issues/552
                annotation = (bytearray, bytes)

            if not isinstance(value, annotation):
                # TODO: Write fallback fail function.
                """
                raise TypeError(
                    'type of {} must be {}; got {} instead'.
                    format(name, qualified_name(annotation), qualified_name(value)))
                """

    return equations
