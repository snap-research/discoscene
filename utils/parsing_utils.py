# python3.7
"""Contains the utility functions for parsing arguments."""

import json
import argparse
import click

__all__ = [
    'parse_int', 'parse_float', 'parse_bool', 'parse_index', 'parse_json',
    'IntegerParamType', 'FloatParamType', 'BooleanParamType', 'IndexParamType',
    'JsonParamType', 'DictAction'
]


def parse_int(arg):
    """Parses an argument to integer.

    Support converting string `none` and `null` to `None`.
    """
    if arg is None:
        return None
    if isinstance(arg, str) and arg.lower() in ['none', 'null']:
        return None
    return int(arg)


def parse_float(arg):
    """Parses an argument to float number.

    Support converting string `none` and `null` to `None`.
    """
    if arg is None:
        return None
    if isinstance(arg, str) and arg.lower() in ['none', 'null']:
        return None
    return float(arg)


def parse_bool(arg):
    """Parses an argument to boolean.

    `None` will be converted to `False`.
    """
    if isinstance(arg, bool):
        return arg
    if arg is None:
        return False
    if arg.lower() in ['1', 'true', 't', 'yes', 'y']:
        return True
    if arg.lower() in ['0', 'false', 'f', 'no', 'n', 'none', 'null']:
        return False
    raise ValueError(f'`{arg}` cannot be converted to boolean!')


def parse_index(arg, min_val=None, max_val=None):
    """Parses indices.

    If the input is a list or tuple, this function has no effect.

    If the input is a string, it can be either a comma separated list of numbers
    `1, 3, 5`, or a dash separated range `3 - 10`. Spaces in the string will be
    ignored.

    Args:
        arg: The input argument to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this field. (default: None)

    Returns:
        A list of integers.

    Raises:
        ValueError: If the input is invalid, i.e., neither a list or tuple, nor
            a string.
    """
    if arg is None or arg == '':
        indices = []
    elif isinstance(arg, int):
        indices = [arg]
    elif isinstance(arg, (list, tuple)):
        indices = list(arg)
    elif isinstance(arg, str):
        indices = []
        if arg.lower() not in ['none', 'null']:
            splits = arg.replace(' ', '').split(',')
            for split in splits:
                numbers = list(map(int, split.split('-')))
                if len(numbers) == 1:
                    indices.append(numbers[0])
                elif len(numbers) == 2:
                    indices.extend(list(range(numbers[0], numbers[1] + 1)))
    else:
        raise ValueError(f'Invalid type of input: `{type(arg)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices


def parse_json(arg):
    """Parses a string-like argument following JSON format.

    - `None` arguments will be kept.
    - Non-string arguments will be kept.
    """
    if not isinstance(arg, str):
        return arg
    try:
        return json.loads(arg)
    except json.decoder.JSONDecodeError:
        return arg


class IntegerParamType(click.ParamType):
    """Defines a `click.ParamType` to parse integer arguments."""

    name = 'int'

    def convert(self, value, param, ctx):  # pylint: disable=inconsistent-return-statements
        try:
            return parse_int(value)
        except ValueError:
            self.fail(f'`{value}` cannot be parsed as an integer!', param, ctx)


class FloatParamType(click.ParamType):
    """Defines a `click.ParamType` to parse float arguments."""

    name = 'float'

    def convert(self, value, param, ctx):  # pylint: disable=inconsistent-return-statements
        try:
            return parse_float(value)
        except ValueError:
            self.fail(f'`{value}` cannot be parsed as a float!', param, ctx)


class BooleanParamType(click.ParamType):
    """Defines a `click.ParamType` to parse boolean arguments."""

    name = 'bool'

    def convert(self, value, param, ctx):  # pylint: disable=inconsistent-return-statements
        try:
            return parse_bool(value)
        except ValueError:
            self.fail(f'`{value}` cannot be parsed as a boolean!', param, ctx)


class IndexParamType(click.ParamType):
    """Defines a `click.ParamType` to parse indices arguments."""

    name = 'index'

    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def convert(self, value, param, ctx):  # pylint: disable=inconsistent-return-statements
        try:
            return parse_index(value, self.min_val, self.max_val)
        except ValueError:
            self.fail(
                f'`{value}` cannot be parsed as a list of indices!', param, ctx)


class JsonParamType(click.ParamType):
    """Defines a `click.ParamType` to parse arguments following JSON format."""

    name = 'json'

    def convert(self, value, param, ctx):
        return parse_json(value)


class DictAction(argparse.Action):
    """Argparse action to split each argument into (key, value) pair.

    Each argument should be with `key=value` format, where `value` should be a
    string with JSON format.

    For example, with an argparse:

    parser.add_argument('--options', nargs='+', action=DictAction)

    , you can use following arguments in the command line:

    --options \
        a=1 \
        b=1.5
        c=true \
        d=null \
        e=[1,2,3,4,5] \
        f='{"x":1,"y":2,"z":3}' \

    NOTE: No space is allowed in each argument. Also, the dictionary-type
    argument should be quoted with single quotation marks `'`.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for argument in values:
            key, val = argument.split('=', maxsplit=1)
            options[key] = parse_json(val)
        setattr(namespace, self.dest, options)
