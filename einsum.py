import numpy as np

from collections import OrderedDict
from itertools import product


VALID_LABELS = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))


class OuterProduct(object):
    def __init__(self, tensors, f_inputs):
        self._tensors = tensors
        self._f_inputs = f_inputs

    def at(self, selector):
        value = 1
        for t in range(len(self._tensors)):
            coord = tuple(selector[l] for l in self._f_inputs[t])
            value *= self._tensors[t][coord]
            if not value:
                return value

        return value


def contract(op, dimensions, axes):
    f_input = list(dimensions.keys())
    contraction = np.zeros([dimensions[l] for l in axes])

    for coord in product(*[range(dimensions[l]) for l in axes]):
        selector = dict((axes[i], coord[i]) for i in range(len(axes)))
        sources = [
            [selector[l]] if l in selector else range(dimensions[l])
            for l in f_input]
        sources = product(*sources)

        value = 0
        for source_coord in sources:
            op_coord = {f_input[i]: source_coord[i] for i in range(len(source_coord))}
            value += op.at(op_coord)

        contraction[coord] = value

    return contraction


def parse_format(f):
    if '->' not in f:
        raise ValueError

    f_inputs, f_output = f.split('->')

    if not f_inputs:
        raise ValueError

    f_inputs = [list(f) for f in f_inputs.split(',')]
    f_output = list(f_output)

    assert len(set(f_output)) == len(f_output)

    for f_input in f_inputs:
        if set(f_input) > VALID_LABELS:
            raise ValueError

    return f_inputs, f_output


def validate_args(tensors, f_inputs):
    assert len(tensors) == len(f_inputs)

    dimensions = OrderedDict()
    for t in range(len(tensors)):
        fmt = f_inputs[t]
        assert tensors[t].ndim == len(fmt)

        for i in range(len(fmt)):
            if fmt[i] in dimensions:
                assert dimensions[fmt[i]] == tensors[t].shape[i]
            else:
                dimensions[fmt[i]] = tensors[t].shape[i]

    return dimensions


def einsum(f, *tensors):
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(tensors, f_inputs)

    op = OuterProduct(tensors, f_inputs)
    axes = [l for l in f_output]
    return contract(op, dimensions, axes)

