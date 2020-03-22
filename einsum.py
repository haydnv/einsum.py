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
            s = tuple(selector.get(l, slice(None)) for l in self._f_inputs[t])
            value *= self._tensors[t][s]

        return value

class Contraction(object):
    def __init__(self, op, dimensions, f_output):
        self._op = op
        self._dimensions = dimensions
        self._f_input = list(dimensions.keys())

    def at(self, selector):
        sources = product(*[
            [selector[l]] if l in selector else range(self._dimensions[l])
            for l in self._f_input])

        value = 0
        for coord in sources:
            value += self._op.at({self._f_input[i]: coord[i] for i in range(len(coord))})

        return value

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

def einsum(f, *tensors, dtype=np.int32):
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(tensors, f_inputs)

    output = np.zeros([dimensions[l] for l in f_output])
    op = OuterProduct(tensors, f_inputs)
    contraction = Contraction(op, dimensions, f_output)
    for coord in product(*[range(dimensions[l]) for l in f_output]):
        selector = {f_output[i]: coord[i] for i in range(len(f_output))}
        output[coord] = contraction.at(selector)

    return output

