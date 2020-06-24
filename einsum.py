import numpy as np

from collections import OrderedDict
from itertools import product


VALID_LABELS = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))


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
        elif len(set(f_input)) < len(f_input):
            raise ValueError("duplicate label {}".format(f_input))

    return f_inputs, f_output


def validate_args(f_inputs, tensors):
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


def outer_product(f_inputs, tensors):
    assert len(f_inputs) == len(tensors)

    dimensions = OrderedDict()
    for i in range(len(tensors)):
        for l, d in zip(f_inputs[i], tensors[i].shape):
            if l not in dimensions:
                dimensions[l] = d

    op_labels = list(dimensions.keys())
    op = np.ones(list(dimensions.values()))
    for i in range(len(tensors)):
        for coord in product(*[range(d) for d in dimensions.values()]):
            selector = dict(zip(op_labels, coord))
            selector = tuple(selector[l] for l in f_inputs[i])
            op[coord] *= tensors[i][selector]

    return op


def contract(op, dimensions, f_output):
    f_input = list(dimensions.keys())
    axes = [l for l in f_output]
    contraction = np.zeros([dimensions[l] for l in axes])

    for coord in product(*[range(dimensions[l]) for l in axes]):
        selector = dict((axes[i], coord[i]) for i in range(len(axes)))
        axis = tuple(selector[l] if l in selector else slice(None) for l in f_input)
        contraction[coord] = np.sum(op[axis])

    return contraction


def einsum(f, *tensors):
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(f_inputs, tensors)

    op = outer_product(f_inputs, tensors)
    contraction = contract(op, dimensions, f_output)
    return contraction

