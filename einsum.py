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


def outer_product(f_inputs, dimensions, tensors):
    tensors = list(tensors)
    assert len(f_inputs) == len(tensors)
    f_output = list(dimensions.keys())

    normalized = []

    while tensors:
        tensor = tensors.pop()
        labels = f_inputs.pop()

        if labels == f_output:
            normalized.append(tensor)
            continue

        source = dict(zip(labels, range(len(labels))))
        permutation = [source[l] for l in f_output if l in labels]
        labels = [labels[axis] for axis in permutation]
        tensor = np.transpose(tensor, permutation)

        i = 0
        while i < len(dimensions):
            if i == len(labels) or labels[i] != f_output[i]:
                tensor = np.expand_dims(tensor, i)
                labels.insert(i, f_output[i])
            else:
                i += 1

        normalized.append(tensor)

    op = normalized.pop()
    while normalized:
        op = op * normalized.pop()

    return op


def contract(op, dimensions, f_output):
    f_input = list(dimensions.keys())
    axis = 0
    while op.ndim > len(f_output):
        assert len(f_input) == op.ndim
        if f_input[axis] not in f_output:
            op = np.sum(op, axis)
            del f_input[axis]
        else:
            axis += 1

    if f_input == f_output:
        return op
    else:
        source = dict(zip(f_input, range(len(f_input))))
        permutation = [source[l] for l in f_output]
        return np.transpose(op, permutation)


def einsum(f, *tensors):
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(f_inputs, tensors)

    op = outer_product(f_inputs, dimensions, tensors)
    contraction = contract(op, dimensions, f_output)
    return contraction

