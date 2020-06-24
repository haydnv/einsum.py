import numpy as np

from collections import OrderedDict
from itertools import product


VALID_LABELS = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))


class LabeledTensor(object):
    def __init__(self, labels, tensor):
        assert len(labels) == tensor.ndim
        self.labels = labels
        self.tensor = tensor

    def __getitem__(self, coord):
        if len(coord) == 1 and list(coord.keys())[0] == self.labels[0]:
            return self.tensor[list(coord.values())[0]]
        else:
            axis_coord = tuple(coord[l] if l in coord else slice(None) for l in self.labels)
            return self.tensor[axis_coord]


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

    tensors = [LabeledTensor(f_inputs[i], tensors[i]) for i in range(len(tensors))]
    dimensions = OrderedDict()
    for t in tensors:
        for l, d in zip(t.labels, t.tensor.shape):
            if l not in dimensions:
                dimensions[l] = d

    op_labels = list(dimensions.keys())
    op = np.ones(list(dimensions.values()))
    for coord in product(*[range(d) for d in dimensions.values()]):
        for t in tensors:
            selector = {op_labels[i]: coord[i] for i in range(len(op_labels))}
            op[coord] *= t[selector]

    return LabeledTensor(op_labels, op)


def contract(op, dimensions, f_output):
    f_input = list(dimensions.keys())
    axes = [l for l in f_output]
    contraction = np.zeros([dimensions[l] for l in axes])

    for coord in product(*[range(dimensions[l]) for l in axes]):
        selector = dict((axes[i], coord[i]) for i in range(len(axes)))
        sources = [
            [selector[l]] if l in selector else range(dimensions[l])
            for l in f_input]

        value = 0
        for source_coord in product(*sources):
            op_coord = {f_input[i]: source_coord[i] for i in range(len(source_coord))}
            value += op[op_coord]

        contraction[coord] = value

    return contraction


def einsum(f, *tensors):
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(f_inputs, tensors)

    op = outer_product(f_inputs, tensors)
    contraction = contract(op, dimensions, f_output)
    return contraction

