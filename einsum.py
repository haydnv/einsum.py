import itertools
import numpy as np

from collections import OrderedDict

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

    labels = list(dimensions.keys())

    ip = inner_product(dimensions, tensors, f_inputs)
    contracted, f_contracted = contract(ip, labels, f_output)

    if not f_output:
        return contracted

    output = np.zeros([dimensions[l] for l in f_output])
    for coord in itertools.product(*[range(dimensions[l]) for l in f_output]):
        selector = {f_output[i]: coord[i] for i in range(len(f_output))}
        output[coord] = select(selector, contracted, f_contracted)

    return output

def contract(tensor, f_input, f_output):
    while set(f_input) > set(f_output):
        for i in range(tensor.ndim):
            if f_input[i] not in f_output:
                tensor = tensor.sum(i)
                del f_input[i]
                break

    return tensor, f_input

def inner_product(dimensions, tensors, f_inputs):
    output = np.ones(list(dimensions.values()))
    f_output = list(dimensions.keys())

    for coord in itertools.product(*[range(d) for d in dimensions.values()]):
        selector = {f_output[i]: coord[i] for i in range(len(f_output))}

        values = []
        for t in range(len(tensors)):
            s = tuple(selector[l] if l in f_output else slice(None) for l in f_inputs[t])
            values.append(tensors[t][tuple(s)])

        output[coord] = np.product(values)

    return output

def select(coord, tensor, fmt):
    assert tensor.ndim == len(fmt)
    axes = tuple(coord.get(fmt[i], slice(None)) for i in range(tensor.ndim))
    return tensor[axes]

