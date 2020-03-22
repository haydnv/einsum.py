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

def validate_args(args, f_args):
    labels = {}
    for arg in range(len(args)):
        fmt = f_args[arg]
        assert args[arg].ndim == len(fmt)
        for i in range(len(fmt)):
            if fmt[i] in labels:
                assert labels[fmt[i]] == args[arg].shape[i]
            else:
                labels[fmt[i]] = args[arg].shape[i]

def einsum(f, *args, dtype=np.int32):
    args = list(args)
    f_inputs, f_output = parse_format(f)
    assert len(args) == len(f_inputs)
    validate_args(args, f_inputs)

    dimensions = OrderedDict()
    for arg in range(len(args)):
        for i in range(len(f_inputs[arg])):
            label = f_inputs[arg][i]
            dim = args[arg].shape[i]
            if label in dimensions:
                assert dimensions[label] == dim
            else:
                dimensions[label] = dim

    labels = list(dimensions.keys())

    ip = inner_product(dimensions, args, f_inputs)
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

