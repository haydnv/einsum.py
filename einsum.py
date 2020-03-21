import itertools
import numpy as np

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

def einsum(f, *args, dtype=np.int32):
    f_inputs, f_output = parse_format(f)
    assert len(args) == len(f_inputs)

    labels = {}
    for arg in range(len(args)):
        for i in range(len(f_inputs[arg])):
            label = f_inputs[arg][i]
            dim = args[arg].shape[i]
            if label in labels:
                assert labels[label] == dim
            else:
                labels[label] = dim

    output_dims = [labels[l] for l in f_output] if f_output else [1]
    output = np.ones(output_dims, dtype)

    for coord in itertools.product(*[range(labels[l]) for l in f_output]):
        coord = {f_output[i]: coord[i] for i in range(len(f_output))}
        select_and_assign(coord, f_inputs, args, f_output, output)

    return output

def select_and_assign(coord, f_inputs, inputs, f_output, output):
    assert set(coord.keys()) == set(f_output)

    output_selector = tuple(coord[l] for l in f_output)

    value = 1
    for arg in range(len(inputs)):
        selection = select(coord, f_inputs[arg], inputs[arg])
        value = value * selection

    output[output_selector] = np.sum(value)

def select(coords, fmt, tensor):
    selector = [slice(None)] * tensor.ndim
    for i in range(tensor.ndim):
        if fmt[i] in coords:
            selector[i] = coords[fmt[i]]

    return tensor[tuple(selector)]

