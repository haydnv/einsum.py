import numpy as np

from einsum import einsum

def test(fmt, *args, dtype=np.int32):
    expected = np.einsum(fmt, *args)
    actual = einsum(fmt, *args, dtype=dtype)
    test_passed = np.allclose(expected, actual)

    if not test_passed:
        print("for input: {}".format(fmt))
        print("with args:")
        print()
        for arg in args:
            print(arg)
            print()

        print("expected")
        print(expected)
        print("actual")
        print(actual)
        print()

    assert test_passed

if __name__ == "__main__":
    A = np.array([1, 2, 3])
    test('i->', A)
    test('i->i', A)

    A = np.array([[1, 1], [2, 2], [3, 3]])
    test('ij->i', A)
    test('ij->j', A)
    test('ij->ij', A)
    test('ij->ji', A)

    A = np.array([[1, 1], [2, 2]])
    test('ij,kl->', A, A)

    A = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [5, 5, 5]])

    B = np.array([[0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 1]])

    test('ij,jk->ijk', A, B)
    test('ij,jk->ik', A, B)

