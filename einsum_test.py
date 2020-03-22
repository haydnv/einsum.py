import numpy as np

from einsum import einsum

def test(fmt, *args, dtype=np.int32):
    expected = np.einsum(fmt, *[np.array(a) for a in args])
    actual = einsum(fmt, *[np.array(a) for a in args], dtype=dtype)
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

    A = np.array([[1, 2, 3], [4, 5, 6]])
    test('ij,ik->ik', A, A)

    A = np.array([[1, 2], [3, 4]])
    test('ik,ik,il->', A, A, A)

    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[5, 6],
                  [7, 8]])

    test('ij,jk->ijk', A, B)
    test('ij,jk->ij', A, B)
    test('ij,jk->ik', A, B)
    test('ij,jk->jk', A, B)
    test('ij,jk->i', A, B)
    test('ij,jk->j', A, B)
    test('ij,jk->k', A, B)
    test('ij,jk->', A, B)

    A = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [5, 5, 5]])

    B = np.array([[0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 1]])

    test('ij,jk->ijk', A, B)
    test('ij,jk->ik', A, B)

    A = np.arange(60.).reshape(3, 4, 5)
    B = np.arange(24.).reshape(4, 3, 2)
    test('ijk,jil->kl', A, B)

    print("all tests passed")

