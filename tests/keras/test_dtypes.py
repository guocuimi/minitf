import pytest

import minitf as tf

tf.random.set_seed(10)

_supported_dtypes = [
    # type, size, name
    (tf.float16, 2, "float16"),
    (tf.float32, 4, "float32"),
    (tf.float64, 8, "float64"),
    (tf.int8, 1, "int8"),
    (tf.int16, 2, "int16"),
    (tf.int32, 4, "int32"),
    (tf.int64, 8, "int64"),
    (tf.uint8, 1, "uint8"),
    (tf.uint16, 2, "uint16"),
    (tf.uint32, 4, "uint32"),
    (tf.uint64, 8, "uint64"),
    (tf.bool, 1, "bool"),
]


@pytest.mark.parametrize('type, size, str', _supported_dtypes)
def test_dtypes(type, size, str):
    dt = tf.dtype(type)
    assert dt.itemsize == size
    assert dt.name == str
    assert dt.type == type

    dt = tf.dtype(str)
    assert dt.itemsize == size
    assert dt.name == str
    assert dt.type == type
