import numpy as np


def clear_cached(obj, properties=None):
    if properties is None:
        properties = []
    elif isinstance(properties, str):
        properties = [properties, ]
    assert type(properties) in (list, tuple), f"properties should be list or tuple, got {type(properties)}"
    for attr in properties:
        if attr in obj.__dict__:
            del obj.__dict__[attr]


# Global cache to store einsum paths
EINSUM_PATH_CACHE = {}


def cached_einsum(subscripts, *operands,
                  optimize='greedy',
                  **kwargs):
    """
    A wrapper for np.einsum that caches the contraction path.
    The cache key is a combination of the subscripts string and the
    shapes of the operand arrays.
    """
    shapes = tuple(op.shape for op in operands)
    cache_key = (subscripts, shapes)

    if cache_key in EINSUM_PATH_CACHE:
        path = EINSUM_PATH_CACHE[cache_key]
    else:
        path = np.einsum_path(subscripts, *operands, optimize=optimize)[0]
        EINSUM_PATH_CACHE[cache_key] = path
    return np.einsum(subscripts, *operands, optimize=path, **kwargs)
